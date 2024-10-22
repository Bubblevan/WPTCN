import torch
import torch.nn as nn

from .tcnLayers import Stage

class FlattenHead(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        n_features,
        head_dropout=0,
        individual=False,
    ):
        super().__init__()

        self.individual = individual
        self.n_features = n_features

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_features):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(d_input, d_output))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(d_input, d_output)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_features):
                z = self.flattens[i](x[:, i, :, :])  # z: [batch_size x d_model * patch_num]
                z = self.linears[i](z)               # z: [batch_size x d_output]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)            # x: [batch_size x n_features x d_output]
        else:
            # print(f"Before flatten, x shape: {x.shape}")  # [batch_size, n_features, d_model, n_patches]
            x = x.mean(dim=1)  # 对特征维度取平均，形状变为 [batch_size, d_model, n_patches]
            # print(f"After mean over features, x shape: {x.shape}")
            x = self.flatten(x)  # 展平，形状变为 [batch_size, d_model * n_patches]
            # print(f"After flatten, x shape: {x.shape}")
            x = self.linear(x)  # 线性层，输出形状 [batch_size, d_output]
            # print(f"After linear, x shape: {x.shape}")
            x = self.dropout(x)
        return x


class BackboneModernTCN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_predict_features,
        patch_size,
        patch_stride,
        downsampling_ratio,
        ffn_ratio,
        num_blocks: list,
        large_size: list,
        small_size: list,
        dims: list,
        small_kernel_merged: bool = False,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_multi_scale: bool = True,
        individual: bool = False,
        freq: str = "h",
    ):
        super().__init__()

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Linear(patch_size, dims[0])
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(
                        dims[i],
                        dims[i + 1],
                        kernel_size=downsampling_ratio,
                        stride=downsampling_ratio,
                    ),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsampling_ratio

        if freq == "h":
            time_feature_num = 4
        elif freq == "t":
            time_feature_num = 5
        else:
            raise NotImplementedError("time_feature_num should be 4 or 5")

        self.te_patch = nn.Sequential(
            nn.Conv1d(
                time_feature_num,
                time_feature_num,
                kernel_size=patch_size,
                stride=patch_stride,
                groups=time_feature_num,
            ),
            nn.Conv1d(time_feature_num, dims[0], kernel_size=1, stride=1, groups=1),
            nn.BatchNorm1d(dims[0]),
        )

        # backbone
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                ffn_ratio,
                num_blocks[stage_idx],
                large_size[stage_idx],
                small_size[stage_idx],
                dmodel=dims[stage_idx],
                nvars=n_features,
                small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout,
            )
            self.stages.append(layer)

        # Multi scale fusing
        self.use_multi_scale = use_multi_scale
        self.up_sample_ratio = downsampling_ratio

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        for i in range(self.num_stage):
            align_dim = dims[-1]
            lat = nn.Conv1d(dims[i], align_dim, kernel_size=1, stride=1)
            self.lat_layer.append(lat)
            smooth = nn.Conv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)
            up_conv = nn.Sequential(
                nn.ConvTranspose1d(
                    align_dim,
                    align_dim,
                    kernel_size=self.up_sample_ratio,
                    stride=self.up_sample_ratio,
                ),
                nn.BatchNorm1d(align_dim),
            )
            self.up_sample_conv.append(up_conv)

        # head
        patch_num = n_steps // patch_stride

        self.n_features = n_features
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            final_patch_num = patch_num // (downsampling_ratio ** (self.num_stage - 1))
            self.head_nf = d_model * final_patch_num
            # print(f"Adjusted self.head_nf: {self.head_nf}")
            self.head = FlattenHead(
                self.head_nf,
                n_predict_features,
                n_features,
                head_dropout,
                individual,
            )

        else:
            if patch_num % pow(downsampling_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsampling_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsampling_ratio, (self.num_stage - 1)) + 1)

            self.head = FlattenHead(
                self.head_nf,
                n_predict_features,
                n_features,
                head_dropout,
                individual,
            )

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()

    def forward(self, x):
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                x = x.reshape(B, M, 1, -1).squeeze(-2)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 1, 3, 2)

            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                x = self.downsample_layers[i](x)
                _, D_, N_ = x.shape
                x = x.reshape(B, M, D_, N_)


            x = self.stages[i](x)
        x = self.head(x)
        return x