import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
import time
from ptflops import get_model_complexity_info

def plot_results(train_accuracies, val_accuracies, f1_macros, f1_micros, conf_matrices):
    epochs = range(1, len(train_accuracies) + 1)

    # 绘制准确率随 epoch 变化
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # 绘制F1分数随 epoch 变化
    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1_macros, label='F1 Score (Macro)', marker='o')
    plt.plot(epochs, f1_micros, label='F1 Score (Micro)', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.legend()
    plt.grid(True)

    # 绘制最后一个epoch的混淆矩阵
    plt.subplot(1, 3, 3)
    final_conf_matrix = conf_matrices[-1]
    im = plt.imshow(final_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Final Epoch)')
    plt.colorbar(im)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 添加标签
    num_classes = final_conf_matrix.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # 添加数值标签
    thresh = final_conf_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(final_conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if final_conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


import time
import torch
import logging
from torchinfo import summary
from ptflops import get_model_complexity_info

def calculate_latency(model, device, test_loader):
    logging.info("Calculating latency...")
    model.eval()
    total_time = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            start_time = time.time()
            model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
            num_samples += inputs.size(0)

    latency = total_time / num_samples
    logging.info(f'Latency: {latency:.6f} seconds per sample')
    return latency

def calculate_memory_usage(model, device, test_loader):
    logging.info("Calculating peak memory usage...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                model(inputs)

        peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
        logging.info(f'Peak Memory Usage: {peak_memory_usage:.2f} MB')
        return peak_memory_usage
    else:
        logging.warning('CUDA not available, unable to measure peak memory usage.')
        return None

def calculate_model_complexity(model, input_size):
    logging.info("Calculating model complexity...")
    model_summary = summary(model, input_size=input_size, verbose=0)
    logging.info(f'Model Summary:\n{model_summary}')

    try:
        flops, params = get_model_complexity_info(model, input_size[1:], as_strings=True,
                                                  print_per_layer_stat=False, verbose=False)
        logging.info(f'FLOPs: {flops}')
        logging.info(f'Parameters: {params}')
    except Exception as e:
        logging.error(f'Unable to calculate FLOPs: {e}')
        flops, params = None, None

    return flops, params
