import os
import psutil
import torch
import time
import numpy as np
from threading import Thread

class ResourceMonitor:
    """监控设备资源使用情况"""
    
    def __init__(self, interval=1.0):
        """
        初始化资源监控器
        
        Args:
            interval: 采样间隔(秒)
        """
        self.interval = interval
        self.running = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory = []
        self.bandwidth_usage = []
        self.last_net_io = None
        self.monitor_thread = None
        
    def start(self):
        """启动资源监控"""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            return
            
        self.running = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory = []
        self.bandwidth_usage = []
        self.last_net_io = psutil.net_io_counters()
        
        self.monitor_thread = Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """停止资源监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.interval*2)
            
    def _monitor_resources(self):
        """资源监控线程"""
        while self.running:
            # CPU使用率
            self.cpu_usage.append(psutil.cpu_percent())
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)
            
            # GPU内存 (如果可用)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                self.gpu_memory.append(gpu_memory)
                
            # 网络带宽估计
            current_net_io = psutil.net_io_counters()
            if self.last_net_io:
                bytes_sent = current_net_io.bytes_sent - self.last_net_io.bytes_sent
                bytes_recv = current_net_io.bytes_recv - self.last_net_io.bytes_recv
                bandwidth = (bytes_sent + bytes_recv) / (1024 * 1024 * self.interval)  # MB/s
                self.bandwidth_usage.append(bandwidth)
            self.last_net_io = current_net_io
            
            time.sleep(self.interval)
            
    def get_statistics(self):
        """获取监控统计信息"""
        stats = {
            'cpu': {
                'mean': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max': np.max(self.cpu_usage) if self.cpu_usage else 0,
            },
            'memory': {
                'mean': np.mean(self.memory_usage) if self.memory_usage else 0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0,
            },
            'bandwidth': {
                'mean': np.mean(self.bandwidth_usage) if self.bandwidth_usage else 0,
                'max': np.max(self.bandwidth_usage) if self.bandwidth_usage else 0,
            }
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'mean': np.mean(self.gpu_memory) if self.gpu_memory else 0,
                'max': np.max(self.gpu_memory) if self.gpu_memory else 0,
            }
            
        return stats