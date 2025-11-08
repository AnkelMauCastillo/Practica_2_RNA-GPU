import threading
import time
import psutil

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.gpu_usage = 0
        self.gpu_memory_used = 0
        self.gpu_memory_total = 0
        
    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.monitoring = True
            thread = threading.Thread(target=self._monitor)
            thread.daemon = True
            thread.start()
            print(" Monitor GPU activado")
        except Exception as e:
            print(f"  No se pudo activar monitor GPU: {e}")
            self.monitoring = False
        
    def _monitor(self):
        import pynvml
        while self.monitoring:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.gpu_usage = util.gpu
                self.gpu_memory_used = memory.used / 1024**3
                self.gpu_memory_total = memory.total / 1024**3
                
                cpu = psutil.cpu_percent()
                memory_ram = psutil.virtual_memory().percent
                
                print(f" Monitor -> GPU: {self.gpu_usage:2.0f}% | VRAM: {self.gpu_memory_used:5.1f}/{self.gpu_memory_total:2.0f}GB | CPU: {cpu:5.1f}% | RAM: {memory_ram:5.1f}%")
                
            except Exception as e:
                print(f"Error en monitor: {e}")
            time.sleep(5)
            
    def stop(self):
        self.monitoring = False