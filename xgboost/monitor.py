import psutil
import threading
import time
import GPUtil
import pynvml
import csv
from datetime import datetime

current_time = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
print(current_time)

def monitor(interval=1, filename='outputs/'+current_time+'.csv'):
    start_time = time.time()

    def monitor_cpu():
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['time', 'elapsed_time', 'cpu', 'memory_cpu', 'gpu', 'process']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                elapsed_time = time.time() - start_time
                cpu_usage = psutil.cpu_percent(interval=interval)
                memory_info = psutil.virtual_memory()
                gpu_utilization = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 'N/A'
                
                # Get GPU processes info
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    # process_info = ', '.join([f"PID: {process.pid}, Memory: {process.usedGpuMemory / (1024.0 ** 2)} МБ" for process in processes])
                    process_info = '/'.join([f"{process.usedGpuMemory / (1024.0 ** 2)}" for process in processes])
                except pynvml.NVMLError as e:
                    process_info = 'N/A'
                finally:
                    pynvml.nvmlShutdown()

                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                writer.writerow({
                    'time': current_time,
                    'elapsed_time': elapsed_time, # sec
                    'cpu': cpu_usage, #percent
                    'memory_cpu': memory_info.percent, # percent
                    'gpu': gpu_utilization, # percent
                    'process': process_info # mb
                })
                ###
                print('time:', current_time,
                    'elapsed_time:', elapsed_time, # sec
                    'cpu:', cpu_usage, #percent
                    'memory_cpu:', memory_info.percent, # percent
                    'gpu:', gpu_utilization, # percent
                    'process:', process_info # mb
                    )
                ###
                time.sleep(interval)

    cpu_thread = threading.Thread(target=monitor_cpu)
    cpu_thread.daemon = True
    cpu_thread.start()


### CPU load
def monitor_cpu_background(interval=1):
    def monitor_cpu():
        while True:
            cpu_usage = psutil.cpu_percent(interval=interval)
            print(f"Использование CPU: {cpu_usage}%")
            time.sleep(interval)

    # Создание потока для мониторинга CPU
    cpu_thread = threading.Thread(target=monitor_cpu)

    # Запуск потока в фоновом режиме
    cpu_thread.daemon = True
    cpu_thread.start()


# cpu memory
def monitor_memory_background(interval=1):
    def monitor_memory():
        while True:
            memory_info = psutil.virtual_memory()
            print(f"Использование памяти: {memory_info.percent}%")
            time.sleep(interval)

    # Создание потока для мониторинга памяти
    memory_thread = threading.Thread(target=monitor_memory)

    # Запуск потока в фоновом режиме
    memory_thread.daemon = True
    memory_thread.start()

# time
def monitor_runtime_background(interval=10):
    start_time = time.time()

    def monitor_runtime():
        while True:
            elapsed_time = time.time() - start_time
            print(f"Прошло времени: {elapsed_time:.2f} секунд")
            time.sleep(interval)

    # Создание потока для мониторинга времени выполнения
    runtime_thread = threading.Thread(target=monitor_runtime)

    # Запуск потока в фоновом режиме
    runtime_thread.daemon = True
    runtime_thread.start()

# gpu load
def monitor_gpu_background(interval=5):
    def monitor_gpu():
        while True:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"Использование GPU {i}: {gpu.load * 100}%")
            time.sleep(interval)

    # Создание потока для мониторинга GPU
    gpu_thread = threading.Thread(target=monitor_gpu)

    # Запуск потока в фоновом режиме
    gpu_thread.daemon = True
    gpu_thread.start()

# gpu processes
def monitor_gpu_processes_background(interval=10):
    def monitor_gpu_processes():
        while True:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

                if processes:
                    print("Процессы на GPU:")
                    for process in processes:
                        # Получение информации о загрузке GPU
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization = utilization.gpu
                        print(f"PID: {process.pid}, Memory: {process.usedGpuMemory / (1024.0 ** 2)} МБ, Загрузка: {gpu_utilization}%")
                else:
                    print("Нет активных процессов на GPU")

            except pynvml.NVMLError as e:
                print(f"Ошибка при получении информации о GPU: {e}")
                
            finally:
                pynvml.nvmlShutdown()

            time.sleep(interval)

    gpu_processes_thread = threading.Thread(target=monitor_gpu_processes)
    gpu_processes_thread.daemon = True
    gpu_processes_thread.start()
