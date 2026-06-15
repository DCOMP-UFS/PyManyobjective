import time
from contextlib import contextmanager

@contextmanager
def measure_cpu_time(label):
    """
    Gerenciador de contexto para medir o tempo de processador (CPU)
    e o tempo real (Wall) de execução de um bloco de código.
    """
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    yield
    end_cpu = time.process_time()
    end_wall = time.perf_counter()
    cpu_duration = end_cpu - start_cpu
    wall_duration = end_wall - start_wall
    print(f"[{label}] Tempo de CPU: {cpu_duration:.6f}s | Tempo Real (Wall): {wall_duration:.6f}s")
