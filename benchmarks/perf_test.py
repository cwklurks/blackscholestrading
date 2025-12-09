
import time
import numpy as np
import matplotlib.pyplot as plt
from models import bs_call_price_vectorized, BlackScholesModel

def naive_bs_call_price_loop(S_array, K, T, r, sigma_array):
    prices = []
    for s, sigma in zip(S_array, sigma_array):
        model = BlackScholesModel(s, K, T, r, sigma)
        prices.append(model.call_price())
    return np.array(prices)

def run_benchmark():
    input_sizes = [1000, 10000, 100000, 1000000]
    vectorized_times = []
    loop_times = []
    
    K = 100.0
    T = 1.0
    r = 0.05
    
    print(f"{'Size':<10} | {'Vectorized (s)':<15} | {'Loop (s)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    for N in input_sizes:
        # Generate random inputs
        S = np.random.uniform(80, 120, N)
        sigma = np.random.uniform(0.1, 0.5, N)
        
        # Benchmark Vectorized
        start = time.time()
        bs_call_price_vectorized(S, K, T, r, sigma)
        vec_time = time.time() - start
        vectorized_times.append(vec_time)
        
        # Benchmark Loop
        start = time.time()
        naive_bs_call_price_loop(S, K, T, r, sigma)
        loop_time = time.time() - start
        loop_times.append(loop_time)
        
        speedup = loop_time / vec_time if vec_time > 0 else 0
        print(f"{N:<10} | {vec_time:<15.5f} | {loop_time:<15.5f} | {speedup:<10.2f}x")
        
    # Generate Chart
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, loop_times, marker='o', label='Naive Loop')
    plt.plot(input_sizes, vectorized_times, marker='o', label='Vectorized (Numpy)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Options')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Black-Scholes Pricing Performance: Vectorized vs Loop')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    output_file = 'bs_performance_benchmark.png'
    plt.savefig(output_file)
    print(f"\nBenchmark chart saved to {output_file}")

if __name__ == "__main__":
    run_benchmark()

