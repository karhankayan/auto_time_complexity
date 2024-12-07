import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from pysr import PySRRegressor

# Algorithm configuration
ALGORITHMS = {
    'binary_search': {
        'function': lambda arr, target: binary_search(arr, target),
        'color': 'blue',
        'theoretical': 'O(log n)',
        'needs_target': True  # Flag for algorithms that need a target value
    },
    'linear_sum': {
        'function': lambda arr, _: linear_sum(arr),
        'color': 'red',
        'theoretical': 'O(n)',
        'needs_target': False
    }
}

# Select which algorithms to run (modify this list to choose algorithms)
SELECTED_ALGORITHMS = ['binary_search', 'linear_sum']

# Operation counting setup
operation_counts = {alg: 0 for alg in ALGORITHMS.keys()}

def count_operations(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global operation_counts
        alg_name = func.__name__
        if alg_name in operation_counts:
            operation_counts[alg_name] = 0
            
        def trace_func(frame, event, arg):
            if event == 'line':
                operation_counts[alg_name] += 1
            return trace_func
            
        sys.settrace(trace_func)
        try:
            result = func(*args, **kwargs)
        finally:
            sys.settrace(None)
        return result, operation_counts.copy()
    return wrapper

@count_operations
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

@count_operations
def linear_sum(arr):
    total = 0
    for num in arr:
        total += num
    return total

def measure_complexity(selected_algorithms):
    input_sizes = np.linspace(10, 1000, 10, dtype=int)
    average_operations = {alg: [] for alg in selected_algorithms}

    for size in input_sizes:
        print(f"Testing input size: {size}")
        arr_sorted = list(range(size))
        target = random.randint(0, size - 1)

        num_runs = 20
        totals = {alg: 0 for alg in selected_algorithms}

        for _ in range(num_runs):
            for key in operation_counts:
                operation_counts[key] = 0
                
            for alg in selected_algorithms:
                if ALGORITHMS[alg]['needs_target']:
                    _, ops = ALGORITHMS[alg]['function'](arr_sorted, target)
                else:
                    _, ops = ALGORITHMS[alg]['function'](arr_sorted, None)
                totals[alg] += ops[alg]

        for alg in selected_algorithms:
            avg = totals[alg] / num_runs
            average_operations[alg].append(avg)
            print(f"  {alg}: Avg Ops = {avg}")

    return input_sizes, average_operations

def fit_complexity_with_pysr(input_sizes, operations, algorithm_name):
    X = input_sizes.reshape(-1, 1)
    y = np.array(operations)
    
    model = PySRRegressor(
        niterations=300,
        binary_operators=['+', '*', '/'],
        unary_operators=['log2'],
        maxsize=10,
        maxdepth=10,
        progress=True
    )
    
    model.fit(X, y)
    print(f"\nComplexity analysis for {algorithm_name}:")
    print(model)
    return model

def plot_complexity_analysis(input_sizes, average_operations, pysr_models, selected_algorithms):
    plt.figure(figsize=(12, 8))
    
    for alg in selected_algorithms:
        plt.scatter(input_sizes, average_operations[alg], 
                   label=f'{alg} (Empirical)',
                   alpha=0.5,
                   color=ALGORITHMS[alg]['color'])
        
        X_smooth = np.linspace(min(input_sizes), max(input_sizes), 200).reshape(-1, 1)
        y_pred = pysr_models[alg].predict(X_smooth)
        plt.plot(X_smooth, y_pred, '--', 
                label=f'{alg} (PySR Fit)',
                color=ALGORITHMS[alg]['color'],
                linewidth=2)

    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Operations')
    plt.title('Algorithm Complexity Analysis')
    plt.legend()
    plt.grid(True)
    
    # Add textbox with theoretical complexities
    theoretical_text = 'Theoretical Complexities:'
    for alg in selected_algorithms:
        theoretical_text += f'\n{alg}: {ALGORITHMS[alg]["theoretical"]}'
    
    plt.text(0.02, 0.98, 
             theoretical_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.show()

def main():
    # Validate selected algorithms
    for alg in SELECTED_ALGORITHMS:
        if alg not in ALGORITHMS:
            raise ValueError(f"Algorithm '{alg}' not found. Available algorithms: {list(ALGORITHMS.keys())}")

    input_sizes, average_operations = measure_complexity(SELECTED_ALGORITHMS)
    
    pysr_models = {}
    for alg in SELECTED_ALGORITHMS:
        pysr_models[alg] = fit_complexity_with_pysr(
            input_sizes,
            average_operations[alg],
            alg
        )
    
    plot_complexity_analysis(input_sizes, average_operations, pysr_models, SELECTED_ALGORITHMS)

if __name__ == "__main__":
    main()