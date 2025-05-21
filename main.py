from quantum_sim import run_quantum_experiment
from analysis_tools import analyze_results
import numpy as np

# 模擬參數
N = 5000  # 數據數量
results = []

for i in range(N):
    basis = np.random.choice(['Z', 'X'])
    a, b = run_quantum_experiment(basis)
    results.append([a, b, 0 if basis == 'Z' else 1])

# 分析結果
analyze_results(np.array(results))
