import matplotlib.pyplot as plt

def analyze_results(data):
    plt.scatter(data[:, 2], data[:, 0], alpha=0.5)
    plt.xlabel('B測量基底 (0=Z, 1=X)')
    plt.ylabel('A測量結果')
    plt.title('A觀測 vs B未來測量基底')
    plt.show()
