import time
import matplotlib.pyplot as plt
import random

# 시간 복잡도 O(1) 알고리즘: 상수 시간 알고리즘
def constant_time(arr):
    return arr[0] if arr else None

# 시간 복잡도 O(n) 알고리즘: 선형 시간 알고리즘
def linear_time(arr):
    total = 0
    for num in arr:
        total += num
    return total

# 시간 복잡도 O(n log n) 알고리즘: 병합 정렬
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 시간 복잡도 O(n^2) 알고리즘: 이중 for 문
def quadratic_time(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(n):
            count += 1
    return count

# 성능 측정을 위한 함수
def measure_time(func, arr):
    start_time = time.time()
    func(arr)
    end_time = time.time()
    return end_time - start_time

# 데이터 생성 및 측정
input_sizes = [10, 100, 1000, 5000, 10000]  # 입력 데이터 크기
algorithms = {
    "O(1)": constant_time,
    "O(n)": linear_time,
    "O(n log n)": merge_sort,
    "O(n^2)": quadratic_time,
}

results = {alg: [] for alg in algorithms}

for size in input_sizes:
    for name, func in algorithms.items():
        arr = [random.randint(0, 10000) for _ in range(size)]
        time_taken = measure_time(func, arr.copy())
        results[name].append(time_taken)

# 시각화
for name, times in results.items():
    plt.plot(input_sizes, times, label=name)

plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity of Various Algorithms')
plt.legend()
plt.grid(True)
plt.show()
