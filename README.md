2020131013 정준혁 수학과 프로그래밍 프로젝트입니다.
총 4개의 프로젝트(총 코드는 6파일)을 만들었으며, 2개는 기본적인 감을 얻기 위한 수학적 내용을 담은 코드(역행렬 구하기, 파스칼 삼각형 프린트로 프롬트에 출력하기)를 만들고 나머지 2개는 알고리즘의 시간 복잡도에 대한 데이터 분석 1개와 심화로 더 공부해본 알고리즘으로 지하철의 출발역과 도착역을 적으면 환승시간을 2~6분의 랜덤 시간으로 잡고 평균적인 이동시간을 구하는 프로젝트를 만들어 보았습니다. 
프로젝트 하나씩 1) 만들게 된 동기와,  2) 코드 설명, 3) 모델의 input과 output에 대한 발표를 하겠습니다. 

1. _제 첫 프로젝트는 역행렬을 구하는 프로젝트입니다._
   1) 만들게 된 동기 : 역행렬의 존재는 수학의 많은 분야에서 매우 중요하며, 이를 구하기란 꽤나 tedious한 과정을 거쳐야함을 느꼈습니다. 그렇다면, 역행렬을 구해주는 프로그램을 하나 만들어서 편하게 역행렬을 구하고자 첫 프로젝트로 역행렬 구하기를 구현해보았습니다. 또한 수업 시간도중 numpy로 array나 matrix를 다룰 수 있다는 것을 배우고, 이를 구체화해보고 심화적으로 코딩을 해보고 싶었습니다.
   2) 다음은 역행렬 구하기 코드입니다. (따로 첨부되어 있습니다)
                    import numpy as np
                    
                    def get_matrix_input(n):
                        print(f"Please enter the {n}x{n} matrix row by row:")
                        matrix = []
                        for i in range(n):
                            row = list(map(float, input().split()))
                            matrix.append(row)
                        return np.array(matrix)
                    
                    def main():
                        n = int(input("Enter the size of the matrix (n): "))
                        matrix = get_matrix_input(n)
                        # 예외처리로 두 case를 구분
                        try:
                            inverse_matrix = np.linalg.inv(matrix)
                            print("1. The inverse matrix exists.")
                            print("2. The inverse matrix is:")
                            print(inverse_matrix)
                        except np.linalg.LinAlgError:
                            print("1. The inverse matrix does not exist.")
                    
                    if __name__ == "__main__":
                        main()
      - numpy를 import하여 작업할 matrix에 대한 내장함수를 이용합니다
      - 함수는 총 두 개가 있습니다. 하나는 matrix의 input을 받아올 get_matrix_input()과 메인 함수가 있습니다.
      - 역행렬이 존재하기 위해선 우선, square matrix여야 하므로, input으로는 n(int) 하나를 입력 받습니다. 이후, get_matrix_input() 함수를 통해서 n을 입력받고 nxn 행렬의 원소를 각각 space를 기준으로 입력하면 행렬이 준비됩니다.
      - main()함수는 Error에 관련된 수업을 듣고, 수정을 하여 훨씬 간단하게 구현해보았습니다. 메인 함수의 동작을 설명하겠습니다. 우선, nxn 행렬에서 n이 될 자연수 하나를 입력받습니다. 이때 파이썬은 conversion을 int(input(...)) 형식으로 n의 타입을 바로 int로 convert할 수 있음을 이용했습니다. 이후 get_matrix_input() 함수를 호출하여 nxn 행렬을 작성하고 역행렬을 만들 준비를 끝냈습니다.
      - try, except 코드는 우선, numpy의 내장된 함수를 활용하여 nxn 행렬의 역행렬을 구해봅니다. 이때, 역행렬이 구해졌으면, print("1. The inverse matrix exists.")부터 코드가 실행되며 역행렬을 출력합니다. 만약, 역행렬이 존재하지 않는 행렬을 입력했다면, np.linalg.LinAlgError가 발생해서, except구문의 print("1. The inverse matrix does not exist.")를 실행하고 종료합니다.

        
   3) 다음은 모델의 input과 output이 어떻게 입력되고 출력되는지 구체적인 결과값입니다. Colab에서 실행하였습니다.
     
<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/d2459a7a-e13b-4f45-8e4a-5a17d181b936">
input으로 3x3행렬을 주었으며 역행렬이 존재하지 않습니다

<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/d3eaacba-7774-44ca-9dad-b9469b53968d">
input으로 2x2행렬을 주었으며 역행렬이 존재합니다

<img width="460" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/3c0ddacd-ee28-43a8-9fc3-479654668287">
input으로 4x4행렬을 주었으며 역행렬이 존재합니다


이처럼, 역행렬의 결과가(존재한다면) 정확하게 구해짐을 확인할 수 있습니다.

2. _제 두 번째 프로젝트는 파스칼 삼각형을 출력하는 프로그램입니다._
   1) 만들게 된 동기 : 이산수학, 조합론 등 파스칼 삼각형은 조합(combination)에 대한 중요한 정보를 알려주며 제 개인적으로 가장 아름다운 삼각형이라고 생각합니다. 1번 프로젝트에서는 프롬트에 결괏값을 출력했다면, matplotlib의 내장함수를 활용해서 이번에는 그림을 그려주는 프로그램을 짜보는 게 어떨까 해서, 그저 아무 데이터를 보여주는 것보다, 의미 있는 데이터를 출력해보고자 파스칼 삼각형을 생각하게 되었습니다. 이 프로젝트는 matplotlib에 대한 이해도를 매우 높여줬으며 코드만으로 그림까지 출력함으로써, 프로그래밍만으로 데이터의 가시화를 해볼 수 있는 경험이었습니다.
   2) 다음은 파스칼삼각형 코드입니다. (따로 첨부되어 있습니다)

                  import matplotlib.pyplot as plt
                  
                  def generate_pascals_triangle(height):
                      triangle = []
                      for i in range(height):
                          row = [1] * (i + 1)
                          for j in range(1, i):
                              row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
                          triangle.append(row)
                      return triangle
                  
                  def plot_pascals_triangle(triangle):
                      fig, ax = plt.subplots(figsize=(1, 1))  # 그림 크기를 5x5 인치로 설정
                      ax.axis('off')
                      
                      for i, row in enumerate(triangle):
                          for j, num in enumerate(row):
                              ax.text(j - i / 2, -i, str(num), ha='center', va='center', fontsize=20)  # 글자 크기를 크게 설정
                      
                      plt.gca().invert_yaxis()
                      plt.show()
                  
                  def main():
                      height = int(input("Enter the height of Pascal's triangle: "))
                      triangle = generate_pascals_triangle(height)
                      plot_pascals_triangle(triangle)
                  
                  if __name__ == "__main__":
                      main()
      - 그래프, 등 데이터를 도식화하는 데에 많이 쓰이는 matplotlib의 pyplot을 사용하였습니다.
      - 함수는 총 3개로, generate_pascals_triangle(height), plot_pascals_triangle(triangle), 메인 함수가 있습니다.
      - generate_pascals_triangle(height)는 height을 input으로 받아, 배열 구조를 활용하여 파스칼 삼각형을 생성합니다.
      - plot_pascals_triangle(triangle)은 위 함수로 만들어진 파스칼 삼각형을 도식화하는 함수입니다.
      - 그림 크기와 글자 크기의 설정에 있어 colab에 실행했을 때, 최적의 그림을 얻을만한 size들로 설정을 해놨습니다. 이때 for문에서 enumerate를 사용하였습니다. 이는 각 루프마다 인덱스와 요소를 반환하여 코드를 간결하게 할 수 있었습니다. ax.text는 도식화할 때 축에 텍스트를 추가하는데 활용이 되어 유용하게 활용했습니다.
      - 메인함수의 실행 : 파스칼 삼각형의 높이를 input으로 받습니다. 이후 파스칼 삼각형을 만들어 내고, 마지막으로 그림으로 도식화합니다.
     
   4) 다음은 모델의 input과 output이 어떻게 입력되고 출력되는지 구체적인 결과값입니다. Colab에서 실행하였습니다.



<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://github.com/joondoogg/Project/assets/146046027/5cefb801-b0dc-4d89-accb-5d86dc1195c3" alt="Pascal Triangle Height 3" width="300">
  <img src="https://github.com/joondoogg/Project/assets/146046027/2813b148-0ba1-4f01-ba91-d96329b8a967" alt="Pascal Triangle Height 5" width="300">
  <img src="https://github.com/joondoogg/Project/assets/146046027/0cf6dd75-c136-466a-b5fd-22b2b9924b4a" alt="Pascal Triangle Height 7" width="300">
</div>
각각 높이가 3, 5, 7인 파스칼 삼각형을 도식화해보았으며, 이처럼 데이터를 도식화하는 프로그램을 구현해보았습니다.


3. _제 세 번째 프로젝트는 알고리즘의 시간 복잡도 분석입니다._
      1) 만들게 된 동기 : 릿코드의 문제를 풀며 알고리즘의 시간 복잡도가 실행 시간, 코드의 효율성을 측정하는 중요한 척도임을 공부했습니다. 그렇다면, O(1), O(n), O(nlogn), O(n^2), O(2^n)의 시간 복잡도를 가진 대표적인 알고리즘들이 실제 n 값에 의해 얼마나 실행 시간 차이가 나는지 분석해보았습니다. 코딩을 해보면서 어떻게 실행시간을 줄이는지, 다른 사람들의 코드는 왜 빠르게 실행되며 훨씬 효율적인지에 대해 궁금했으며, 저 또한 효율적인 코드를 구현하여 실행시간을 단축하고 싶어서 이 프로젝트를 하여 효율적인 코딩을 하고자 하였습니다. 또한 2번 프로젝트에서 파스칼 삼각형을 그림 그리는 데에 pylot을 사용하였듯이, 이번에는 그래프를 그리는 데에 matplotlib의 pyplot을 활용했습니다.
      2) 다음은 알고리즘 코드입니다. (총 3개의 코드로 따로 첨부되어 있습니다. 일부로 밑에는 최종적인 O(2^n)까지 포함하는 코드로 설명했습니다)
      3개의 코드가 따로 첨부된 이유는, 하나는 O(2^n)까지 포함, 하나는 O(n^2)까지 포함, 마지막 하나는 O(nlogn)까지만 포함하여 구체적으로 각각을 비교해보고 싶어서 각각의 데이터를 그래프화 해보았습니다.
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
                  
                  # 시간 복잡도 O(2^n) 알고리즘: 피보나치 수 계산 (재귀)
                  def exponential_time(n):
                      if n <= 1:
                          return n
                      return exponential_time(n - 1) + exponential_time(n - 2)
                  
                  # 성능 측정을 위한 함수
                  def measure_time(func, arg):
                      start_time = time.time()
                      result = func(arg)
                      end_time = time.time()
                      return end_time - start_time
                  
                  # 데이터 생성 및 측정
                  input_sizes = [10, 20, 30, 40]  # 피보나치의 경우 n 값으로 사용
                  algorithms = {
                      "O(1)": constant_time,
                      "O(n)": linear_time,
                      "O(n log n)": merge_sort,
                      "O(n^2)": quadratic_time,
                      "O(2^n)": exponential_time
                  }
                  
                  results = {alg: [] for alg in algorithms}
                  
                  for size in input_sizes:
                      for name, func in algorithms.items():
                          if name == "O(2^n)":
                              time_taken = measure_time(func, size)  # 피보나치의 경우 n 값을 직접 전달
                          else:
                              arr = [random.randint(0, 10000) for _ in range(size * 10)]  # 나머지 경우 배열 크기 조정
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
각각의 알고리즘을 간략히 설명하겠습니다
      - O(1) : 변수 대입
      - O(n) : n 크기의 배열을 한 번 순회하여 O(n)이 되도록 하였습니다
      - O(nlogn) : sorting 중에서 merge sort를 구현하였습니다. 다른 sorting 같은 경우 최악의 경우 O(n^2)이 될 수도 있으며, 릿코드 풀이에 merge sort를 자주 썼습니다
      - O(n^2) : for 중첩문으로 구현하였습니다
      - O(2^n) : 가장 대표적인 피보나치 재귀함수를 구현하였습니다
      - measure_time(func, arg) 함수는 그래프의 y축이 될 실행시간을 측정하는 함수입니다. 이를 위해 time을 import하였습니다
      - 코드의 마지막은 pyplot을 이용해서 그래프를 만드는 과정입니다.

   3) 다음은 모델의 input과 output이 어떻게 입력되고 출력되는지 구체적인 결과값입니다. Colab에서 실행하였습니다.
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img width="568" alt="스크린샷 2024-06-20 오후 11 40 41" src="https://github.com/joondoogg/Project/assets/146046027/02bf9531-0d20-47b0-bafa-8516241282b2">
  <img width="554" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/6950dd91-5516-44b0-9d8b-c87df1dce656">
  <img width="569" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/fb4427da-7251-4ce4-889f-b855e7c8059a">
</div>

   - 첫 사진은 O(2^n) 알고리즘까지 포함하여 n=40까지만 test를 했는데도 유의미한 실행시간 차이가 났으며, 구체적으로 algo_with_2n.py의 실행결과입니다. (실행시간이 꽤 깁니다) 
   
   - 두 번째 사진은 O(n^2) 알고리즘까지 포함하여 n=10000까지 실행하였으며, O(n^2) 또한 확실한 실행시간 차이가 났습니다. algo_with_n2.py의 실행결과입니다.
     
   - 마지막 사진은 O(nlogn) 알고리즘까지 포함하여 n=100,000까지 실행했으며, 실행시간이 전부 거의 1초 이하인 것이 보입니다. 특히, O(nlogn)의 시간복잡도를 가지는 알고리즘은 다양하므로 그 중 3개(quick sort, heap sort, merge sort)를 구체적으로 비교하였습니다. algo_with_nlogn.py의 실행결과입니다.
     
   - 결론 : 결과적으로 보았을 때, 시간 복잡도가 작을 수록 즉, O(1)일 수록 압도적으로 데이터가 커질 때(n이 100,000까지), 실행시간이 유의미한 차이로 줄어든다는 결과가 도출되었습니다. 그러나 실제로 O(1), O(n)의 시간복잡도를 가지는 알고리즘을 활용하여 구체적인 문제 풀이나 현실에 적용하기는 부족한 면이 많이 보입니다. 특히 sorting이 대표적인 예시입니다. 따라서 O(nlogn)의 알고리즘이 합리적으로 효율적인 측면, 현실적인 측면 모두 충족되는 알고리즘임을 알 수 있습니다.
   - 추가적으로) 마지막 사진의 O(nlogn) 알고리즘 사이의 차이는 조금 더 공부해본 바, 메모리를 쓰는 방식으로(temporal / spatial locality) 올 수 있는 차이, 각각의 구현에서 쓰이는 내장함수 차이 등에 의해 약간의 차이가 있게 되었습니다.


      
