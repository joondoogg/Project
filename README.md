2020131013 정준혁 수학과 프로그래밍 프로젝트입니다.
총 4개의 프로젝트(총 코드는 6파일)을 만들었으며, 2개는 기본적인 감을 얻기 위한 수학적 내용을 담은 코드(역행렬 구하기, 파스칼 삼각형 프린트로 프롬트에 출력하기)를 만들고 나머지 2개는 알고리즘의 시간 복잡도에 대한 데이터 분석 1개와 심화로 더 공부해본 알고리즘으로 지하철의 출발역과 도착역을 적으면 환승시간을 2~6분의 랜덤 시간으로 잡고 평균적인 이동시간을 구하는 프로젝트를 만들어 보았습니다. 
프로젝트 하나씩 1) 만들게 된 동기와,  2) 코드 설명, 3) 모델의 input과 output에 대한 발표를 하겠습니다. 

1. 제 첫 프로젝트는 역행렬을 구하는 프로젝트입니다.
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

<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/d3eaacba-7774-44ca-9dad-b9469b53968d">

<img width="460" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/3c0ddacd-ee28-43a8-9fc3-479654668287">
역행렬의 결과가(존재한다면) 정확하게 구해짐을 확인할 수 있습니다.


2. 제 두 번째 프로젝트는 파스칼 삼각형을 출력하는 프로그램입니다.
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
     
   3) 다음은 모델의 input과 output이 어떻게 입력되고 출력되는지 구체적인 결과값입니다. Colab에서 실행하였습니다.
<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/5cefb801-b0dc-4d89-accb-5d86dc1195c3">
<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/2813b148-0ba1-4f01-ba91-d96329b8a967">
<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/0cf6dd75-c136-466a-b5fd-22b2b9924b4a">
높이 3, 5, 7의 파스칼 삼각형을 출력하였습니다.

      





