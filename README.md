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

        
  3) 다음은 모델의 input과 output이 어떻게 입력되고 출력되는지 구체적인 결과값입니다.
<img width="386" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/d2459a7a-e13b-4f45-8e4a-5a17d181b936">
<img width="460" alt="image" src="https://github.com/joondoogg/Project/assets/146046027/3c0ddacd-ee28-43a8-9fc3-479654668287">








