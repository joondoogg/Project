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
