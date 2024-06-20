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
