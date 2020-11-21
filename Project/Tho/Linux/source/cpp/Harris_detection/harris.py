import cv2

if __name__ == "__main__":
    img = cv2.imread("./Project/Tho/Linux/data/lena.jpg", cv2.IMREAD_GRAYSCALE)
    print(img[101][150])
