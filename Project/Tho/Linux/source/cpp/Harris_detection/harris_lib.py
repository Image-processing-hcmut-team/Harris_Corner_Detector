import numpy as np
import cv2
import time


def nothing(x):
    pass


if __name__ == "__main__":
    start = time.time()
    img = cv2.imread("./Project/Tho/Linux/data/ex7.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = np.array([[41, 57, 98, 91, 84, 28, 20], [8, 95, 89, 22, 72, 37, 64],
    #                 [98, 96, 65, 85, 19, 83, 38], [64, 10, 52, 76, 51, 87, 36],
    #                 [36, 8, 77, 39, 41, 31, 29], [34, 12, 95, 79, 95, 51, 77],
    #                 [78, 76, 82, 47, 66, 22, 40]], dtype=np.uint8)
    img_dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # img_dx_norm = cv2.normalize(img_dx, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # img_dy_norm = cv2.normalize(img_dy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_dx_squared = np.square(img_dx)
    img_dy_squared = np.square(img_dy)
    test = np.sum(img_dx)

    window_size = np.array([3, 3], dtype=np.int)
    window_size_halved = np.array(np.floor(window_size / 2), dtype=np.int)

    R_values = np.zeros([img.shape[0],
                        img.shape[1]], dtype=np.float)
    # print(R_values.shape)
    for i in range(window_size_halved[0], img.shape[0] - window_size_halved[0]):
        print(i)
        for j in range(window_size_halved[1], img.shape[1] - window_size_halved[1]):
            # print(i - window_size_halved[0])
            # print(i + window_size_halved[0])
            # print(j - window_size_halved[1])
            # print(j + window_size_halved[1])
            # M_11 = np.sum(img_dx_squared[i - window_size_halved[0]:
            #               i + window_size_halved[0] + 1,
            #               j - window_size_halved[1]:
            #               j + window_size_halved[1] + 1])

            M_11 = img_dx_squared[i - window_size_halved[0]:
                                  i + window_size_halved[0] + 1,
                                  j - window_size_halved[1]:
                                  j + window_size_halved[1] + 1].sum()

            # M_22 = np.sum(img_dy_squared[i - window_size_halved[0]:
            #               i + window_size_halved[0] + 1,
            #               j - window_size_halved[1]:
            #               j + window_size_halved[1] + 1])

            M_22 = img_dy_squared[i - window_size_halved[0]:
                                  i + window_size_halved[0] + 1,
                                  j - window_size_halved[1]:
                                  j + window_size_halved[1] + 1].sum()

            M_21 = np.sum(img_dx[i - window_size_halved[0]:
                          i + window_size_halved[0] + 1,
                          j - window_size_halved[1]:
                          j + window_size_halved[1] + 1] *
                          img_dy[i - window_size_halved[0]:
                          i + window_size_halved[0] + 1,
                          j - window_size_halved[1]:
                          j + window_size_halved[1] + 1])
            trace = M_11+M_22
            det = M_11*M_22 - np.square(M_22)
            if trace > 0:
                R_values[i, j] = det / trace
            # else:
            #     print("Error")
            # pass
    stop = time.time()
    # print(R_values)
    print(stop - start)

    cv2.namedWindow("Result")
    cv2.createTrackbar("threshold", "Result", 0, 6000, nothing)
    while True:
        threshold = cv2.getTrackbarPos("threshold", "Result")
        percent = R_values.max()*np.power(10, -threshold/1000)
        output = R_values.copy()
        output[output < percent] = 0
        output[output > percent] = 255
        corner = output.astype(np.uint8)

        cv2.imshow("Result", corner)
        cv2.waitKey(100)
