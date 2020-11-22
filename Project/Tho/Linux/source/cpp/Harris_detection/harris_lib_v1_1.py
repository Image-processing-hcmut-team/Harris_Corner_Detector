import numpy as np
import cv2
import time


def nothing(x):
    pass


if __name__ == "__main__":
    start = time.time()
    img_origin = cv2.imread("./Project/Tho/Linux/data/lena.jpg")
    # img_origin = cv2.resize(img_origin, (200, 200))
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread("./Project/Tho/Linux/data/lena.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (200, 200))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Testing image for debug
    # img = np.array([[41, 57, 98, 91, 84, 28, 20], [8, 95, 89, 22, 72, 37, 64],
    #                 [98, 96, 65, 85, 19, 83, 38], [64, 10, 52, 76, 51, 87, 36],
    #                 [36, 8, 77, 39, 41, 31, 29], [34, 12, 95, 79, 95, 51, 77],
    #                 [78, 76, 82, 47, 66, 22, 40]], dtype=np.uint8)

    # Initialize window size
    window_size = np.array([5, 5], dtype=np.int)
    window_size_halved = np.array(np.floor(window_size / 2), dtype=np.int)

    # Calculate Ix and Iy based on Sobel filter
    img_dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    img_dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate (Ix)^2 integral
    img_dx_squared = np.square(img_dx)
    img_dx_squared_integral = cv2.integral(img_dx_squared)

    # Calculate (Iy)^2 integral
    img_dy_squared = np.square(img_dy)
    img_dy_squared_integral = cv2.integral(img_dy_squared)

    # Calculate Ix*Iy integral
    img_dxy = img_dx * img_dy
    img_dxy_integral = cv2.integral(img_dxy)

    # Initialize response matrix
    R_values = np.zeros([img.shape[0],
                        img.shape[1]], dtype=np.float)

    # Integrate and calculate all value in matrix R_values
    for i in range(window_size_halved[0], img.shape[0] - window_size_halved[0]):
        for j in range(window_size_halved[1], img.shape[1] - window_size_halved[1]):
            # Intergral implementation
            M_11 = img_dx_squared_integral[i - window_size_halved[0], j - window_size_halved[1]] +\
                img_dx_squared_integral[i + window_size_halved[0] + 1, j + window_size_halved[1] + 1] -\
                img_dx_squared_integral[i - window_size_halved[0], j + window_size_halved[1] + 1] -\
                img_dx_squared_integral[i + window_size_halved[0] + 1, j - window_size_halved[1]]

            M_22 = img_dy_squared_integral[i - window_size_halved[0], j - window_size_halved[1]] +\
                img_dy_squared_integral[i + window_size_halved[0] + 1, j + window_size_halved[1] + 1] -\
                img_dy_squared_integral[i - window_size_halved[0], j + window_size_halved[1] + 1] -\
                img_dy_squared_integral[i + window_size_halved[0] + 1, j - window_size_halved[1]]

            M_12 = img_dxy_integral[i - window_size_halved[0], j - window_size_halved[1]] +\
                img_dxy_integral[i + window_size_halved[0] + 1, j + window_size_halved[1] + 1] -\
                img_dxy_integral[i - window_size_halved[0], j + window_size_halved[1] + 1] -\
                img_dxy_integral[i + window_size_halved[0] + 1, j - window_size_halved[1]]
            # End of Intergral implementation

            # # Harris response formula implementation
            # trace = M_11+M_22
            # det = M_11*M_22 - np.square(M_12)
            # if trace > 0:
            #     R_values[i, j] = det / trace
            # pass
            # # End of Harris implementation

            # Shi - tomashi formula implementation
            M = np.array([[M_11, M_12], [M_12, M_22]], dtype=float)
            A = np.min(np.linalg.eigvalsh(M))
            R_values[i, j] = A
            # End of Shi - tomashi implementation
    stop = time.time()
    print("Total processing time: %0.2f (s)" % (stop - start))

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

# Old implementation
"""
# Normal loop implementation
# M_11 = np.sum(img_dx_squared[i - window_size_halved[0]:
#               i + window_size_halved[0] + 1,
#               j - window_size_halved[1]:
#               j + window_size_halved[1] + 1])

# M_22 = np.sum(img_dy_squared[i - window_size_halved[0]:
#               i + window_size_halved[0] + 1,
#               j - window_size_halved[1]:
#               j + window_size_halved[1] + 1])

# M_21 = np.sum(img_dx[i - window_size_halved[0]:
#               i + window_size_halved[0] + 1,
#               j - window_size_halved[1]:
#               j + window_size_halved[1] + 1] *
#               img_dy[i - window_size_halved[0]:
#               i + window_size_halved[0] + 1,
#               j - window_size_halved[1]:
#               j + window_size_halved[1] + 1])
# End of Normal loop implementation
"""

"""
# Square precalculate implementation
# M_11 = img_dx_squared[i - window_size_halved[0]:
#                       i + window_size_halved[0] + 1,
#                       j - window_size_halved[1]:
#                       j + window_size_halved[1] + 1].sum()

# M_22 = img_dy_squared[i - window_size_halved[0]:
#                       i + window_size_halved[0] + 1,
#                       j - window_size_halved[1]:
#                       j + window_size_halved[1] + 1].sum()

# M_21 = img_dxy[i - window_size_halved[0]:
#                i + window_size_halved[0] + 1,
#                j - window_size_halved[1]:
#                j + window_size_halved[1] + 1].sum()
# End of Square precalculate implementation
"""
