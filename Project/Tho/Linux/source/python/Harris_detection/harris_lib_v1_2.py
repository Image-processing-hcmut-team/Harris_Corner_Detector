import numpy as np
import cv2
import time


def nothing(x):
    pass


def corner_detection(img_path, window_size, method):
    start = time.time()
    window_size = np.asarray(window_size, dtype=np.int)
    img_origin = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    window_size_halved = np.array(np.floor(window_size / 2), dtype=np.int)

    # Calculate Ix and Iy based on Sobel filter
    img_dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    img_dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

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
    R_values = np.zeros([img_gray.shape[0],
                        img_gray.shape[1]], dtype=np.float)

    # Integrate and calculate all value in matrix R_values
    for i in range(window_size_halved[0], img_gray.shape[0] - window_size_halved[0]):
        if i % 10 == 0:
            print("Completed %0.2f%%" % (100*(i - window_size_halved[0]) /
                  (img_gray.shape[0] - window_size_halved[0])))
        for j in range(window_size_halved[1], img_gray.shape[1] - window_size_halved[1]):
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
            if method == "Harris":
                # Harris response formula implementation
                trace = M_11+M_22
                det = M_11*M_22 - np.square(M_12)
                if trace > 0:
                    R_values[i, j] = det / trace
                pass
                # End of Harris implementation
            elif method == "Shi-Tomasi":
                # Shi - tomashi formula implementation
                M = np.array([[M_11, M_12], [M_12, M_22]], dtype=float)
                A = np.min(np.linalg.eigvalsh(M))
                R_values[i, j] = A
                # End of Shi - tomashi implementation
            else:
                raise NameError("Wrong method name, try \"Harris\" or \"Shi-Tomasi\"")

    stop = time.time()
    print("Completed 100%")
    print("Total processing time: %0.2f" % (stop - start))
    return R_values


if __name__ == "__main__":
    R_values = corner_detection("./Project/Tho/Linux/data/lena.jpg", (5, 5), "Harris")
    cv2.namedWindow("Result")
    cv2.createTrackbar("threshold", "Result", 2000, 6000, nothing)
    while True:
        threshold = cv2.getTrackbarPos("threshold", "Result")
        percent = R_values.max()*np.power(10, -threshold/1000)
        output = R_values.copy()
        output[output < percent] = 0
        output[output > percent] = 255
        corner = output.astype(np.uint8)
        cv2.imshow("Result", corner)
        if cv2.waitKey(100) & 0xff == 27:
            break
