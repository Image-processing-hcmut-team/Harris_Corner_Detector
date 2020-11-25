import numpy as np
import cv2
import time
from skimage.feature import peak_local_max


def nothing(x):
    pass


def corner_detection(img_origin, window_size, method, output_type, *args, **kwargs):
    start = time.time()
    window_size = np.asarray(window_size, dtype=np.int)
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    window_size_halved = np.array(np.floor(window_size / 2), dtype=np.int)

    # Calculate Ix and Iy based on Sobel filter
    img_dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)
    img_dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)

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
    if method == "Fast_Harris":
        M11_matrix = np.zeros([img_gray.shape[0],
                              img_gray.shape[1]], dtype=np.float)
        M22_matrix = np.zeros([img_gray.shape[0],
                              img_gray.shape[1]], dtype=np.float)
        M12_matrix = np.zeros([img_gray.shape[0],
                              img_gray.shape[1]], dtype=np.float)
        # trace_matrix = np.zeros([img_gray.shape[0],
        #                         img_gray.shape[1]], dtype=np.float)
        # det_matrix = np.zeros([img_gray.shape[0],
        #                         img_gray.shape[1]], dtype=np.float)

    # Integrate and calculate all value in matrix R_values
    for i in range(window_size_halved[0], img_gray.shape[0] - window_size_halved[0]):
        if i % 10 == 0:
            print("Completed %0.2f%%" % (100*(i - window_size_halved[0]) /
                  (img_gray.shape[0] - window_size_halved[0])))
        for j in range(window_size_halved[1], img_gray.shape[1] - window_size_halved[1]):
            if method != "Fast_Harris":
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
                    raise NameError("Wrong method name, try \"Harris\" or \"Shi-Tomasi\" or \"Fast_Harris\"")
            else:
                M11_matrix[i, j] = img_dx_squared_integral[i - window_size_halved[0], j - window_size_halved[1]] +\
                    img_dx_squared_integral[i + window_size_halved[0] + 1, j + window_size_halved[1] + 1] -\
                    img_dx_squared_integral[i - window_size_halved[0], j + window_size_halved[1] + 1] -\
                    img_dx_squared_integral[i + window_size_halved[0] + 1, j - window_size_halved[1]]
                M22_matrix[i, j] = img_dy_squared_integral[i - window_size_halved[0], j - window_size_halved[1]] +\
                    img_dy_squared_integral[i + window_size_halved[0] + 1, j + window_size_halved[1] + 1] -\
                    img_dy_squared_integral[i - window_size_halved[0], j + window_size_halved[1] + 1] -\
                    img_dy_squared_integral[i + window_size_halved[0] + 1, j - window_size_halved[1]]
                M12_matrix[i, j] = img_dxy_integral[i - window_size_halved[0], j - window_size_halved[1]] +\
                    img_dxy_integral[i + window_size_halved[0] + 1, j + window_size_halved[1] + 1] -\
                    img_dxy_integral[i - window_size_halved[0], j + window_size_halved[1] + 1] -\
                    img_dxy_integral[i + window_size_halved[0] + 1, j - window_size_halved[1]]
    if method == "Fast_Harris":
        R_values = (np.multiply(M11_matrix, M22_matrix) - np.square(M12_matrix))-0.004*np.square(M11_matrix+M22_matrix)
    stop = time.time()
    print("Completed 100%")
    print("Total processing time: %0.2f" % (stop - start))
    if output_type == "R_matrix":
        return R_values
    elif output_type == "Dirrect_img":
        if "min_distance" in kwargs:
            if "response_min" in kwargs:
                img_out = img_origin.copy()
                if kwargs['response_min'] > 0:
                    min_response = R_values.max()*np.power(10, 1-1/kwargs['response_min'])
                else:
                    min_response = R_values.max()
                local_max = peak_local_max(R_values, min_distance=kwargs['min_distance'], threshold_abs=min_response)
                for i in range(local_max.shape[0]):
                    img_out[local_max[i, 0]-1:local_max[i, 0]+2, local_max[i, 1]-1:local_max[i, 1]+2] = (0, 0, 255)
                return img_out
            else:
                raise ValueError("Missing argument \"response_min\"")
        else:
            raise ValueError("Missing argument \"min_distance\"")
    else:
        raise ValueError("Wrong output type, try \"R_matrix\" or \"Dirrect_img\"")


if __name__ == "__main__1":
    img = cv2.imread("./Project/Tho/Linux/data/lena.jpg")
    R_values = corner_detection(img, window_size=(5, 5), method="Fast_Harris", output_type="R_matrix")
    cv2.namedWindow("Result")
    cv2.createTrackbar("threshold", "Result", 2000, 6000, nothing)
    while True:
        img_out = img.copy()
        threshold = cv2.getTrackbarPos("threshold", "Result")
        min_response = R_values.max()*np.power(10, -threshold/1000)
        local_max = peak_local_max(R_values, min_distance=2, threshold_abs=min_response)
        for i in range(local_max.shape[0]):
            img_out[local_max[i, 0]-1:local_max[i, 0]+2, local_max[i, 1]-1:local_max[i, 1]+2] = (0, 0, 255)
        cv2.imshow("Result", img_out)
        if cv2.waitKey(100) & 0xff == 27:
            break

if __name__ == "__main__":
    img = cv2.imread("./Project/Tho/Linux/data/lena.jpg")
    img_corner = corner_detection(img, window_size=(5, 5), method="Fast_Harris",
                                  output_type="Dirrect_img", response_min=0.5, min_distance=2)
    while True:
        cv2.imshow("Result", img_corner)
        if cv2.waitKey(100) & 0xff == 27:
            break
