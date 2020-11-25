import numpy as np
import cv2
from skimage.feature import peak_local_max
from harris_lib_v1_4 import corner_detection


def nothing(x):
    pass


if __name__ == "__main__":
    img_1 = cv2.imread("./Project/Tho/Linux/data/tue_1.jpg")
    img_2 = cv2.imread("./Project/Tho/Linux/data/tue_2.jpg")
    R_values_1 = corner_detection(img_1, window_size=(5, 5), method="Harris", output_type="R_matrix")
    R_values_2 = corner_detection(img_2, window_size=(5, 5), method="Harris", output_type="R_matrix")
    cv2.namedWindow("Result 1")
    cv2.createTrackbar("threshold", "Result 1", 625, 6000, nothing)
    while True:
        img_out_1 = img_1.copy()
        img_out_2 = img_2.copy()
        threshold = cv2.getTrackbarPos("threshold", "Result 1")
        min_response_1 = R_values_1.max()*np.power(10, -threshold/1000)

        local_max_1 = peak_local_max(R_values_1, min_distance=2, threshold_abs=min_response_1)
        for i in range(local_max_1.shape[0]):
            img_out_1[local_max_1[i, 0]-1:local_max_1[i, 0]+2, local_max_1[i, 1]-1:local_max_1[i, 1]+2] = (0, 0, 255)

        min_response_2 = R_values_2.max()*np.power(10, -threshold/1000)
        local_max_2 = peak_local_max(R_values_2, min_distance=2, threshold_abs=min_response_2)
        for i in range(local_max_2.shape[0]):
            img_out_2[local_max_2[i, 0]-1:local_max_2[i, 0]+2, local_max_2[i, 1]-1:local_max_2[i, 1]+2] = (0, 0, 255)

        cv2.imshow("Result 1", img_out_1)
        cv2.imshow("Result 2", img_out_2)
        if cv2.waitKey(100) & 0xff == 27:
            break
