import numpy as np
import cv2
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from harris_lib_v1_4 import corner_detection

if __name__ == "__main__":
    img_1 = cv2.imread("./Project/Tho/Linux/data/lena_2")
    img_2 = cv2.imread("./Project/Tho/Linux/data/lena_1")

    R_values_1 = corner_detection(img_1, window_size=(5, 5), method="Harris", output_type="R_matrix")
    R_values_2 = corner_detection(img_2, window_size=(5, 5), method="Harris", output_type="R_matrix")

    min_response_1 = R_values_1.max()*np.power(10, -625/1000)
    min_response_2 = R_values_2.max()*np.power(10, -625/1000)

    local_max_1 = peak_local_max(R_values_1, min_distance=5, threshold_abs=min_response_1)
    local_max_2 = peak_local_max(R_values_2, min_distance=5, threshold_abs=min_response_2)

    feature_vecs_1 = np.zeros((local_max_1.shape[0], 147), dtype=int)
    feature_vecs_2 = np.zeros((local_max_2.shape[0], 147), dtype=int)

    for i in range(local_max_1.shape[0]):
        feature_vecs_1[i, :] = img_1[local_max_1[i, 0]-3:local_max_1[i, 0]+4,
                                     local_max_1[i, 1]-3:local_max_1[i, 1]+4].flatten()
    for i in range(local_max_2.shape[0]):
        feature_vecs_2[i, :] = img_2[local_max_2[i, 0]-3:local_max_2[i, 0]+4,
                                     local_max_2[i, 1]-3:local_max_2[i, 1]+4].flatten()

    correspond_coord = np.zeros((feature_vecs_2.shape[0], 2), dtype=np.int)
    correspond_dist = np.zeros(feature_vecs_2.shape[0], dtype=np.float)
    for i in range(feature_vecs_2.shape[0]):
        correspond_dist[i], coord = cKDTree(feature_vecs_1).query(feature_vecs_2[i, :], k=1)
        # correspond_corner[i, :] = local_max_1[cKDTree(feature_vecs_1).query(feature_vecs_2[i, :], k=1)[1], :]
        correspond_coord[i, :] = local_max_1[coord, :]
    shortest_dist = correspond_dist.argsort()[:10]
    translate_const = np.zeros(2, dtype=np.float)
    for i in range(shortest_dist.shape[0]):
        translate_const += local_max_2[shortest_dist[i], :] - correspond_coord[shortest_dist[i], :]
    translate_const = np.round(translate_const/shortest_dist.shape[0] + 0.5).astype(np.int)
    img_out = cv2.copyMakeBorder(img_2, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)
    img_out[200+translate_const[0]:200+translate_const[0]+img_1.shape[0],
            200+translate_const[1]:200+translate_const[1]+img_1.shape[1]] = img_1
    cv2.imshow("Result", img_out)
    cv2.waitKey(0)
