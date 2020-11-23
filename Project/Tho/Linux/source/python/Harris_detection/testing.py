from skimage.feature import peak_local_max
import numpy as np

if __name__ == "__main__":
    img = np.array([[41, 57, 98, 91, 84, 28, 20], [8, 95, 89, 22, 72, 37, 64],
                    [98, 96, 65, 85, 19, 83, 38], [64, 10, 52, 76, 51, 87, 36],
                    [36, 8, 77, 39, 41, 31, 29], [34, 12, 95, 79, 95, 51, 77],
                    [78, 76, 82, 47, 66, 22, 40]], dtype=np.uint8)
    xy = peak_local_max(img, min_distance=1, threshold_abs=5, exclude_border=False)
    print(xy)
