import cv2
import numpy as np

class MapPreprocessor:
    def __init__(self, K=5, attempts=10, max_iter=20, eps=0.9):
        """
        Initialize the preprocessor with k-means parameters.
        K: Number of clusters.
        attempts: Number of k-means attempts.
        max_iter: Maximum iterations for k-means.
        eps: Epsilon for k-means termination criteria.
        """
        self.K = K
        self.attempts = attempts
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    def process(self, image, output_binary_path="../data/processed_images/binary_map.png", 
                      output_npy_path="../data/processed_images/cspace.npy"):
        """
        Process the input image using k-means clustering and generate a binary map.
        The binary map will have water (cluster with index 0) set to white (255)
        and all other pixels set to black (0).
        """
        # Reshape image for k-means
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        
        ret, label, center = cv2.kmeans(Z, self.K, None, self.criteria, self.attempts, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        label_2D = label.reshape((image.shape[0], image.shape[1]))
        
        water_label = 0
        binary_map = np.where(label_2D == water_label, 255, 0).astype(np.uint8)
        
        cv2.imwrite(output_binary_path, binary_map)
        np.save(output_npy_path, binary_map)
        
        return binary_map
