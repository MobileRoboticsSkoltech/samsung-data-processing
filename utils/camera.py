import cv2

class Camera:
    def __init__(self, K, dist_coefs, height, width):
        self.K = K
        self.dist_coefs = dist_coefs
        self.height = height
        self.width = width
        self.shape = (self.height, self.width)
        
        self.K_undist, self.map_x_undist, self.map_y_undist = self.__get_undist_params()
        
    def __get_undist_params(self):
        # OpenCV has its own view on shape style
        shape = (self.width, self.height)
        K_undist, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, shape, 1, shape)

        map_x_undist, map_y_undist = cv2.initUndistortRectifyMap(
            self.K, self.dist_coefs, None, K_undist, shape, cv2.CV_32FC1
        )
    
        return K_undist, map_x_undist, map_y_undist
