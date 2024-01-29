import numpy as np

class KalmanFilter:

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):

        # Control variables u and initial state x
        self.u = np.array([[u_x], [u_y]]) 
        self.x = np.array([[0], [0], [0], [0]])

        # System model matrices A and B
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) 
        self.B = np.array([[(dt**2)/2, 0], [0, (dt**2)/2], [dt, 0], [0, dt]])

        # Measurement matrix H
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Process noise covariance matrix Q
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0], [0, (dt**4)/4, 0, (dt**3)/2], [(dt**3)/2, 0, dt**2, 0], [0, (dt**3)/2, 0, dt**2]]) * std_acc**2

        # Measurement noise covariance matrix R
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])

        # Prediction error covariance matrix P
        self.P = np.eye(self.A.shape[0])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    # Update the state estimate
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)