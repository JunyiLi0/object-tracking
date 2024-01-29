import numpy as np
import cv2
from Detector import detect
from KalmanFilter import KalmanFilter

kf = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
cap = cv2.VideoCapture('randomball.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        centers, radii = detect(frame)
        for centroid, r in zip(centers, radii):
            x_est = centroid
            kf.predict()
            kf.update(centroid)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), r, (0, 255, 0), -1)
            cv2.rectangle(frame, (int(kf.x[0]) - r, int(kf.x[1]) - r), (int(kf.x[0]) + r, int(kf.x[1]) + r), (255, 0, 0), 2)
            cv2.rectangle(frame, (int(x_est[0]) - r, int(x_est[1]) - r), (int(x_est[0]) + r, int(x_est[1]) + r), (0, 0, 255), 2)
            cv2.line(frame, (int(kf.x[0]), int(kf.x[1])), (int(x_est[0]), int(x_est[1])), (0, 0, 0), 2)
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
