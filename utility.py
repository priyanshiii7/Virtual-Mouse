#code to get angle between the fingers, and to get distNCE BW FINGERS

import numpy as np

def get_angle(a , b, c):    #a,b,c distance bw fingers - point 5 = a (base of finger), 6 = b(middle line in finger), 8 = c (tip)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])  #difference bw angle in ab, x axis and bc and x axis
    angle = np.abs(np.degrees(radians))  #converting the ans from radians to degrees
    return angle 

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return 
    
    (x1, y1), (x2, y2), = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 -y1)    #cal the euclidian distance here
    return np.interp(L, [0,1], [0,1000])
