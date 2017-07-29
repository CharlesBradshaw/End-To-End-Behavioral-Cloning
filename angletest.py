import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def newAngle(angle_a ,side2 = 4, side1 = 8):
    angle_a = np.deg2rad(angle_a)
    side3 = side1*side1 + side2*side2 + -2 * side1 * side2 * np.cos(angle_a)
    print(side3)
    return np.rad2deg(np.arcsin(np.sin(angle_a)*side1/side3))



side1 = 5
side2 = 7
angle_a = 49
angle_a = np.deg2rad(angle_a)

side1_temp = side1*side1
side2_temp = side2*side2
next_temp = -2 * side1 * side2 * np.cos(angle_a)
add_temp = np.sqrt(side1_temp + side2_temp + next_temp)

print(side1_temp,side2_temp,next_temp,add_temp)