import cv2
import numpy as np
import os


img_path = "C:\\Users\\HP\\OneDrive\\Desktop\\png-clipart-table-wood-furniture-table-angle-rectangle-thumbnail.png"


if not os.path.isfile(img_path):
    print('Image file not found:', img_path)
else:
    
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray, gray)


    focal_length = 1.0  
    baseline = 0.1  
    Q = np.float32([[1, 0, 0, -img.shape[1]/2],
                    [0, -1, 0,  img.shape[0]/2],
                    [0, 0, 0,  -focal_length],
                    [0, 0, 1/baseline, 0]])
    points = cv2.reprojectImageTo3D(disparity, Q)

    
    mask = disparity > 0
    points = points[mask]

   
    header = ['ply\n', 'format ascii 1.0\n', 'element vertex {}\n'.format(
        points.shape[0]), 'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n']
    with open('point_cloud.ply', 'w') as f:
        f.writelines(header)
        np.savetxt(f, points, fmt='%.3f')
