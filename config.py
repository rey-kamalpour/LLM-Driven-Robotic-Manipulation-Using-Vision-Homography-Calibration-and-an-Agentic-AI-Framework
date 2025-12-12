import numpy as np
import os
import json

WORKING_DIR = "."



default_port="/dev/ttyACM0"

# 3x3 HOMOGRAPHY matrix for pixel -> robot (X,Y)
M = np.array([
    [-0.030698729717284084,     0.4702576860470572,     182.62747907914147],
    [0.43989816094190504,    	-0.000429296026037292,  -151.44667223879978],
   [-0.00007643435523797676,  0.0001150414337183519, 1.0]
], dtype=np.float64)

block_height = -42

z_above = 100           
z_table = -52         
block_height_mm = 10  
block_length_mm = 17    
stack_delta_mm = 2    
side_offset_mm = 14    

capture_wait_time = 10
camera_index = 0


SAVE_FILE = "vision_robot_homography_4aruco.json"

def Get_calibrate_H():
     # --- Calibration / Load existing homography ---
       # 👈 اعلام اینکه از نسخه‌ی global استفاده می‌کنیم
    global M
    # --- Calibration / Load existing homography ---
    if os.path.exists(SAVE_FILE):
        print(f"📁 Calibration file found: {SAVE_FILE}")
        with open(SAVE_FILE, "r") as f:
            data = json.load(f)
        M = np.array(data["homography"], dtype=float)
        print(M)
        print("✅ Calibration file loaded successfully.")
    else:
        return 0

Get_calibrate_H()