import cv2
import numpy as np
import json
import math
import os
from pydobot import Dobot

# ============ Dobot Parameters ============
DOBOT_PORT = "/dev/ttyACM0"   # or "COM3" on Windows
BOARD_Z   = -47
PATH_Z    = BOARD_Z + 5
TRAVEL_Z  = BOARD_Z + 20
TOOL_R    = 0
#HOME      = (225.2, 0, 150.96, TOOL_R)
SPEED_XY  = 70
SPEED_Z   = 70

# ============ Camera Settings ============
CAM_INDEX = 0
SAVE_FILE = "vision_robot_homography_4aruco.json"

# ============ ArUco Detection ============
def detect_aruco_markers(cap, needed_ids=(0,1,2,3)):
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_new = True
    except:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        use_new = False

    centers = {}
    print("🎯 Detecting 4 ArUco markers (IDs 0,1,2,3)... Press ESC to continue.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # 🌀 Flip 180° before processing
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_new:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id in needed_ids:
                    c = corners[i][0]
                    center = c.mean(axis=0)
                    centers[marker_id] = center
                    cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                    cv2.putText(frame, f"ID {marker_id}",
                                tuple(c[0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("ArUco Detection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyWindow("ArUco Detection")
    if len(centers) < 4:
        print(f"⚠️ Only {len(centers)} markers detected. Make sure all 4 are visible.")
    else:
        print("✅ All 4 markers detected.")
    return centers


# ============ Main Program ============
def main():
    print("🔌 Connecting to Dobot...")
    try:
        device = Dobot(port=DOBOT_PORT)
    except Exception as e:
        print("❌ Failed to connect to Dobot:", e)
        return
    device.speed(SPEED_XY, SPEED_Z)
    #device.move_to(*HOME)
    device.home()
    print("✅ Dobot connected and moved to HOME position.")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Failed to open camera. Try changing CAM_INDEX (0, 1, or 2).")
        return

    # --- Calibration / Load existing homography ---
    if os.path.exists(SAVE_FILE):
        print(f"📁 Calibration file found: {SAVE_FILE}")
        with open(SAVE_FILE, "r") as f:
            data = json.load(f)
        H = np.array(data["homography"], dtype=float)
        print("✅ Calibration file loaded successfully.")
    else:
        centers = detect_aruco_markers(cap)
        if len(centers) < 4:
            print("⚠️ Not all 4 markers detected. Exiting.")
            return

        aruco_real = {}
        for marker_id in [0,1,2,3]:
            input(f"\n👉 Move the robot tool tip to the center of ArUco ID={marker_id} and press Enter...")
            pose, _ = device.get_pose()
            x, y, z, r = pose
            aruco_real[marker_id] = np.array([x, y], dtype=float)
            print(f"📍 Real coordinates for marker {marker_id}: {aruco_real[marker_id].tolist()}")

        #device.move_to(*HOME)
        device.home()
        img_pts = np.array([centers[i] for i in sorted(centers.keys())], dtype=float)
        real_pts = np.array([aruco_real[i] for i in sorted(aruco_real.keys())], dtype=float)
        H, mask = cv2.findHomography(img_pts, real_pts, cv2.RANSAC, 2.0)
        print(f"✅ Homography computed ({int(mask.sum())}/{len(mask)} inliers).")

        data = {
            "homography": H.tolist(),
            "aruco_img_pts": img_pts.tolist(),
            "aruco_real_pts": real_pts.tolist()
        }
        with open(SAVE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Calibration data saved to {SAVE_FILE}.")

    def image_to_robot(x_img, y_img):
        p = np.array([[[x_img, y_img]]], dtype=float)
        p_r = cv2.perspectiveTransform(p, H)[0][0]
        return p_r[0], p_r[1]

    print("\n👆 Click on the image to move the robot. Press ESC to quit.")
    mouse_xy = (0,0)

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_xy
        mouse_xy = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            Xf, Yf = image_to_robot(x, y)
            print(f"🖱️ Click ({x},{y}) → Move to (X={Xf:.2f}, Y={Yf:.2f}) mm")
            device.move_to(Xf, Yf, TRAVEL_Z, TOOL_R)
            device.move_to(Xf, Yf, PATH_Z, TOOL_R)
            device.move_to(Xf, Yf, TRAVEL_Z, TOOL_R)
            #device.move_to(*HOME)
            device.home()

    cv2.namedWindow("Live View")
    cv2.setMouseCallback("Live View", on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # 🌀 Rotate 180° for live display
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        mx, my = mouse_xy
        Xr, Yr = image_to_robot(mx, my)
        cv2.putText(frame, f"X={Xr:.1f}, Y={Yr:.1f} mm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.circle(frame, (int(mx), int(my)), 4, (0,255,255), -1)
        cv2.imshow("Live View", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    #device.move_to(*HOME)
    device.home()
    device.close()
    print("\n🏁 Finished — Robot returned to HOME position.")

if __name__ == "__main__":
    main()
