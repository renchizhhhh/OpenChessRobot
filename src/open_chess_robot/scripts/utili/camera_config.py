#!/home/charles/panda/panda_env310/bin/python3.10

import os

import cv2
from pathlib import Path
import typing
import numpy as np

import pyzed.sl as sl
from pyzed.sl import Camera as Zedcam

from setup_configurations import MARKER_SIZE, MARKER_TYPE, CAM_IP

FRAMEWIDTH = 1920

# define names of ArUco tags supported by OpenCV.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    from: https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markers-which-is-better
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    print(f"detected:\n rvecs {rvecs} \ntvecs:{tvecs}")
    return np.asarray(rvecs), np.asarray(tvecs), np.asarray(trash)
    # return rvecs, tvecs, trash


class Camera:
    def __init__(
        self,
        ip: str,
        port: int,
        name: str = "default",
    ) -> None:
        self.name = name
        self.ip = ip
        self.port = port
        self.zed_cam = Zedcam()
        # open streaming camera
        self._init_streaming()
        self.runtime = sl.RuntimeParameters(enable_depth=True) # NOTE: currently depth is not used
        # camera intrinsics (default is the left camera)
        self.cur_side = "left" # record the last call for the side of get_image
        self.camera_matrix = None
        self.dist_coeff = None
        # store the image/depth/point cloud
        self.img_mat = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        # store the marker coordinates in pixels and in the world
        self.marker_pixel_coordinates = None
        self.detected_markers = dict()

    def _init_streaming(self):
        init_param = sl.InitParameters()
        init_param.set_from_stream(self.ip, self.port)
        status = self.zed_cam.open(init_param)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            raise ConnectionError("streaming not successfully initialized")

        print(f"camera {self.name} streaming at {self.ip} port {self.port}")

    def _load_calibration_from_file(self, side="left"):
        calibrationFile = cv2.FileStorage("calibration_t1", cv2.FILE_STORAGE_READ)
        if side == "left":
            self.camera_matrix = calibrationFile.getNode("mtx_l").mat()
        else:
            self.camera_matrix = calibrationFile.getNode("mtx_r").mat()
        self.dist_coeff = dist_r = calibrationFile.getNode("dist_r").mat()
        calibrationFile.release()
        print(f'Calibration of camera: {self.camera_matrix} {self.dist_coeff}')

    def _load_calibration(self, side="left"):
        calibration = self.zed_cam.get_camera_information().calibration_parameters
        if side == "left":
            cal = calibration.left_cam
        else:
            cal = calibration.right_cam
        self.camera_matrix = np.array(
            [
                [cal.fx, 0, cal.cx],
                [0, cal.fy, cal.cy],
                [0, 0, 1],
            ]
        )
        self.dist_coeff = cal.disto

    def close(self):
        self.zed_cam.close()

    def reset(self):
        import time
        self.close()
        time.sleep(0.3)
        self._init_streaming()
        time.sleep(1)

    def get_img(self, side: str = "left") -> np.ndarray:
        """get image from the streaming camera

        Args:
            side (str, optional): choose left or right camera to retrieve image. Defaults to "left".

        Raises:
            Exception: camera is not streaming

        Returns:
            np.ndarray: image from the streaming in the (b,g,r) order
        """
        if side == "left":
            view_side = sl.VIEW.LEFT
        elif side == "right":
            view_side = sl.VIEW.RIGHT
        else:
            raise ValueError("camera side should either be left or right.")
        self.cur_side = side
        if self.zed_cam.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            self.zed_cam.retrieve_image(self.img_mat, view_side)
        else:
            # reopen streaming camera
            print("current runtime: {}", self.zed_cam.grab(self.runtime))
            self.reset()
            self.runtime = sl.RuntimeParameters()
            self.zed_cam.retrieve_image(self.img_mat, view_side)
            # raise Exception(f" current problem: {self.zed_cam.grab(self.runtime)}\n cannot grab image from the streaming")
        # delete the alpha channel from (b,g,r,a) to make opencv compatible
        img = np.delete(self.img_mat.get_data(), 3, 2)
        return img

    def detect_markers(
        self,
        frame: np.ndarray,
        marker_type: str = MARKER_TYPE,
        show=False,
        refine=False,
    ):
        """find the corners of markers in pixels with their id respectively

        Args:
            frame (np.ndarray): RGB image in ndarray
            marker_type (str, optional): marker type in the marker dict. Defaults to "DICT_4X4_50".
            show (bool, optional): draw the results on some images. Defaults to False.

        Raises:
            Exception: wrong auruco marker type

        Returns:
            corners, ids : list(np.array), list(int)?
        """
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        # detector parameters: https://docs.opencv.org/4.5.3/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
        params = cv2.aruco.DetectorParameters_create()
        # rgbframe = cv2.cvtColor(gframe, cv2.COLOR_GRAY2BGR)
        try:
            dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[marker_type])
        except KeyError:
            raise Exception("indicated marker is not listed in the ARUCO_DICT")
        corners, ids, _ = cv2.aruco.detectMarkers(
            gframe, dictionary=dictionary, parameters=params
        )
        ids = ids.reshape(-1)
        corners = np.array(corners).reshape((-1, 4, 2))
        for i, id in enumerate(ids):
            marker_properties = dict()
            marker_properties["corners2img"] = corners[i]
            marker_properties["pos2img"] = np.mean(corners[i], axis=0, dtype=np.uint8)
            self.detected_markers[f"{id}"] = marker_properties
        if refine:
            num_markers = len(ids)
            valid_res = 1
            for i in range(100):
                if valid_res > 9:  # threshold for stop
                    break
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gframe, dictionary=dictionary, parameters=params
                )
                corners = np.array(corners).reshape((-1, 4, 2))
                ids = ids.reshape(-1)
                if len(ids) == num_markers:
                    valid_res += 1
                    for id, corner in zip(ids, corners):
                        self.detected_markers[f"{id}"]["corners2img"] += corner
            for id in ids:
                self.detected_markers[f"{id}"]["corners2img"] /= valid_res
                self.detected_markers[f"{id}"]["pos2img"] = np.mean(
                    self.detected_markers[f"{id}"]["corners2img"], axis=0, dtype=np.uint8
                )
            corners = np.array([self.detected_markers[f"{id}"]["corners2img"] for id in ids])
        if show:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imwrite("detected.png", frame)
        return corners, ids

    def locate_markers(self, corners, ids, marker_size=MARKER_SIZE, refine=False):
        """locate marker coordinates to the sl.IMAGE coordinate frame by cv2.aruco"""
        self._load_calibration(self.cur_side)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.camera_matrix, self.dist_coeff
        )
        rvec, tvec = rvec.squeeze(), tvec.squeeze()
        for i, id in enumerate(ids):
            self.detected_markers[f"{id}"]["pos2camera"] = tvec[i]
            self.detected_markers[f"{id}"]["rot2camera"] = rvec[i]
        if refine:
            # now only refine the pos
            valid_res = 1
            for i in range(100):
                if valid_res > 9:
                    break
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, self.camera_matrix, self.dist_coeff
                )
                rvec, tvec = rvec.squeeze(), tvec.squeeze()
                valid_res += 1
                for i, id in enumerate(ids):
                    self.detected_markers[f"{id}"]["pos2camera"] += tvec[i]
            pos = [self.detected_markers[f"{id}"]["pos2camera"] for id in ids]
            for id in ids:
                self.detected_markers[f"{id}"]["pos2camera"] /= valid_res

    def camera_angle_in_ypr(self, rvec):
        """extract the euler angle from rvec"""
        # TODO: not finished and tested
        rotM = np.zeros(shape=(3, 3))
        cv2.Rodrigues(rvec, rotM, jacobian=0)
        ypr = cv2.RQDecomp3x3(rotM)
        return ypr

    def locate_pixels(self, coordinates):
        """locate pixel coordinates to the sl.IMAGE coordinate frame by point clouds"""
        # TODO: not finished and tested
        # a tuple, first element being the ERROR_CODE, second the 4D
        # numpy array, the last element of which is zero
        origin_trans = None
        rot_matrix = None
        coor_offset = None
        self.zed_cam.retrieve_measure(
            self.point_cloud, measure=sl.MEASURE.XYZ, type=sl.MEM.CPU
        )
        points_value = self.point_cloud.get_value(
            coordinates[0], coordinates[1], memory_type=sl.MEM.CPU
        )
        # Check if pc_value contains valid values
        if points_value[0] is sl.ERROR_CODE.SUCCESS:
            if not np.any(np.isnan(points_value[1])):
                diff = np.reshape(points_value[1][0:3], (3, 1)) - (origin_trans * 100)
                coor = np.matmul(rot_matrix, diff)
                coor = coor - coor_offset
                return np.transpose(coor)
        else:
            print(f"ERROR: FAILURE TO READ THE POINT CLOUD: {points_value[0]}")


if __name__ == "__main__":
    my_cam = Camera(
        ip=CAM_IP,
        port=30000,
        name="1",
    )
    corners, ids = my_cam.detect_markers(frame=my_cam.get_img(), show=True)
    if any(ids):
        my_cam.locate_markers(corners, ids)
        print(my_cam.detected_markers.items())
