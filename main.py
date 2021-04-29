# Uncomment when using the realsense camera
import math

import pyrealsense2.pyrealsense2 as rs  # For (most) Linux and Macs
# import pyrealsense2 as rs # For Windows
import numpy as np
import logging
import time
import datetime
import drone_lib
# import fg_camera_sim
import cv2
import imutils
import random
import logging
import traceback
import sys
import os
import glob
import shutil
from pathlib import Path
import argparse

log = None  # logger instance
GRIPPER_OPEN = 1087
GRIPPER_CLOSED = 1940
gripper_state = GRIPPER_CLOSED  # assume gripper is closed by default

IMG_SNAPSHOT_PATH = '/dev/drone_data/mission_data/cam_pex003'
IMG_WRITE_RATE = 10  # write every 10 frames to disk...

# Various mission states:
# We start out in "seek" mode, if we think we have a target, we move to "confirm" mode,
# If target not confirmed, we move back to "seek" mode.
# Once a target is confirmed, we move to "target" mode.
# After positioning to target and calculating a drop point, we move to "deliver" mode
# After delivering package, we move to RTL to return home.
MISSION_MODE_SEEK = 0
MISSION_MODE_CONFIRM = 1
MISSION_MODE_TARGET = 2
MISSION_MODE_DELIVER = 4
MISSION_MODE_RTL = 8

# Tracks the state of the mission
mission_mode = MISSION_MODE_SEEK

# x,y center for 640x480 camera resolution.
FRAME_HORIZONTAL_CENTER = 320.0
FRAME_VERTICAL_CENTER = 240.0

# Number of frames in a row we need to confirm a suspected target
REQUIRED_SIGHT_COUNT = 1  # must get 60 target sightings in a row to be sure of actual target

# Violet target
COLOR_RANGE_MIN = (110, 100, 75)
COLOR_RANGE_MAX = (160, 255, 255)

# Blue (ish) target
# COLOR_RANGE_MIN = (80, 50, 50)
# COLOR_RANGE_MAX = (105, 255, 255)

# Smallest object radius to consider (in pixels)
MIN_OBJ_RADIUS = 10

UPDATE_RATE = 1  # How many frames do we wait to execute on.

TARGET_RADIUS_MULTI = 1.7  # 1.5 x the radius of the target is considered a "good" landing if drone is inside of it.

# Font for use with the information window
font = cv2.FONT_HERSHEY_SIMPLEX

# variables
drone = None
counter = 0
direction1 = "unknown"
direction2 = "unknown"
inside_circle = False

# tracks number of attempts to re-acquire a target (if lost)
target_locate_attempts = 0

# Holds the size of a potential target's radius
target_circle_radius = 0

# info related to last (potential) target sighting
last_obj_lon = None
last_obj_lat = None
last_obj_alt = None
last_obj_heading = None
last_point = None  # center point in pixels

# Uncomment below when using actual realsense camera
# Configure realsense camera stream
pipeline = rs.pipeline()
config = rs.config()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")

# todo: uncomment when deploying...
# args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

live = True


def release_grip(seconds=2):
    sec = 1

    while sec <= seconds:
        override_gripper_state(GRIPPER_OPEN)
        time.sleep(1)
        sec += 1


def override_gripper_state(state=GRIPPER_CLOSED):
    global gripper_state
    gripper_state = state
    drone.channels.overrides['7'] = gripper_state


def backup_prev_experiment(path):
    if os.path.exists(path):
        if len(glob.glob(f'{path}/*')) > 0:
            time_stamp = time.time()
            shutil.move(os.path.normpath(path),
                        os.path.normpath(f'{path}_{time_stamp}'))

    Path(path).mkdir(parents=True, exist_ok=True)


def clear_path(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)


def start_camera_stream():
    logging.info("configuring rgb stream.")
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    logging.info("Starting camera streams...")
    profile = pipeline.start(config)


def get_cur_frame(attempts=5, flip_v=False):
    # Wait for a coherent pair of frames: depth and color
    tries = 0

    # This will capture the frames from the simulator.
    # If using an actual camera, comment out the two lines of
    # code below and replace with code that returns a single frame
    # from your camera.
    # image = fg_camera_sim.get_cur_frame()
    # return cv2.resize(image, (int(FRAME_HORIZONTAL_CENTER * 2), int(FRAME_VERTICAL_CENTER * 2)))

    # Code below can be used with the realsense camera...
    while tries <= attempts:
        try:
            frames = pipeline.wait_for_frames()
            rgb_frame = frames.get_color_frame()
            rgb_frame = np.asanyarray(rgb_frame.get_data())

            if flip_v:
                rgb_frame = cv2.flip(rgb_frame, 0)
            return rgb_frame
        except Exception:
            print(Exception)

        tries += 1


def get_ground_distance(height, hypotenuse):
    import math

    # Assuming we know the distance to object from the air
    # (the hypotenuse), we can calculate the ground distance
    # by using the simple formula of:
    # d^2 = hypotenuse^2 - height^2

    return math.sqrt(hypotenuse ** 2 - height ** 2)


def calc_new_location_to_target(from_lat, from_lon, heading, distance1):
    from geopy import distance
    from geopy import Point

    # given: current latitude, current longitude,
    #        heading = bearing in degrees,
    #        distance from current location (in meters)

    origin = Point(from_lat, from_lon)
    destination = distance.distance(
        kilometers=(distance1 * .001)).destination(origin, heading)

    return destination.latitude, destination.longitude


def check_for_initial_target(net, swapRB=False):
    frame = get_cur_frame()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5, swapRB=swapRB, crop=False)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.25:  # args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", frame)

    if 'startX' in locals():
        center = ((endX - startX) / 2 + startX, (endX - startX) / 2 + startX)
        x = startX
        y = startY
        radius = (endY - startY + endX - startX) / 2

        return center, radius, (x, y), frame

    else:
        return None


def determine_drone_actions(last_point, frame, target_sightings):
    return


def conduct_mission(net):
    # Here, we will loop until we find a human target and deliver the care package,
    # or until the drone's flight plan completes (and we land).
    logging.info("Searching for target...")

    target_sightings = 0
    global counter, mission_mode, last_point, last_obj_lon, \
        last_obj_lat, last_obj_alt, \
        last_obj_heading, target_circle_radius

    logging.info("Starting camera feed...")
    start_camera_stream()

    while drone.armed:  # While the drone's mission is executing...

        if drone.mode == "RTL":
            mission_mode = MISSION_MODE_RTL
            logging.info("RTL mode activated.  Mission ended.")
            break

        # take a snapshot of current location
        location = drone.location.global_relative_frame
        last_lon = location.lon
        last_lat = location.lat
        last_alt = location.alt
        last_heading = drone.heading

        # look for a target in current frame
        center, radius, (x, y), frame = check_for_initial_target(net)

        if center is not None:

            logging.info(f"(Potential) target acquired @"
                         f"({center[0], center[1]}) with radius {radius}.")

            target_sightings += 1

            # We're looking for a person/pedestrian...
            last_point = center

            if mission_mode == MISSION_MODE_SEEK:
                logging.info(f"Locking in on lat {last_lat}, lon {last_lon}, "
                             f"alt {last_alt}, heading {last_heading}.")

                last_obj_lon = last_lon
                last_obj_lat = last_lat
                last_obj_alt = last_alt
                last_obj_heading = last_heading

            # TODO: draw bounding box around potential target in the current frame...


        else:
            # We have no target in the current frame.
            logging.info("No target found; continuing search...")
            cv2.putText(frame, "Scanning for target...", (10, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            target_sightings = 0  # reset target sighting
            last_point = None

        # Time to adjust drone's position?
        if (counter % UPDATE_RATE) == 0 \
                or mission_mode != MISSION_MODE_SEEK:
            # determine drone's next actions (if any)
            determine_drone_actions(last_point, frame, target_sightings)

        # Display information in windowed frame:
        cv2.putText(frame, direction1, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, direction2, (10, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw (blue) marker in center of frame that indicates the
        # drone's relative position to the target
        # (assuming camera is centered under the drone).
        cv2.circle(frame,
                   (int(FRAME_HORIZONTAL_CENTER), int(FRAME_VERTICAL_CENTER)),
                   10, (255, 0, 0), -1)

        # Now, show stats for informational purposes only
        # cv2.imshow("Real-time Detect", frame)
        if (counter % IMG_WRITE_RATE) == 0:
            cv2.imwrite(f"{IMG_SNAPSHOT_PATH}/frm_{counter}.png", frame)

        # key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #    break

        if mission_mode == MISSION_MODE_RTL:
            break  # mission is over.

        counter += 1


def angle():
    # gets current angle of the realsense

    anglepip = rs.pipeline()
    angleconfig = rs.config()
    angleconfig.enable_stream(rs.stream.accel)
    anglepip.start(angleconfig)

    camera_angle = 0

    try:
        while True:
            f1 = anglepip.wait_for_frames()
            accel = f1[0].as_motion_frame().get_motion_data()

            if not accel:
                print("no frame at cam ")
                continue

            accel_angle_x = math.degrees(math.atan2(accel.y, accel.z))
            accel_angle_x = int(accel_angle_x)
            camera_angle = accel_angle_x
            break

    finally:
        anglepip.stop()

    print("camera angle = " + str(camera_angle))
    return camera_angle


def main():
    global drone
    global log

    # Setup a log file for recording important activities during our session.
    log_file = time.strftime("Ethan_and_Josh_PEX03_%Y%m%d-%H%M%S") + ".log"

    # prepare log file...
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    log = logging.getLogger(__name__)

    log.info("PEX 03 start.")

    # Connect to the autopilot
    # drone = drone_lib.connect_device("127.0.0.1:14550", log=log)
    drone = drone_lib.connect_device("/dev/ttyACM0", baud=115200, log=log)

    # Create a message listener using the decorator.
    print(f"Finder above ground: {drone.rangefinder.distance}")

    # Test latch - ensure open/close.
    release_grip(2)

    # If the autopilot has no mission, terminate program
    drone.commands.download()
    time.sleep(1)

    log.info("Looking for mission to execute...")
    if drone.commands.count < 1:
        log.info("No mission to execute.")
        return

    # load serialized caffe model from disk
    print("[INFO] loading model...")

    # for now, just directly supply args here...
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

    # Arm the drone.
    drone_lib.arm_device(drone, log=log)

    # takeoff and climb 45 meters
    drone_lib.device_takeoff(drone, 20, log=log)

    try:
        # start mission
        drone_lib.change_device_mode(drone, "AUTO", log=log)

        log.info("backing up old images...")

        # Backup any previous images and create new empty folder for current experiment.
        backup_prev_experiment(IMG_SNAPSHOT_PATH)

        # Now, look for target...
        conduct_mission(net)

        # Mission is over; disarm and disconnect.
        log.info("Disarming device...")
        drone.armed = False
        drone.close()
        log.info("End of demonstration.")
    except Exception as e:
        log.info(f"Program exception: {traceback.format_exception(*sys.exc_info())}")
        raise
