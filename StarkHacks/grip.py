import cv2
import numpy as np
import requests
import base64
from PIL import Image
import io

API_KEY = "B6SbZ6O16rLkmYWK2JR7"
MODEL_ID = "starkhacks-2026/3"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #base off qualcomm cam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gripper_types = ["MECHANICAL", "MAGNETIC", "SUCTION"] #we are not yet using suction for actual implementation

SIZE_THRESHOLD = 10
PIXELS_PER_MM = 6 #must be calibrated based on camera default height
ITEM_MAP = {
    #update gripper type association to be a tuple of all necessary attributes (gripper type, weight, cog)
    "brake_pads": "MAGNETIC",
    "lug_nut" : "MAGNETIC",
    "reinforcement_plate" : "MAGNETIC",
    "rotary_seal" : "MAGNETIC",
    "sheet_metal" : "MAGNETIC",
    "shims" : "MAGNETIC",
    "ball_bearing": "MECHANICAL",
    "cabin_air_filter": "MECHANICAL", #suction
    "car_fob": "MECHANICAL", #suction
    "cartridge_filter" : "MECHANICAL", #suction
    "copper_bus_bar" : "MECHANICAL",
    "dashboard_panel" : "MECHANICAL",
    "door_handle" : "MECHANICAL", #suction
    "door_seal" : "MECHANICAL",
    "flat_filter" : "MECHANICAL",
    "floor_panel" : "MECHANICAL",
    "ford_emblem" : "MECHANICAL", #suction
    "gear_shift" : "MECHANICAL",
    "oil_dispstick" : "MECHANICAL",
    "oil_filter" : "MECHANICAL",
    "oil_funnel" : "MECHANICAL",
    "piston_rod" : "MECHANICAL",
    "spark_plug" : "MECHANICAL",
    "spur_gear": "MECHANICAL",
    "tapered_roller_bearing": "MECHANICAL",
    "weather_strip_segment": "MECHANICAL",
    "wire_harness_segment": "MECHANICAL"
}

def assign_best_fit(smooth, magnetic, simple, small): #booleans
    if not simple or not small:
        return "MECHANICAL"
    if magnetic and smooth:
        return "MAGNETIC"
    elif not magnetic and smooth and simple and small:
        return "SUCTION"
    return "MECHANICAL"

def capture(): #captures the image
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError("Cannot read from camera")

    # frame_rgb is a numpy array, converted to RBG for Roboflow AI model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    sort(img_b64)

def sort(img, frame_rgb): #sorts first using Roboflow machine learning trained ai model
    obj_name = roboflow_identify(img)

    #if it cannot find object, use manual algorithm
    if obj_name == None:
        obj_info = manual_identify(frame_rgb)

    #second entry in tuple is assigned gripper type, first entry is object name
    match obj_info[1]:
        case "MECHANICAL":
            mechanical_grip(obj_info)
        case "MAGNETIC":
            magnetic_grip(obj_info)
        case "SUCTION":
            suction_grip(obj_info)

def roboflow_identify(img): #uses roboflow to return image name
    response = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={"api_key": API_KEY},
        data=img,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    predictions = response.json().get("predictions", [])
    if not predictions:
        return None
    else:
        names = list(set(p["class"] for p in predictions))
        #print("Objects detected:", names) -- debug
        best = max(predictions, key=lambda p: p["confidence"])
        return best["class"]

def manual_identify(img):
    #collect all attributes, pass them into assign_best_fit
    return None

def is_smooth(img): #returns true if smooth, false if porous
    MC_SMOOTHNESS_THRESHOLD = 0.12
    MC_POROUS_THRESHOLD = 0.5
    GC_SMOOTHNESS_THRESHOLD = 50
    GC_POROUS_THRESHOLD = 200

    #UNFINISHED
    return False

def find_object(img):
    #returns pixel coordinates representing the bounding box of the object
    return None

def is_magnetic(obj):
    #determine magnetism based on input by linear hall sensor
    return False

def is_simple(obj):
    #simplicity is based on shape
    return False

def is_small(obj):
    #determine size of shape based on amount of pixels & pixels per mm
    return False

def estimate_weight(frame_rgb):
    #based on size of object and an estimate of what the material is / its density
    return 0

def estimate_center_of_gravity(frame_rgb): #returns pixel coordinates of cog
    #assumes event mass distribution across object
    gray_img = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #invert to cv2.THRESH_BINARY if background is lighter than part
    #currently inverted because of test circumstances

    moments = cv2.moments(mask)

    #no part detected
    if moments["m00"] == 0:
        raise Exception("No part detected")

    #part detected
    else:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return (int(round(cx)) / PIXELS_PER_MM, int(round(cy)) / PIXELS_PER_MM)

    return None




def mechanical_grip(obj_info):
    #execute mechanical gripping
    return None

def magnetic_grip(obj_info):
    #execute magnetic gripping
    return None

def suction_grip(obj_info):
    #execute suction gripping
    return None



