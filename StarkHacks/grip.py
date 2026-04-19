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

AVERAGE_DENSITY = .0047 #g/mm3
AVERAGE_HEIGHT = 22 #mm
SIZE_THRESHOLD = 1200 #mm
COMPLEXITY_THRESHOLD = 12
PIXELS_PER_MM = 6 #must be calibrated based on camera default height
ITEM_MAP = {
    #update gripper type association to be a tuple of all necessary attributes (gripper type, weight, cog)
    #weight in grams, cog in mm x mm
    "brake_pads": ("MAGNETIC", 1000, (82.5, 25.5)),
    "lug_nut" : ("MAGNETIC", 100, (12.5, 12.5)),
    "reinforcement_plate" : ("MAGNETIC", 1450, (152.5, 101.5)),
    "rotary_seal" : ("MAGNETIC", 1500, (100, 100, 20)),
    "sheet_metal" : ("MAGNETIC", 2200, (152.5, 152.5)),
    "shims" : ("MAGNETIC", 60, (25, 19)),
    "ball_bearing": ("MECHANICAL", 150, (0, 0, 0)),
    "cabin_air_filter": ("MECHANICAL", 400, (127, 101.5)), #suction
    "car_fob": ("MECHANICAL", 50, (38, 19)), #suction
    "cartridge_filter" : ("MECHANICAL", 600, (55, 55, 95)), #suction
    "copper_bus_bar" : ("MECHANICAL", 600, (100, 25, 2.5)),
    "dashboard_panel" : ("MECHANICAL", 10000, (600, 200, 70)),
    "door_handle" : ("MECHANICAL", 300, (127, 25.5)), #suction
    "door_seal" : ("MECHANICAL", 1200, (100, 0, 0)),
    "flat_filter" : ("MECHANICAL", 350, (75, 50, 25)),
    "floor_panel" : ("MECHANICAL", 2200, (750, 600, 1)),
    "ford_emblem" : ("MECHANICAL", 150, (114.5, 44.5)), #suction
    "gear_shift" : ("MECHANICAL", 1000, (63.5, 63.5)),
    "oil_dispstick" : ("MECHANICAL", 70, (381, 6)),
    "oil_filter" : ("MECHANICAL", 250, (51, 38)),
    "oil_funnel" : ("MECHANICAL", 150, (76, 76)),
    "piston_rod" : ("MECHANICAL", 2250, (115, 51)),
    "spark_plug" : ("MECHANICAL", 55, (12.5, 12.5)),
    "spur_gear": ("MECHANICAL", 550, (25.5, 25.5)),
    "tapered_roller_bearing": ("MECHANICAL", 400, (38, 38)),
    "weather_strip_segment": ("MECHANICAL", 150, (150, 12.5)),
    "wire_harness_segment": ("MECHANICAL", 200, (101.5, 6))
}

def get_info(obj_name): #returns a tuple of the form (object name,
    return obj_name, *ITEM_MAP[obj_name]

def capture(): #captures the image
    ret, frame = cap.read()

    if not ret:
        raise IOError("Cannot read from camera")

    # frame_rgb is a numpy array, converted to RBG for Roboflow AI model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_b64, frame_rgb

def sort(img, frame_rgb): #sorts first using Roboflow machine learning trained ai model
    obj_name = roboflow_identify(img)
    obj_info = get_info(obj_name)

    #if it cannot find object, use manual algorithm
    if obj_name == None:
        obj_info = manual_identify(frame_rgb, img)

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
        name = best["class"]
        return name

def manual_identify(frame_rgb, img):
    #collect all attributes, pass them into assign_best_fit
    obj_coords = find_object(frame_rgb)
    if obj_coords is None: return None
    smooth = is_smooth(frame_rgb)
    magnetic = is_magnetic()
    simple = is_simple(img)
    small = is_small(*obj_coords)
    gripper_type = assign_best_fit(smooth, magnetic, simple, small)
    return "unidentified object", gripper_type, estimate_weight(frame_rgb), estimate_center_of_gravity(frame_rgb)

def auto_invert_decision(gray):
    #pre-designed algorithm that utilizes 3 methods of determining whether or not an input should be inverted
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Method 1: white pixel ratio
    otsu_t, mask = cv2.threshold(blurred, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio  = np.count_nonzero(mask) / mask.size
    vote_ratio   = white_ratio > 0.5

    # Method 2: center vs border brightness
    h, w = gray.shape
    bw, bh = int(w*0.15), int(h*0.15)
    border = np.concatenate([
        gray[:bh,:].ravel(), gray[-bh:,:].ravel(),
        gray[bh:-bh,:bw].ravel(), gray[bh:-bh,-bw:].ravel()])
    centre = gray[h//4:3*h//4, w//4:3*w//4].ravel()
    vote_region = float(np.mean(centre)) < float(np.mean(border))

    # Method 3: Otsu threshold value
    vote_otsu = otsu_t < 127

    votes = [vote_ratio, vote_region, vote_otsu]
    invert_count = sum(votes)

    return invert_count >= 2

def assign_best_fit(smooth, magnetic, simple, small): #booleans
    if not simple or not small:
        return "MECHANICAL"
    if magnetic and smooth:
        return "MAGNETIC"
    elif not magnetic and smooth and simple and small:
        return "SUCTION"
    return "MECHANICAL"

def is_smooth(img): #returns true if smooth, false if porous
    MC_SMOOTHNESS_THRESHOLD = 0.12
    MC_POROUS_THRESHOLD = 0.5

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    ps, scores = 16, []
    for y in range(0, h-ps, ps//2):
        for x in range(0, w-ps, ps//2):
            p = img[y:y+ps, x:x+ps].astype(float)
            d = p.max() + p.min()
            if d > 0: scores.append((p.max() - p.min())/d)
    mc = float(np.mean(scores)) if scores else 0

    return mc < MC_SMOOTHNESS_THRESHOLD

def find_object(img):
    #returns pixel coordinates representing the bounding box of the object

    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours( thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h

def contour_map(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    decision = auto_invert_decision(gray)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if decision: mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    main = max(contours, key=cv2.contourArea)
    return main

def vertex_count(main):
    EPSILON = 0.02

    perimeter = cv2.arcLength(main, closed=True)

    epsilon = EPSILON * perimeter

    approx = cv2.approxPolyDP(
        main,
        epsilon,
        closed=True
    )

    return len(approx)

def estimate_area(main):
    return cv2.contourArea(main) / PIXELS_PER_MM / PIXELS_PER_MM #px per mm squared

def is_magnetic(ser):
    FERROUS_THRESHOLD = 2.0
    ser.write(b"HALL_CHECK\n")
    raw = ser.readline().decode().strip()
    # e.g. "HALL:0.84G"
    gauss = float(raw.split(":")[1].replace("G", ""))
    return gauss > FERROUS_THRESHOLD

def is_simple(frame_rgb):
    #simplicity is based on number of shape vertices
    return vertex_count(contour_map(frame_rgb)) < COMPLEXITY_THRESHOLD

def is_small(x, y, w, h):
    #uses bounding box instead of actual shape area because overall dimensions matter more than precise area
    w /= PIXELS_PER_MM
    h /= PIXELS_PER_MM
    area = w * h
    return area < SIZE_THRESHOLD

def estimate_weight(frame_rgb):
    #based on size of object and an estimate of what the density is

    area = estimate_area(contour_map(frame_rgb))
    return area * AVERAGE_DENSITY * AVERAGE_HEIGHT

def estimate_center_of_gravity(frame_rgb): #returns mm estimate of cog
    #assumes even mass distribution across object
    gray_img = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if auto_invert_decision(gray_img):mask = cv2.bitwise_not(mask) #invert if needed

    moments = cv2.moments(mask)

    #no part detected
    if moments["m00"] == 0:
        raise Exception("No part detected")

    #part detected
    else:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return int(round(cx)) / PIXELS_PER_MM, int(round(cy)) / PIXELS_PER_MM


def mechanical_grip(obj_info):
    #execute mechanical gripping
    return None

def magnetic_grip(obj_info):
    #execute magnetic gripping
    return None

def suction_grip(obj_info):
    #execute suction gripping
    return None

def main():
    img_b64, frame_rgb = capture()
    sort(img_b64, frame_rgb)
    cap.release()

