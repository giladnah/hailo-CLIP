import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import json
import random
import time
import multiprocessing
import hailo

class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.running = True
        CLOTHES_JSON_PATH = "data_all.json"
        CLOTHES_FOLDER = os.path.join("static", "clothes")
        # Load the clothes mapping from JSON
        with open(CLOTHES_JSON_PATH, "r", encoding="utf-8") as f:
            clothes_map = json.load(f)
        self.clothes_map = clothes_map
        self.MAX_QUEUE_SIZE = 3
        self.labels_queue = multiprocessing.Queue(maxsize=self.MAX_QUEUE_SIZE)
        client_process = multiprocessing.Process(target=self.label_to_css, args=(self.labels_queue,))
        client_process.start()
        # matched_file = self.parse_lable("a men wearing REFLECTIVE EFFECT JACKET")
    
    def parse_lable(self,lable_str):
        if "men" in lable_str:
            gender = "Men"
        elif "women" in lable_str:
            gender = "Women"
        else:
            # Not recognized, default or pick randomly
            gender = None
            # We also assume the item is after "wearing".
        # For example, "a men wearing REFLECTIVE EFFECT JACKET"
        # -> item_str = "REFLECTIVE EFFECT JACKET"
        parts = lable_str.split("wearing")
        if len(parts) > 1:
            item_str = parts[1].strip().upper()  # "REFLECTIVE EFFECT JACKET"
        else:
            item_str = None
    
        # Attempt to look up the file
        matched_file = None
        if gender and item_str:
            if gender in self.clothes_map and item_str in self.clothes_map[gender]:
                matched_file = self.clothes_map[gender][item_str][0]
        return matched_file

    def update_image(self, choose_random = None):
        pass
    
    def choose_random(self):
        gender = random.choice(list(self.clothes_map.keys()))
        item_str = random.choice(list(self.clothes_map[gender].keys()))
        file = random.choice(list(self.clothes_map[gender][item_str]))
        return file
    
    def label_to_css(self, queue_in,) -> None:
        start = time.time()
        while True:
            if not queue_in.empty():
                label = queue_in.get()
                now_time = time.time()
                if now_time - start < 2:
                    continue
                start = time.time()
                matched_file = self.parse_lable(label)
                print(label)
                self.update_image(matched_file)
            else:
                if time.time() - start > 5:
                    self.update_image(self.choose_random())
                    start = time.time()
                    print("update image from else")

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count


def app_callback(self, pad, info, user_data):
    """
    This is the callback function that will be called when data is available
    from the pipeline.
    Processing time should be kept to a minimum in this function.
    If longer processing is needed, consider using a separate thread / process.
    """
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK
    string_to_print = ""
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    if len(detections) == 0:
        detections = [roi] # Use the ROI as the detection
    # Parse the detections
    for detection in detections:
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id = None
        label = None
        confidence = 0.0
        for track_id_obj in track:
            track_id = track_id_obj.get_id()
        if track_id is not None:
            string_to_print += f'Track ID: {track_id} '
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        if len(classifications) > 0:
            string_to_print += ' CLIP Classifications:'
            for classification in classifications:
                label = classification.get_label()
                user_data.labels_queue.put(label)
                confidence = classification.get_confidence()
                string_to_print += f'Label: {label} Confidence: {confidence:.2f} '
            string_to_print += '\n'
        if isinstance(detection, hailo.HailoDetection):
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
    # if string_to_print:
        # print(string_to_print)
    return Gst.PadProbeReturn.OK
