import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

class app_callback_class:
    def __init__(self, shared_data, shared_config, shared_lock, hysteresis_threshold=5):
        self.frame_count = 0
        self.shared_data = shared_data
        self.shared_config = shared_config
        self.shared_lock = shared_lock
        self.current_id = 0
        self.current_x = 0.5
        self.current_y = 0.5
        self.hysteresis_threshold = hysteresis_threshold
        self.hysteresis_counter = 0

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def update_user_data(self, detection_data):
        new_best_detection = None
        best_score = float('-inf')
        current_detection = None

        for detection in detection_data:
            if 'detection' not in detection or 'coordinates' not in detection:
                continue

            bbox = detection['detection']['bbox']
            area = bbox['width'] * bbox['height']

            # Calculate score based on area only
            score = area

            if score > best_score:
                best_score = score
                new_best_detection = detection

            if detection['track_id'] == self.current_id:
                current_detection = detection

        if new_best_detection:
            if self.current_id == 0 or new_best_detection['track_id'] != self.current_id:
                self.hysteresis_counter += 1
                if self.hysteresis_counter >= self.hysteresis_threshold:
                    self.current_id = new_best_detection['track_id']
                    current_detection = new_best_detection
                    self.hysteresis_counter = 0
            else:
                self.hysteresis_counter = 0
        else:
            self.hysteresis_counter += 1
            if self.hysteresis_counter >= self.hysteresis_threshold:
                self.current_id = 0
                self.current_x = 0.5
                self.current_y = 0.5
                self.hysteresis_counter = 0

        # Update coordinates if the current detection is found
        if current_detection:
            self.current_x = current_detection['coordinates']['x']
            self.current_y = current_detection['coordinates']['y']


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
    detection_data = []
    for detection in detections:
        detection_info = {}
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id = None
        label = None
        confidence = 0.0

        for track_id_obj in track:
            track_id = track_id_obj.get_id()
        if track_id is None:
            continue  # Skip this detection if track_id is None

        detection_info['track_id'] = track_id

        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        if len(classifications) > 0:
            detection_info['classifications'] = []
            for classification in classifications:
                label = classification.get_label()
                confidence = classification.get_confidence()
                detection_info['classifications'].append({
                    'label': label,
                    'confidence': confidence
                })

        if isinstance(detection, hailo.HailoDetection):
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            detection_info['detection'] = {
                'label': label,
                'confidence': confidence,
                'bbox': {
                    'xmin': bbox.xmin(),
                    'ymin': bbox.ymin(),
                    'width': bbox.width(),
                    'height': bbox.height()
                }
            }
            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                nose = points[0]
                nose_x = (nose.x() * bbox.width() + bbox.xmin())
                nose_y = (nose.y() * bbox.height() + bbox.ymin())
                detection_info['coordinates'] = {
                    'x': nose_x,
                    'y': nose_y
                }

        detection_data.append(detection_info)
    user_data.update_user_data(detection_data)
    print(f"current_id: {user_data.current_id}, current_x: {user_data.current_x}, current_y: {user_data.current_y}")
    # Update shared data
    with user_data.shared_lock:
        user_data.shared_data['eyes_x'] = user_data.current_x
        user_data.shared_data['eyes_y'] = 1.0 - user_data.current_y
        user_data.shared_data['neck_left_right'] = user_data.current_x

        string_to_print += f"eyes_x: {user_data.shared_data['eyes_x']}, eyes_y: {user_data.shared_data['eyes_y']}"

    if False:
        print(string_to_print)
    return Gst.PadProbeReturn.OK
