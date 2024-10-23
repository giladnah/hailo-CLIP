import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import signal
import threading
# Try to import hailo python module
try:
    import hailo
except ImportError:
    sys.exit("Failed to import hailo python module. Make sure you are in hailo virtual environment.")

try:
    from picamera2 import Picamera2
except ImportError:
    pass # Available only on Pi OS

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# A sample class to be used in the callback function
# This example allows to:
# 1. Count the number of frames
# 2. Setup a multiprocessing queue to pass the frame to the main thread
# Additional variables and functions can be added to this class as needed

class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.frame_queue = multiprocessing.Queue(maxsize=3)
        self.running = True

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        else:
            return None

def dummy_callback(pad, info, user_data):
    """
    A minimal dummy callback function that returns immediately.

    Args:
        pad: The GStreamer pad
        info: The probe info
        user_data: User-defined data passed to the callback

    Returns:
        Gst.PadProbeReturn.OK
    """
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Common functions
# -----------------------------------------------------------------------------------------------
def get_caps_from_pad(pad: Gst.Pad):
    caps = pad.get_current_caps()
    if caps:
        # We can now extract information from the caps
        structure = caps.get_structure(0)
        if structure:
            # Extracting some common properties
            format = structure.get_value('format')
            width = structure.get_value('width')
            height = structure.get_value('height')
            return format, width, height
    else:
        return None, None, None

# This function is used to display the user data frame
def display_user_data_frame(user_data: app_callback_class):
    while user_data.running:
        frame = user_data.get_frame()
        if frame is not None:
            cv2.imshow("User Frame", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def get_default_parser():
    parser = argparse.ArgumentParser(description="Hailo App Help")
    parser.add_argument(
        "--input", "-i", type=str, default="picamera",
        help="Input source. Can be a file, USB or RPi camera (CSI camera module). \
        For RPi camera use '-i rpi' (Still in Beta). \
        Defaults to /dev/video0"
    )
    parser.add_argument("--use-frame", "-u", action="store_true", help="Use frame from the callback function")
    parser.add_argument("--show-fps", "-f", action="store_true", help="Print FPS on sink")
    parser.add_argument(
        "--disable-sync", action="store_true",
        help="Disables display sink sync, will run as fast as possible. Relevant when using file source."
    )
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to a dot file pipeline.dot")
    return parser

#---------------------------------------------------------
# Pipeline helper functions
#---------------------------------------------------------

def get_source_type(input_source):
    # This function will return the source type based on the input source
    # return values can be "file", "mipi" or "usb"
    print("input_source: ", input_source)
    if input_source.startswith("/dev/video"):
        return 'usb'
    elif input_source.startswith("rpi"):
        return 'rpi'
    elif input_source.startswith("picamera"):
        return 'picamera'
    else:
        return 'file'

def QUEUE(name, max_size_buffers=3, max_size_bytes=0, max_size_time=0, leaky='no'):
    """
    Creates a GStreamer queue element string with the specified parameters.

    Args:
        name (str): The name of the queue element.
        max_size_buffers (int, optional): The maximum number of buffers that the queue can hold. Defaults to 3.
        max_size_bytes (int, optional): The maximum size in bytes that the queue can hold. Defaults to 0 (unlimited).
        max_size_time (int, optional): The maximum size in time that the queue can hold. Defaults to 0 (unlimited).
        leaky (str, optional): The leaky type of the queue. Can be 'no', 'upstream', or 'downstream'. Defaults to 'no'.

    Returns:
        str: A string representing the GStreamer queue element with the specified parameters.
    """
    q_string = f'queue name={name} leaky={leaky} max-size-buffers={max_size_buffers} max-size-bytes={max_size_bytes} max-size-time={max_size_time} '
    return q_string

def SOURCE_PIPELINE(video_source, video_format='RGB', video_width=640, video_height=640, name='source'):
    """
    Creates a GStreamer pipeline string for the video source.

    Args:
        video_source (str): The path or device name of the video source.
        video_format (str, optional): The video format. Defaults to 'RGB'.
        video_width (int, optional): The width of the video. Defaults to 640.
        video_height (int, optional): The height of the video. Defaults to 640.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'source'.

    Returns:
        str: A string representing the GStreamer pipeline for the video source.
    """
    source_type = get_source_type(video_source)

    if source_type == 'rpi':
        source_element = (
            f'libcamerasrc name={name} ! '
            f'video/x-raw, format={video_format}, width=1536, height=864 ! '
        )
    elif source_type == 'usb':
        source_element = (
            f'v4l2src device={video_source} name={name} ! image/jpeg, framerate=30/1 ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f'decodebin name={name}_decodebin ! '
        )
    elif source_type == 'picamera':
        source_element = (
            f'appsrc name=app_source is-live=true leaky-type=downstream max-buffers=2 ! '
            # f'video/x-raw, format={video_format}, width={video_height}, height={video_width} ! '
            'videoflip name=videoflip video-direction=horiz ! '
            f'video/x-raw, format={video_format}, width={video_width}, height={video_height} ! '
        )
    else:
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f'{QUEUE(name=f"{name}_queue_dec264")} ! '
            'qtdemux ! h264parse ! avdec_h264 max-threads=2 ! '
        )
    source_pipeline = (
        f'{source_element} '
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f'videoscale name={name}_videoscale n-threads=2 ! '
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f'videoconvert n-threads=3 name={name}_convert qos=false ! '
        # f'video/x-raw, format={video_format}, pixel-aspect-ratio=1/1 ! '
        f'video/x-raw, format={video_format}, width={video_width}, height={video_height} ! '
    )

    return source_pipeline

def INFERENCE_PIPELINE(hef_path, post_process_so, batch_size=1, config_json=None, post_function_name=None, additional_params='', name='inference'):
    """
    Creates a GStreamer pipeline string for inference and post-processing using a user-provided shared object file.
    This pipeline includes videoscale and videoconvert elements to convert the video frame to the required format.
    The format and resolution are automatically negotiated based on the HEF file requirements.

    Args:
        hef_path (str): The path to the HEF file.
        post_process_so (str): The path to the post-processing shared object file.
        batch_size (int, optional): The batch size for the hailonet element. Defaults to 1.
        config_json (str, optional): The path to the configuration JSON file. If None, no configuration is added. Defaults to None.
        post_function_name (str, optional): The name of the post-processing function. If None, no function name is added. Defaults to None.
        additional_params (str, optional): Additional parameters for the hailonet element. Defaults to ''.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'inference'.

    Returns:
        str: A string representing the GStreamer pipeline for inference.
    """
    # Configure config path if provided
    if config_json is not None:
        config_str = f' config-path={config_json} '
    else:
        config_str = ''

    # Configure function name if provided
    if post_function_name is not None:
        function_name_str = f' function-name={post_function_name} '
    else:
        function_name_str = ''

    # Construct the inference pipeline string
    inference_pipeline = (
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f'videoscale name={name}_videoscale n-threads=2 qos=false ! '
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name={name}_videoconvert n-threads=2 ! '
        f'{QUEUE(name=f"{name}_hailonet_q")} ! '
        f'hailonet name={name}_hailonet hef-path={hef_path} batch-size={batch_size} {additional_params} force-writable=true ! '
        f'{QUEUE(name=f"{name}_hailofilter_q")} ! '
        f'hailofilter name={name}_hailofilter so-path={post_process_so} {config_str} {function_name_str} qos=false '
    )

    return inference_pipeline

def DETECTION_PIPELINE(hef_path, batch_size=1, labels_json=None, additional_params='', name='detection'):
    """
    Creates a GStreamer pipeline string for detection inference, using HailoRT post-processing.
    This pipeline is compatible with detection models which is compiled with HailoRT post-processing.

    Args:
        hef_path (str): The path to the HEF file.
        batch_size (int, optional): The batch size for the hailonet element. Defaults to 1.
        labels_json (str, optional): The path to the labels JSON file. If None, no labels are added. Defaults to None.
        additional_params (str, optional): Additional parameters for the hailonet element. Defaults to ''.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'detection'.

    Returns:
        str: A string representing the GStreamer pipeline for detection.
    """
    post_process_so = os.path.join(os.environ.get('TAPPAS_POST_PROC_DIR', ''), 'libyolo_hailortpp_post.so')
    detection_pipeline = INFERENCE_PIPELINE(hef_path=hef_path, post_process_so=post_process_so, batch_size=batch_size, config_json=labels_json, additional_params=additional_params, name=name)
    return detection_pipeline

def INFERENCE_PIPELINE_WRAPPER(inner_pipeline, bypass_max_size_buffers=20, name='inference_wrapper'):
    """
    Creates a GStreamer pipeline string that wraps an inner pipeline with a hailocropper and hailoaggregator.
    This allows to keep the original video resolution and color-space (format) of the input frame.
    The inner pipeline should be able to do the required conversions and rescale the detection to the original frame size.

    Args:
        inner_pipeline (str): The inner pipeline string to be wrapped.
        bypass_max_size_buffers (int, optional): The maximum number of buffers for the bypass queue. Defaults to 20.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'inference_wrapper'.

    Returns:
        str: A string representing the GStreamer pipeline for the inference wrapper.
    """
    # Get the directory for post-processing shared objects
    tappas_post_process_dir = os.environ.get('TAPPAS_POST_PROC_DIR', '')
    whole_buffer_crop_so = os.path.join(tappas_post_process_dir, 'cropping_algorithms/libwhole_buffer.so')

    # Construct the inference wrapper pipeline string
    inference_wrapper_pipeline = (
        f'{QUEUE(name=f"{name}_input_q")} ! '
        f'hailocropper name={name}_crop so-path={whole_buffer_crop_so} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true '
        f'hailoaggregator name={name}_agg '
        f'{name}_crop. ! {QUEUE(max_size_buffers=bypass_max_size_buffers, name=f"{name}_bypass_q")} ! {name}_agg.sink_0 '
        f'{name}_crop. ! {inner_pipeline} ! {name}_agg.sink_1 '
        f'{name}_agg. ! {QUEUE(name=f"{name}_output_q")} '
    )

    return inference_wrapper_pipeline

def DISPLAY_PIPELINE(video_sink='xvimagesink', sync='true', show_fps='false', name='hailo_display'):
    """
    Creates a GStreamer pipeline string for displaying the video.
    It includes the hailooverlay plugin to draw bounding boxes and labels on the video.

    Args:
        video_sink (str, optional): The video sink element to use. Defaults to 'xvimagesink'.
        sync (str, optional): The sync property for the video sink. Defaults to 'true'.
        show_fps (str, optional): Whether to show the FPS on the video sink. Should be 'true' or 'false'. Defaults to 'false'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'hailo_display'.

    Returns:
        str: A string representing the GStreamer pipeline for displaying the video.
    """
    # Construct the display pipeline string
    display_pipeline = (
        f'{QUEUE(name=f"{name}_hailooverlay_q")} ! '
        f'hailooverlay name={name}_hailooverlay ! '
        f'{QUEUE(name=f"{name}_videoconvert_q")} ! '
        f'videoconvert name={name}_videoconvert n-threads=2 qos=false ! '
        f'{QUEUE(name=f"{name}_q")} ! '
        f'fpsdisplaysink name={name} video-sink={video_sink} sync={sync} text-overlay={show_fps} signal-fps-measurements=true '
    )

    return display_pipeline

def USER_CALLBACK_PIPELINE(name='identity_callback'):
    """
    Creates a GStreamer pipeline string for the user callback element.

    Args:
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'identity_callback'.

    Returns:
        str: A string representing the GStreamer pipeline for the user callback element.
    """
    # Construct the user callback pipeline string
    user_callback_pipeline = (
        f'{QUEUE(name=f"{name}_q")} ! '
        f'identity name={name} '
    )

    return user_callback_pipeline
# -----------------------------------------------------------------------------------------------
# GStreamerApp class
# -----------------------------------------------------------------------------------------------
class GStreamerApp:
    def __init__(self, args, user_data: app_callback_class):
        # Set the process title
        setproctitle.setproctitle("Hailo Python App")

        # Create an empty options menu
        self.options_menu = args

        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)

        # Initialize variables
        tappas_post_process_dir = os.environ.get('TAPPAS_POST_PROC_DIR', '')
        if tappas_post_process_dir == '':
            print("TAPPAS_POST_PROC_DIR environment variable is not set. Please set it to by sourcing setup_env.sh")
            exit(1)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.postprocess_dir = tappas_post_process_dir
        self.video_source = self.options_menu.input
        self.source_type = get_source_type(self.video_source)
        self.user_data = user_data
        self.video_sink = "xvimagesink"
        self.pipeline = None
        self.loop = None
        self.processes = []

        # Set Hailo parameters; these parameters should be set based on the model used
        self.batch_size = 1
        self.video_width = 640
        self.video_height = 640
        self.network_format = "RGB"
        self.hef_path = None
        self.app_callback = None

        # Set user data parameters
        user_data.use_frame = self.options_menu.use_frame

        self.sync = "false" if (self.options_menu.disable_sync or self.source_type != "file") else "true"
        self.show_fps = "true" if self.options_menu.show_fps else "false"

        if self.options_menu.dump_dot:
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = self.current_path

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def create_pipeline(self):
        # Initialize GStreamer
        Gst.init(None)

        pipeline_string = self.get_pipeline_string()
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except Exception as e:
            print(e)
            print(pipeline_string)
            sys.exit(1)

        # Connect to hailo_display fps-measurements
        if self.show_fps:
            print("Showing FPS")
            self.pipeline.get_by_name("hailo_display").connect("fps-measurements", self.on_fps_measurement)

        # Create a GLib Main Loop
        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.shutdown()
        # QOS
        elif t == Gst.MessageType.QOS:
            # Handle QoS message here
            qos_element = message.src.get_name()
            print(f"QoS message received from {qos_element}")
        return True


    def on_eos(self):
        if self.source_type == "file":
             # Seek to the start (position 0) in nanoseconds
            success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
            if success:
                print("Video rewound successfully. Restarting playback...")
            else:
                print("Error rewinding the video.")
        else:
            self.shutdown()


    def shutdown(self, signum=None, frame=None):
        print("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.loop.quit)


    def get_pipeline_string(self):
        # This is a placeholder function that should be overridden by the child class
        return ""

    def dump_dot_file(self):
        print("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False

    def run(self):
        # Add a watch for messages on the pipeline's bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        # Connect pad probe to the identity element
        identity = self.pipeline.get_by_name("identity_callback")
        if identity is None:
            print("Warning: identity_callback element not found, add <identity name=identity_callback> in your pipeline where you want the callback to be called.")
        else:
            identity_pad = identity.get_static_pad("src")
            identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self.user_data)

        hailo_display = self.pipeline.get_by_name("hailo_display")
        if hailo_display is None:
            print("Warning: hailo_display element not found, add <fpsdisplaysink name=hailo_display> to your pipeline to support fps display.")
        else:
            xvimagesink = hailo_display.get_by_name("xvimagesink0")
            if xvimagesink is not None:
                xvimagesink.set_property("qos", False)

        # Disable QoS to prevent frame drops
        disable_qos(self.pipeline)

        # Start a subprocess to run the display_user_data_frame function
        if self.options_menu.use_frame:
            display_process = multiprocessing.Process(target=display_user_data_frame, args=(self.user_data,))
            display_process.start()

        if self.source_type == "picamera":
            # use threading instead of multiprocessing
            picam_proc = threading.Thread(target=self.picamera_process)
            self.processes.append(picam_proc)
            picam_proc.start()

        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)

        # Dump dot file
        if self.options_menu.dump_dot:
            GLib.timeout_add_seconds(3, self.dump_dot_file)

        # Run the GLib event loop
        self.loop.run()

        # Clean up
        self.user_data.running = False
        self.pipeline.set_state(Gst.State.NULL)
        if self.options_menu.use_frame:
            display_process.terminate()
            display_process.join()
        for proc in self.processes:
            proc.terminate()
            proc.join()



    def picamera_process(self, picamera_config=None):
        appsrc = self.pipeline.get_by_name("app_source")
        appsrc.set_property("is-live", True)
        print("appsrc properties: ", appsrc)

        # Initialize Picamera2
        with Picamera2() as picam2:
            # Default configuration
            main = {'size': (1920, 1080), 'format': 'RGB888'}
            lores = {'size': (self.video_width, self.video_height), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
            if picamera_config is not None:
                # Ensure 'main' and 'lores' streams are present
                if 'lores' not in picamera_config or 'main' not in picamera_config:
                    print("User configuration is missing 'main' or 'lores' stream. Using default configuration.")
                else:
                    # Use the provided configuration directly
                    config = picam2.create_preview_configuration(
                        main=picamera_config['main'],
                        lores=picamera_config['lores'],
                        controls=picamera_config.get('controls', {})
                    )

            # Configure the camera with the created configuration
            picam2.configure(config)

            # Update GStreamer caps based on 'lores' stream
            lores_stream = config['lores']
            format_str = 'RGB' if lores_stream['format'] == 'RGB888' else self.network_format
            width, height = lores_stream['size']
            appsrc.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, format={format_str}, width={width}, height={height}, "
                    f"framerate=30/1, pixel-aspect-ratio=1/1"
                )
            )

            picam2.start()

            first_frame = True
            pipeline_clock = None
            print("picamera_process started")
            while True:
                frame_data = picam2.capture_array('lores')
                if frame_data is None:
                    print("Failed to capture frame.")
                    break
                # Corrected lines
                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                frame = np.ascontiguousarray(frame)
                # Create Gst.Buffer by wrapping the frame data
                buffer = Gst.Buffer.new_wrapped(frame.tobytes())
                if first_frame:
                    buffer.pts = 0
                    buffer.dts = Gst.CLOCK_TIME_NONE
                    buffer.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
                else:
                    if pipeline_clock is None:
                        pipeline_clock = self.pipeline.get_clock()
                        if not pipeline_clock:
                            print("Failed to get pipeline clock.")
                            break
                    current_time = pipeline_clock.get_time()
                    buffer.pts = current_time
                    buffer.dts = buffer.pts
                    buffer.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)

                # Push the buffer to appsrc
                ret = appsrc.emit('push-buffer', buffer)
                if ret != Gst.FlowReturn.OK:
                    print("Failed to push buffer:", ret)
                    break


                if first_frame:
                    first_frame = False
                    # Wait for the pipeline to reach the PLAYING state
                    print("Waiting for the pipeline to reach the PLAYING state")
                    state_change_return, current_state, pending_state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
                    print(f"state_change_return: {state_change_return}, current_state: {current_state}, pending_state: {pending_state}")
                    if current_state != Gst.State.PLAYING:
                        print(f"Pipeline failed to reach PLAYING state. Current state: {current_state}")
                        return
                    else:
                        # Get the pipeline clock after the pipeline is playing
                        pipeline_clock = self.pipeline.get_clock()
                        if not pipeline_clock:
                            print("Failed to get pipeline clock.")
                            return

# ---------------------------------------------------------
# Functions used to get numpy arrays from GStreamer buffers
# ---------------------------------------------------------

def handle_rgb(map_info, width, height):
    # The copy() method is used to create a copy of the numpy array. This is necessary because the original numpy array is created from buffer data, and it does not own the data it represents. Instead, it's just a view of the buffer's data.
    return np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()

def handle_nv12(map_info, width, height):
    y_plane_size = width * height
    uv_plane_size = width * height // 2
    y_plane = np.ndarray(shape=(height, width), dtype=np.uint8, buffer=map_info.data[:y_plane_size]).copy()
    uv_plane = np.ndarray(shape=(height//2, width//2, 2), dtype=np.uint8, buffer=map_info.data[y_plane_size:]).copy()
    return y_plane, uv_plane

def handle_yuyv(map_info, width, height):
    return np.ndarray(shape=(height, width, 2), dtype=np.uint8, buffer=map_info.data).copy()

FORMAT_HANDLERS = {
    'RGB': handle_rgb,
    'NV12': handle_nv12,
    'YUYV': handle_yuyv,
}

def get_numpy_from_buffer(buffer, format, width, height):
    """
    Converts a GstBuffer to a numpy array based on provided format, width, and height.

    Args:
        buffer (GstBuffer): The GStreamer Buffer to convert.
        format (str): The video format ('RGB', 'NV12', 'YUYV', etc.).
        width (int): The width of the video frame.
        height (int): The height of the video frame.

    Returns:
        np.ndarray: A numpy array representing the buffer's data, or a tuple of arrays for certain formats.
    """
    # Map the buffer to access data
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        raise ValueError("Buffer mapping failed")

    try:
        # Handle different formats based on the provided format parameter
        handler = FORMAT_HANDLERS.get(format)
        if handler is None:
            raise ValueError(f"Unsupported format: {format}")
        return handler(map_info, width, height)
    finally:
        buffer.unmap(map_info)

# ---------------------------------------------------------
# Useful functions for working with GStreamer
# ---------------------------------------------------------

def disable_qos(pipeline):
    """
    Iterate through all elements in the given GStreamer pipeline and set the qos property to False
    where applicable.
    When the 'qos' property is set to True, the element will measure the time it takes to process each buffer and will drop frames if latency is too high.
    We are running on long pipelines, so we want to disable this feature to avoid dropping frames.
    :param pipeline: A GStreamer pipeline object
    """
    # Ensure the pipeline is a Gst.Pipeline instance
    if not isinstance(pipeline, Gst.Pipeline):
        print("The provided object is not a GStreamer Pipeline")
        return

    # Iterate through all elements in the pipeline
    it = pipeline.iterate_elements()
    while True:
        result, element = it.next()
        if result != Gst.IteratorResult.OK:
            break

        # Check if the element has the 'qos' property
        if 'qos' in GObject.list_properties(element):
            # Set the 'qos' property to False
            element.set_property('qos', False)
            print(f"Set qos to False for {element.get_name()}")
