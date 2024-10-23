import os
import argparse
import logging
import sys
import signal
import importlib.util
from functools import partial
import gi
import threading
import cv2
import numpy as np
from multiprocessing import Manager, Process
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib
from clip_app.logger_setup import setup_logger, set_log_level
from clip_app.clip_pipeline import get_pipeline
from clip_app.text_image_matcher import text_image_matcher
from clip_app import gui
from clip_app import EyeController
try:
    from picamera2 import Picamera2
except ImportError:
    pass # Available only on Pi OS

import ipdb #TBD
# add logging
logger = setup_logger()
set_log_level(logger, logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo online CLIP app")
    parser.add_argument("--input", "-i", type=str, default="picamera", help="URI of the input stream. Default is /dev/video0. Use '--input demo' to use the demo video.")
    parser.add_argument("--detector", "-d", type=str, choices=["person", "face", "none"], default="person", help="Which detection pipeline to use.")
    parser.add_argument("--json-path", type=str, default=None, help="Path to JSON file to load and save embeddings. If not set, embeddings.json will be used.")
    parser.add_argument("--disable-sync", action="store_true",help="Disables display sink sync, will run as fast as possible. Relevant when using file source.")
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to a dot file.")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="Detection threshold.")
    parser.add_argument("--show-fps", "-f", action="store_true", help="Print FPS on sink.")
    parser.add_argument("--enable-callback", action="store_true", help="Enables the use of the callback function.")
    parser.add_argument("--callback-path", type=str, default=None, help="Path to the custom user callback file.")
    parser.add_argument("--disable-runtime-prompts", action="store_true", help="When set, app will not support runtime prompts. Default is False.")

    return parser.parse_args()

def load_custom_callback(callback_path=None):
    if callback_path:
        spec = importlib.util.spec_from_file_location("custom_callback", callback_path)
        custom_callback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_callback)
    else:
        import clip_app.user_callback as custom_callback
    return custom_callback

def on_destroy(window):
    logger.info("Destroying window...")
    window.quit_button_clicked(None)

def main():
    args = parse_arguments()
    # args.enable_callback = True
    # Define Manager and keep it active
    with Manager() as manager:
        shared_lock = manager.Lock()
        shared_data = manager.dict(EyeController.EyeDataController.get_shared_data_structure())
        shared_config = manager.dict(EyeController.EyeDataController.get_shared_config_structure())
        # Setup EyeController
        shared_config['eyes_x']['smoothing'] = 1.0
        shared_data['update_config'] = True

        eye_process = Process(target=EyeController.run_eye_controller, args=(shared_data, shared_config, shared_lock))
        eye_process.start()
        # Create an instance of the user app callback class
        custom_callback_module = load_custom_callback(args.callback_path)
        app_callback = custom_callback_module.app_callback
        app_callback_class = custom_callback_module.app_callback_class

        logger = setup_logger()
        set_log_level(logger, logging.INFO)
        user_data = app_callback_class(shared_data, shared_config, shared_lock)
        win = AppWindow(args, user_data, app_callback)
        win.connect("destroy", on_destroy)
        win.show_all()
        Gtk.main()
        # close eye process
        eye_process.terminate()
        eye_process.join()

class AppWindow(Gtk.Window):
    # Add GUI functions to the AppWindow class
    build_ui = gui.build_ui
    add_text_boxes = gui.add_text_boxes
    update_text_boxes = gui.update_text_boxes
    update_text_prefix = gui.update_text_prefix
    quit_button_clicked = gui.quit_button_clicked
    on_text_box_updated = gui.on_text_box_updated
    on_slider_value_changed = gui.on_slider_value_changed
    on_negative_check_button_toggled = gui.on_negative_check_button_toggled
    on_ensemble_check_button_toggled = gui.on_ensemble_check_button_toggled
    on_load_button_clicked = gui.on_load_button_clicked
    on_save_button_clicked = gui.on_save_button_clicked
    update_progress_bars = gui.update_progress_bars
    on_track_id_update = gui.on_track_id_update
    disable_text_boxes = gui.disable_text_boxes

    # Add the get_pipeline function to the AppWindow class
    get_pipeline = get_pipeline

    if True: #TBD
        def on_debug_button_clicked(self, button):
            logger.info("Debug button clicked")
            with self.user_data.shared_lock:
                ipdb.set_trace()
                # self.user_data.shared_data['eyes_x'] = 0.3


                # # self.user_data.shared_config['eyes_x']['smoothing'] = 0.3
                # # To update the shared_config dictionary, you need to access the nested dictionary,
                # # modify it, and set it back to shared_config
                # # Access the nested shared_config dictionary
                # config = self.user_data.shared_config['neck_left_right']
                # # Modify the nested dictionary
                # config['smoothing'] = 0.1
                # # Set the modified dictionary back to shared_config
                # self.user_data.shared_config['neck_left_right'] = config
                # # Verify the change
                # print(self.user_data.shared_config['neck_left_right']['smoothing'])
                # # To trigger the update, set the 'update_config' key to True
                # self.user_data.shared_data['update_config'] = True

            return True


    def __init__(self, args, user_data, app_callback):
        Gtk.Window.__init__(self, title="Clip App")
        self.set_border_width(10)
        self.set_default_size(1, 1)
        self.fullscreen_mode = False
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        # move self.current_path one directory up to get the path to the workspace
        self.current_path = os.path.dirname(self.current_path)
        os.environ["GST_DEBUG_DUMP_DOT_DIR"] = self.current_path

        self.tappas_postprocess_dir = os.environ.get('TAPPAS_POST_PROC_DIR', '')
        if self.tappas_postprocess_dir == '':
            logger.error("TAPPAS_POST_PROC_DIR environment variable is not set. Please set it by sourcing setup_env.sh")
            sys.exit(1)

        self.dump_dot = args.dump_dot
        self.sync_req = 'false' if args.disable_sync else 'true'
        self.show_fps = args.show_fps
        self.enable_callback = args.enable_callback or args.callback_path is not None
        self.json_file = os.path.join(self.current_path, "embeddings.json") if args.json_path is None else args.json_path
        if args.input == "demo":
            self.input_uri = os.path.join(self.current_path, "resources", "clip_example.mp4")
            self.json_file = os.path.join(self.current_path, "example_embeddings.json") if args.json_path is None else args.json_path
        else:
            self.input_uri = args.input
        self.detector = args.detector
        self.user_data = user_data
        self.app_callback = app_callback

        self.video_width = 1280
        self.video_height = 720
        self.network_format = "RGB"

        # get current path
        Gst.init(None)
        self.pipeline = self.create_pipeline()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        # get xvimagesink element and disable qos
        # xvimagesink is instantiated by fpsdisplaysink
        hailo_display = self.pipeline.get_by_name("hailo_display")
        xvimagesink = hailo_display.get_by_name("xvimagesink0")
        xvimagesink.set_property("qos", False)

        # get text_image_matcher instance
        self.text_image_matcher = text_image_matcher
        self.text_image_matcher.set_threshold(args.detection_threshold)

        # Add text_image_matcher to callback data
        user_data.text_image_matcher = self.text_image_matcher

        # picamera support
        self.processes = []
        self.source_type = "picamera" #TBD
        if self.source_type == "picamera":
            # use threading instead of multiprocessing
            picam_proc = threading.Thread(target=self.picamera_process)
            self.processes.append(picam_proc)
            picam_proc.start()

        # build UI
        self.build_ui(args)

        if True: #TBD
            # Create a vertical box to hold all widgets
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.add(vbox)

            # Add debug button
            self.debug_button = Gtk.Button(label="Debug")
            self.debug_button.connect("clicked", self.on_debug_button_clicked)
            self.ui_vbox.pack_start(self.debug_button, False, False, 0)


        # set runtime
        if args.disable_runtime_prompts:
            logger.info("No text embedding runtime selected, adding new text is disabled. Loading %s", self.json_file)
            self.disable_text_boxes()
            self.on_load_button_clicked(None)
        else:
            self.text_image_matcher.init_clip()

        if self.text_image_matcher.model_runtime is not None:
            logger.info("Using %s for text embedding", self.text_image_matcher.model_runtime)
            self.on_load_button_clicked(None)

        # Connect pad probe to the identity element
        if self.enable_callback:
            identity = self.pipeline.get_by_name("identity_callback")
            if identity is None:
                logger.warning("identity_callback element not found, add <identity name=identity_callback> in your pipeline where you want the callback to be called.")
            else:
                identity_pad = identity.get_static_pad("src")
                identity_pad.add_probe(Gst.PadProbeType.BUFFER, partial(self.app_callback, self), self.user_data)

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        # start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

        if self.dump_dot:
            GLib.timeout_add_seconds(5, self.dump_dot_file)

        self.update_text_boxes()

        # Define a timeout duration in nanoseconds (e.g., 5 seconds)
        timeout_ns = 5 * Gst.SECOND

        # Wait until state change is done or until the timeout occurs
        state_change_return, _state, _pending = self.pipeline.get_state(timeout_ns)

        if state_change_return == Gst.StateChangeReturn.SUCCESS:
            logger.info("Pipeline state changed to PLAYING successfully.")
        elif state_change_return == Gst.StateChangeReturn.ASYNC:
            logger.info("State change is ongoing asynchronously.")
        elif state_change_return == Gst.StateChangeReturn.FAILURE:
            logger.info("State change failed.")
        else:
            logger.warning("Unknown state change return value.")

    def picamera_process(self, picamera_config=None):
        appsrc = self.pipeline.get_by_name("app_source")
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)
        print("appsrc properties: ", appsrc)

        # Initialize Picamera2
        with Picamera2() as picam2:
            # Default configuration
            main = {'size': (2560, 3560), 'format': 'RGB888'}
            lores = {'size': (self.video_width, self.video_height), 'format': 'RGB888'}

            controls = {'FrameRate': 30}
            config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)

            # Configure the camera with the created configuration
            picam2.configure(config)

            # Update GStreamer caps based on 'lores' stream
            lores_stream = config['lores']
            format_str = 'RGB' if lores_stream['format'] == 'RGB888' else self.network_format
            width, height = lores_stream['size']
            print(f"Picamera2 configuration: width={width}, height={height}, format={format_str}")
            appsrc.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, format={format_str}, width={width}, height={height}, "
                    f"framerate=30/1, pixel-aspect-ratio=1/1"
                )
            )

            picam2.start()

            frame_count = 0
            print("picamera_process started")
            while True:
                frame_data = picam2.capture_array('lores')
                if frame_data is None:
                    print("Failed to capture frame.")
                    break

                # Convert frame data if necessary
                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                frame = np.ascontiguousarray(frame)

                # Create Gst.Buffer by wrapping the frame data
                buffer = Gst.Buffer.new_wrapped(frame.tobytes())

                # Set buffer PTS and duration
                buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
                buffer.pts = frame_count * buffer_duration
                buffer.duration = buffer_duration

                # Push the buffer to appsrc
                ret = appsrc.emit('push-buffer', buffer)
                if ret != Gst.FlowReturn.OK:
                    print("Failed to push buffer:", ret)
                    break

                frame_count += 1


    def dump_dot_file(self):
        logger.info("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False


    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("Error: %s %s", err, debug)
            self.shutdown()
        # print QOS messages
        elif t == Gst.MessageType.QOS:
            # print which element is reporting QOS
            src = message.src.get_name()
            logger.info("QOS from %s", src)
        return True


    def on_eos(self):
        logger.info("EOS received, shutting down the pipeline.")
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.exit_application)

    def shutdown(self, signum=None, frame=None):
        # Shutdown EyeController
        with self.user_data.shared_lock:
            self.user_data.shared_data['enable'] = False # Disable EyeController
            self.user_data.shared_data['shutdown_flag'] = True # Set shutdown flag
        # Shutdown the pipeline
        logger.info("Sending EOS event to the pipeline...")
        # Send EOS to the source element if possible
        source = self.pipeline.get_by_name('source')
        if source:
            source.send_event(Gst.Event.new_eos())
        else:
            self.pipeline.send_event(Gst.Event.new_eos())
        # Schedule forced exit after a timeout
        GLib.timeout_add_seconds(5, self.force_exit)

    def exit_application(self):
        logger.info("Exiting application...")
        for proc in self.processes:
            # proc.stop()
            proc.join()
        # Destroy the window
        self.destroy()
        # Quit the GTK main loop
        Gtk.main_quit()
        return False  # Remove the idle callback

    def force_exit(self):
        logger.warning("Forced exit after timeout.")
        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.exit_application)
        return False  # Remove the timeout callback

    def create_pipeline(self):
        pipeline_str = get_pipeline(self)
        logger.info('PIPELINE:\ngst-launch-1.0 %s', pipeline_str)
        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            logger.error("An error occurred while parsing the pipeline: %s", e)
        return pipeline

if __name__ == "__main__":
    main()
