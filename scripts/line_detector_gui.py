#!/usr/bin/env python3
"""
Line Detector Parameter Tuning GUI

A GUI tool for real-time parameter tuning of the etrobo_line_detector node.
Displays the image_with_lines topic and allows interactive parameter adjustment.

Updated to work with AdaptiveLineTracker - removed edge detection parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, Any, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


class LineDetectorParameterGUI:
    """Main GUI class for parameter tuning"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Line Detector Parameter Tuner")
        self.root.geometry("1200x800")

        # ROS2 components
        self.ros_node = None
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=1)
        self.current_image = None

        # Parameter tracking
        self.parameters = {}
        self.parameter_widgets = {}

        # GUI setup
        self.setup_gui()
        self.setup_ros()

        # Start image update thread
        self.update_thread_running = True
        self.update_thread = threading.Thread(
            target=self.update_image_display, daemon=True)
        self.update_thread.start()

        # Fetch initial parameters from node (after GUI is setup)
        # Delay to ensure ROS is ready and node is running
        self.root.after(3000, self.fetch_initial_parameters)

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for parameters
        self.param_frame = ttk.Frame(main_frame, width=400)
        self.param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.param_frame.pack_propagate(False)

        # Right panel for image and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Image display
        self.setup_image_display(right_frame)

        # Status and controls
        self.setup_status_controls(right_frame)

        # Parameter sections
        self.setup_parameter_sections()

    def setup_image_display(self, parent):
        """Setup image display area"""
        image_frame = ttk.LabelFrame(parent, text="Image Display")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Canvas for image with double buffering
        self.image_canvas = tk.Canvas(image_frame, bg='black', width=640, height=480,
                                      highlightthickness=0)  # Remove border for smoother display
        self.image_canvas.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Image display variables
        self.canvas_image_id = None
        self.placeholder_text_id = self.image_canvas.create_text(320, 240, text="Waiting for image...",
                                                                 fill='white', font=('Arial', 16))

    def setup_status_controls(self, parent):
        """Setup status and control buttons"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.status_labels = {}
        self.status_labels['connection'] = ttk.Label(
            status_frame, text="Connection: Disconnected")
        self.status_labels['connection'].pack(anchor=tk.W, padx=5, pady=2)

        self.status_labels['frame'] = ttk.Label(status_frame, text="Frame: 0")
        self.status_labels['frame'].pack(anchor=tk.W, padx=5, pady=2)

        self.status_labels['lines'] = ttk.Label(status_frame, text="Lines: 0")
        self.status_labels['lines'].pack(anchor=tk.W, padx=5, pady=2)

        # Control buttons
        button_frame = ttk.LabelFrame(control_frame, text="Controls")
        button_frame.pack(side=tk.RIGHT)

        ttk.Button(button_frame, text="Reset",
                   command=self.reset_parameters).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Save", command=self.save_parameters).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Load", command=self.load_parameters).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Exit", command=self.on_closing).pack(
            side=tk.LEFT, padx=2)

    def setup_parameter_sections(self):
        """Setup parameter input sections"""
        # Scrollable frame for parameters
        canvas = tk.Canvas(self.param_frame)
        scrollbar = ttk.Scrollbar(
            self.param_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Parameter definitions organized by category
        self.parameter_definitions = {
            "I/O Settings": {
                "image_topic": {"type": "string", "default": "camera/image_raw"},
                "publish_image_with_lines": {"type": "bool", "default": True}
            },
            "Black Line Detection": {
                "roi_x": {"type": "int", "default": -1, "min": -1, "max": 1920},
                "roi_y": {"type": "int", "default": -1, "min": -1, "max": 1080},
                "roi_w": {"type": "int", "default": -1, "min": -1, "max": 1920},
                "roi_h": {"type": "int", "default": -1, "min": -1, "max": 1080},
                "hsv_lower_s": {"type": "int", "default": 0, "min": 0, "max": 255},
                "hsv_upper_s": {"type": "int", "default": 255, "min": 0, "max": 255},
                "hsv_upper_v": {"type": "int", "default": 80, "min": 0, "max": 255},
                "hsv_dilate_kernel": {"type": "int", "default": 3, "min": 1, "max": 21, "step": 2},
                "hsv_dilate_iter": {"type": "int", "default": 1, "min": 0, "max": 10},
                "show_mask": {"type": "bool", "default": False},
                "show_contours": {"type": "bool", "default": False}
            },
            "Blue Line Detection": {
                "blue_detection_enabled": {"type": "bool", "default": True},
                "blue_lower_h": {"type": "int", "default": 100, "min": 0, "max": 180},
                "blue_upper_h": {"type": "int", "default": 130, "min": 0, "max": 180},
                "blue_lower_s": {"type": "int", "default": 50, "min": 0, "max": 255},
                "blue_upper_s": {"type": "int", "default": 255, "min": 0, "max": 255},
                "blue_lower_v": {"type": "int", "default": 50, "min": 0, "max": 255},
                "blue_upper_v": {"type": "int", "default": 255, "min": 0, "max": 255}
            },
            "Line Tracking": {
                "line_scan_step": {"type": "int", "default": 5, "min": 1, "max": 20},
                "min_line_width": {"type": "double", "default": 6.0, "min": 1.0, "max": 50.0},
                "line_width_importance": {"type": "double", "default": 2.0, "min": 1.0, "max": 5.0},
                "min_contour_score": {"type": "double", "default": 10.0, "min": 1.0, "max": 50.0},
                "min_segments_straight": {"type": "int", "default": 5, "min": 1, "max": 20},
                "min_segments_curve": {"type": "int", "default": 3, "min": 1, "max": 20}
            },
            "Contour Tracking": {
                "tracker_enabled": {"type": "bool", "default": True},
                "tracker_max_missed_frames": {"type": "int", "default": 5, "min": 1, "max": 20},
                "tracker_max_distance": {"type": "double", "default": 75.0, "min": 10.0, "max": 200.0},
                "tracker_process_noise": {"type": "double", "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001},
                "tracker_measurement_noise": {"type": "double", "default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001},
                "tracker_speed_threshold": {"type": "double", "default": 5.0, "min": 1.0, "max": 20.0},
                "tracker_debug": {"type": "bool", "default": False}
            },
            "Line Merging": {
                "merger_enabled": {"type": "bool", "default": False},
                "merger_method": {"type": "choice", "choices": ["direction_endpoint", "kalman_graph"], "default": "direction_endpoint"},
                "merger_max_angle_diff": {"type": "double", "default": 20.0, "min": 5.0, "max": 90.0},
                "merger_max_endpoint_dist": {"type": "double", "default": 50.0, "min": 10.0, "max": 200.0},
                "merger_min_line_length": {"type": "double", "default": 30.0, "min": 10.0, "max": 100.0},
                "merger_prediction_frames": {"type": "int", "default": 5, "min": 1, "max": 20},
                "merger_trajectory_threshold": {"type": "double", "default": 30.0, "min": 10.0, "max": 100.0},
                "merger_confidence": {"type": "double", "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}
            },
            "Calibration": {
                "camera_height_meters": {"type": "double", "default": 0.2, "min": 0.05, "max": 1.0},
                "landmark_distance_meters": {"type": "double", "default": 0.59, "min": 0.1, "max": 2.0},
                "calib_timeout_sec": {"type": "double", "default": 60.0, "min": 0.0, "max": 300.0},
                "calib_hsv_s_max": {"type": "int", "default": 16, "min": 0, "max": 255},
                "calib_hsv_v_min": {"type": "int", "default": 100, "min": 0, "max": 255},
                "calib_hsv_v_max": {"type": "int", "default": 168, "min": 0, "max": 255},
                "calib_min_area": {"type": "int", "default": 80, "min": 10, "max": 1000},
                "calib_roi_x": {"type": "int", "default": 200, "min": -1, "max": 1920},
                "calib_roi_y": {"type": "int", "default": 150, "min": -1, "max": 1080},
                "calib_roi_w": {"type": "int", "default": 240, "min": -1, "max": 1920},
                "calib_roi_h": {"type": "int", "default": 180, "min": -1, "max": 1080}
            },
            "Localization": {
                "landmark_map_x": {"type": "double", "default": -0.409, "min": -5.0, "max": 5.0},
                "landmark_map_y": {"type": "double", "default": 1.0, "min": -5.0, "max": 5.0}
            }
        }

        # Create parameter widgets
        self.create_parameter_widgets(scrollable_frame)

    def create_parameter_widgets(self, parent):
        """Create widgets for all parameters"""
        for category, params in self.parameter_definitions.items():
            # Category frame
            category_frame = ttk.LabelFrame(parent, text=category)
            category_frame.pack(fill=tk.X, padx=5, pady=5)

            for param_name, config in params.items():
                self.create_parameter_widget(
                    category_frame, param_name, config)

    def create_parameter_widget(self, parent, param_name: str, config: Dict[str, Any]):
        """Create a single parameter widget"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)

        # Label
        label = ttk.Label(frame, text=param_name, width=20)
        label.pack(side=tk.LEFT)

        # Widget based on type
        widget_frame = ttk.Frame(frame)
        widget_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        if config["type"] == "bool":
            var = tk.BooleanVar(value=config["default"])
            widget = ttk.Checkbutton(widget_frame, variable=var,
                                     command=lambda: self.parameter_changed(param_name, var.get()))
            widget.pack()

        elif config["type"] == "choice":
            var = tk.StringVar(value=config["default"])
            widget = ttk.Combobox(widget_frame, textvariable=var, values=config["choices"],
                                  state="readonly", width=15)
            widget.bind("<<ComboboxSelected>>",
                        lambda e: self.parameter_changed(param_name, var.get()))
            widget.pack()

        elif config["type"] in ["int", "double"]:
            var = tk.DoubleVar(value=config["default"])

            if "min" in config and "max" in config:
                # Scale widget for ranged values
                widget = ttk.Scale(widget_frame, from_=config["min"], to=config["max"],
                                   variable=var, orient=tk.HORIZONTAL,
                                   command=lambda v: self.parameter_changed(param_name,
                                                                            int(float(v)) if config["type"] == "int" else float(v)))
                widget.pack(fill=tk.X)

                # Value display
                value_label = ttk.Label(
                    widget_frame, text=str(config["default"]), width=8)
                value_label.pack()
                self.parameter_widgets[f"{param_name}_label"] = value_label
            else:
                # Entry widget for unrestricted values
                entry = ttk.Entry(widget_frame, textvariable=var, width=15)
                entry.bind("<Return>",
                           lambda e: self.parameter_changed(param_name,
                                                            int(var.get()) if config["type"] == "int" else var.get()))
                entry.pack()

        elif config["type"] == "string":
            var = tk.StringVar(value=config["default"])
            entry = ttk.Entry(widget_frame, textvariable=var, width=20)
            entry.bind("<Return>", lambda e: self.parameter_changed(
                param_name, var.get()))
            entry.pack()

        # Store widget reference
        self.parameter_widgets[param_name] = var
        # Initialize with default (will be overwritten by node parameters if available)
        self.parameters[param_name] = config["default"]

    def setup_ros(self):
        """Setup ROS2 node and connections"""
        def ros_thread():
            try:
                rclpy.init()
                self.ros_node = LineDetectorGUINode(self.image_callback)
                rclpy.spin(self.ros_node)
            except Exception as e:
                print(f"ROS error: {e}")
            finally:
                if self.ros_node:
                    self.ros_node.destroy_node()
                try:
                    rclpy.shutdown()
                except:
                    pass  # Ignore shutdown errors

        self.ros_thread = threading.Thread(target=ros_thread, daemon=True)
        self.ros_thread.start()

        # Wait a moment for ROS to initialize
        time.sleep(1.0)
        self.update_connection_status()

    def image_callback(self, cv_image, lines_count=0):
        """Callback for new images"""
        try:
            # Put new image in queue (non-blocking)
            if not self.image_queue.full():
                self.image_queue.put((cv_image, lines_count), block=False)
        except queue.Full:
            pass  # Skip frame if queue is full

    def update_image_display(self):
        """Update image display thread"""
        frame_count = 0
        last_update_time = 0
        min_update_interval = 1.0 / 30.0  # Limit to 30 FPS to reduce flicker

        while self.update_thread_running:
            try:
                # Get latest image
                cv_image, lines_count = self.image_queue.get(timeout=0.1)

                current_time = time.time()
                # Skip frame if updating too frequently
                if current_time - last_update_time < min_update_interval:
                    continue

                # Convert to tkinter format
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)

                # Resize to fit canvas
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:
                    pil_image = self.resize_image_to_fit(
                        pil_image, canvas_width, canvas_height)

                    # Create PhotoImage
                    new_image = ImageTk.PhotoImage(pil_image)

                    # Update display in main thread
                    self.root.after(0, self.update_canvas_image, new_image)

                    last_update_time = current_time

                # Update status (less frequently)
                frame_count += 1
                if frame_count % 5 == 0:  # Update status every 5 frames
                    self.root.after(0, lambda: self.update_status(
                        'frame', f"Frame: {frame_count}"))
                    self.root.after(0, lambda: self.update_status(
                        'lines', f"Lines: {lines_count}"))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Image update error: {e}")

    def resize_image_to_fit(self, pil_image, max_width, max_height):
        """Resize image to fit within canvas while maintaining aspect ratio"""
        img_width, img_height = pil_image.size

        # Calculate scaling factor
        scale_w = max_width / img_width
        scale_h = max_height / img_height
        scale = min(scale_w, scale_h)

        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        return pil_image.resize((new_width, new_height), Image.LANCZOS)

    def update_canvas_image(self, new_image):
        """Update canvas with current image"""
        try:
            # Remove placeholder text on first image
            if self.placeholder_text_id:
                self.image_canvas.delete(self.placeholder_text_id)
                self.placeholder_text_id = None

            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            x = canvas_width // 2
            y = canvas_height // 2

            # Update existing image or create new one
            if self.canvas_image_id:
                # Update existing image to avoid flicker
                self.image_canvas.itemconfig(
                    self.canvas_image_id, image=new_image)
            else:
                # Create new image item
                self.canvas_image_id = self.image_canvas.create_image(
                    x, y, image=new_image)

            # Keep reference to prevent garbage collection
            self.current_image = new_image

        except Exception as e:
            print(f"Canvas update error: {e}")

    def update_status(self, key, text):
        """Update status label"""
        if key in self.status_labels:
            self.status_labels[key].config(text=text)

    def update_connection_status(self):
        """Update connection status"""
        if self.ros_node:
            self.update_status('connection', "Connection: Connected")
        else:
            self.update_status('connection', "Connection: Disconnected")

    def fetch_initial_parameters(self):
        """Fetch initial parameters from the node on startup"""
        if not self.ros_node:
            print("Warning: ROS node not ready for parameter fetching")
            return

        # Build list of all parameter names from definitions
        param_names = []
        for category, params in self.parameter_definitions.items():
            for param_name in params.keys():
                # Skip individual ROI components, get arrays instead
                if not (param_name.endswith('_x') or param_name.endswith('_y') or
                        param_name.endswith('_w') or param_name.endswith('_h')):
                    param_names.append(param_name)

        # Add ROI array parameters
        param_names.extend(['roi', 'calib_roi'])
        
        print(f"Attempting to fetch {len(param_names)} parameters from node...")
        print(f"Parameter names: {param_names[:10]}...")  # Show first 10 for debugging

        # Get parameters from node
        fetched_params = self.ros_node.get_parameters_from_node(param_names)

        if fetched_params:
            print(f"Successfully fetched {len(fetched_params)} parameters from node")
            # Note: 'fetch' status label doesn't exist, using 'connection' instead
            self.update_status(
                'connection', f"Connected - Loaded {len(fetched_params)} parameters")

            # Update GUI widgets and internal parameters
            updated_count = 0
            for param_name, value in fetched_params.items():
                if param_name in self.parameter_widgets:
                    try:
                        self.parameter_widgets[param_name].set(value)
                        self.parameters[param_name] = value
                        updated_count += 1

                        # Update value label for scale widgets
                        if f"{param_name}_label" in self.parameter_widgets:
                            self.parameter_widgets[f"{param_name}_label"].config(
                                text=str(value))
                    except tk.TclError as e:
                        print(
                            f"Warning: Failed to set parameter {param_name}={value}: {e}")
            print(f"Updated {updated_count} GUI widgets with node parameters")
        else:
            print("Warning: No parameters fetched from node, using defaults")
            self.update_status(
                'connection', "Connected - Using default parameters")

    def parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes"""
        self.parameters[param_name] = value

        # Update value label for scale widgets
        if f"{param_name}_label" in self.parameter_widgets:
            self.parameter_widgets[f"{param_name}_label"].config(
                text=str(value))

        # Send to ROS node
        if self.ros_node:
            self.ros_node.set_parameter(param_name, value)

    def reset_parameters(self):
        """Reset all parameters to default values"""
        for category, params in self.parameter_definitions.items():
            for param_name, config in params.items():
                if param_name in self.parameter_widgets:
                    self.parameter_widgets[param_name].set(config["default"])
                    self.parameter_changed(param_name, config["default"])

    def save_parameters(self):
        """Save current parameters to file"""
        messagebox.showinfo(
            "Save", "Parameter save functionality not implemented yet")

    def load_parameters(self):
        """Load parameters from file"""
        messagebox.showinfo(
            "Load", "Parameter load functionality not implemented yet")

    def on_closing(self):
        """Handle window close"""
        self.update_thread_running = False
        if self.ros_node:
            self.ros_node.destroy_node()
        self.root.destroy()

    def run(self):
        """Start the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


class LineDetectorGUINode(Node):
    """ROS2 node for GUI communication"""

    def __init__(self, image_callback):
        super().__init__('line_detector_gui')

        self.image_callback = image_callback
        self.bridge = CvBridge()

        # Image subscriber
        self.image_sub = self.create_subscription(
            ImageMsg,
            '/image_with_lines',
            self.image_topic_callback,
            10
        )

        # Parameter clients
        self.param_client = self.create_client(
            SetParameters, 'etrobo_line_detector/set_parameters')
        self.param_get_client = self.create_client(
            GetParameters, 'etrobo_line_detector/get_parameters')

        self.get_logger().info("Line Detector GUI Node started")

    def image_topic_callback(self, msg):
        """Handle incoming images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # TODO: Extract line count from image or separate topic
            self.image_callback(cv_image, 0)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def set_parameter(self, param_name: str, value: Any):
        """Set parameter on target node"""
        if not self.param_client.service_is_ready():
            self.get_logger().warn("Parameter service not ready")
            return

        # Convert ROI parameters
        if param_name.startswith('roi_') or param_name.startswith('calib_roi_'):
            # Handle ROI as array parameter
            self._update_roi_parameter(param_name, value)
            return

        # Create parameter message
        param = Parameter()
        param.name = param_name

        if isinstance(value, bool):
            param.value = ParameterValue(
                type=ParameterType.PARAMETER_BOOL, bool_value=value)
        elif isinstance(value, int):
            param.value = ParameterValue(
                type=ParameterType.PARAMETER_INTEGER, integer_value=value)
        elif isinstance(value, float):
            param.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=value)
        elif isinstance(value, str):
            param.value = ParameterValue(
                type=ParameterType.PARAMETER_STRING, string_value=value)
        else:
            self.get_logger().warn(
                f"Unsupported parameter type for {param_name}: {type(value)}")
            return

        # Send parameter
        request = SetParameters.Request()
        request.parameters = [param]

        future = self.param_client.call_async(request)
        # Note: In a production version, we'd handle the response

    def _update_roi_parameter(self, param_name: str, value: Any):
        """Update ROI array parameters by combining individual components"""
        if param_name.startswith('calib_roi_'):
            roi_param_name = 'calib_roi'
            components = ['calib_roi_x', 'calib_roi_y',
                          'calib_roi_w', 'calib_roi_h']
        else:  # roi_
            roi_param_name = 'roi'
            components = ['roi_x', 'roi_y', 'roi_w', 'roi_h']

        # Get all current values - use default values since we don't track them here
        roi_values = [-1, -1, -1, -1]  # default values

        # Create array parameter
        param = Parameter()
        param.name = roi_param_name
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER_ARRAY,
            integer_array_value=roi_values
        )

        # Send parameter
        request = SetParameters.Request()
        request.parameters = [param]
        future = self.param_client.call_async(request)

    def get_parameters_from_node(self, param_names: list) -> dict:
        """Get parameters from the target node"""
        # Wait a bit for service to be ready
        timeout_start = time.time()
        while not self.param_get_client.service_is_ready():
            if time.time() - timeout_start > 3.0:
                self.get_logger().warn("Parameter get service not ready after 3 seconds")
                return {}
            time.sleep(0.1)
        
        self.get_logger().info(f"Parameter service is ready, fetching {len(param_names)} parameters")

        request = GetParameters.Request()
        request.names = param_names

        try:
            future = self.param_get_client.call_async(request)
            # Wait for response (blocking call)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.result() is not None:
                response = future.result()
                param_dict = {}

                for i, param_name in enumerate(param_names):
                    if i < len(response.values):
                        param_value = response.values[i]

                        # Convert parameter value based on type
                        if param_value.type == ParameterType.PARAMETER_BOOL:
                            param_dict[param_name] = param_value.bool_value
                        elif param_value.type == ParameterType.PARAMETER_INTEGER:
                            param_dict[param_name] = param_value.integer_value
                        elif param_value.type == ParameterType.PARAMETER_DOUBLE:
                            param_dict[param_name] = param_value.double_value
                        elif param_value.type == ParameterType.PARAMETER_STRING:
                            param_dict[param_name] = param_value.string_value
                        elif param_value.type == ParameterType.PARAMETER_INTEGER_ARRAY:
                            # Handle ROI arrays
                            if param_name == 'roi' and len(param_value.integer_array_value) >= 4:
                                param_dict['roi_x'] = param_value.integer_array_value[0]
                                param_dict['roi_y'] = param_value.integer_array_value[1]
                                param_dict['roi_w'] = param_value.integer_array_value[2]
                                param_dict['roi_h'] = param_value.integer_array_value[3]
                            elif param_name == 'calib_roi' and len(param_value.integer_array_value) >= 4:
                                param_dict['calib_roi_x'] = param_value.integer_array_value[0]
                                param_dict['calib_roi_y'] = param_value.integer_array_value[1]
                                param_dict['calib_roi_w'] = param_value.integer_array_value[2]
                                param_dict['calib_roi_h'] = param_value.integer_array_value[3]
                            else:
                                param_dict[param_name] = list(
                                    param_value.integer_array_value)

                return param_dict
            else:
                self.get_logger().warn("Failed to get parameters from node")
                return {}

        except Exception as e:
            self.get_logger().error(f"Error getting parameters: {e}")
            return {}


def main():
    """Main entry point"""
    try:
        gui = LineDetectorParameterGUI()
        gui.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
