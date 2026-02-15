import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { 
  Move, 
  Activity, 
  Eye, 
  Brain, 
  Gamepad2, 
  Cpu, 
  Shield, 
  Radio,
  ChevronRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SDKCategory {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  apis: {
    name: string;
    description: string;
    code: string;
    language: string;
  }[];
}

const sdkCategories: SDKCategory[] = [
  {
    id: 'motion-control',
    title: 'Motion Control SDK',
    description: 'APIs to control joints, movement, balance, gait',
    icon: Move,
    apis: [
      {
        name: 'Joint Control',
        description: 'Control individual robot joints with precision',
        language: 'python',
        code: `import serial
import time
import json

class JointController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection
    
    def set_joint_angle(self, joint_id, angle_degrees):
        """Set joint angle in degrees"""
        command = {
            "cmd": "set_angle",
            "joint": joint_id,
            "angle": angle_degrees
        }
        self.serial.write(json.dumps(command).encode())
        return self.read_response()
    
    def set_joint_velocity(self, joint_id, velocity):
        """Set joint velocity in deg/s"""
        command = {
            "cmd": "set_velocity",
            "joint": joint_id,
            "velocity": velocity
        }
        self.serial.write(json.dumps(command).encode())
        return self.read_response()
    
    def get_joint_status(self, joint_id):
        """Get current joint position and torque"""
        command = {"cmd": "get_status", "joint": joint_id}
        self.serial.write(json.dumps(command).encode())
        response = self.read_response()
        return response
    
    def read_response(self):
        """Read JSON response from robot"""
        response = self.serial.readline().decode('utf-8').strip()
        return json.loads(response) if response else None

# Usage
controller = JointController()
controller.set_joint_angle("left_arm", 45.0)
status = controller.get_joint_status("left_arm")
print(f"Position: {status['angle']}, Torque: {status['torque']}")`,
      },
      {
        name: 'Walk Primitive',
        description: 'Prebuilt walking motion with speed control',
        language: 'python',
        code: `import numpy as np
import time

class WalkController:
    def __init__(self, joint_controller):
        self.controller = joint_controller
        self.is_walking = False
    
    def walk(self, speed=1.0, direction=0, duration=5.0):
        """
        Execute walking motion
        speed: meters per second (0.5 - 2.0)
        direction: degrees (0=forward, 90=right, -90=left, 180=back)
        duration: seconds to walk
        """
        self.is_walking = True
        start_time = time.time()
        step_frequency = speed / 0.5  # Hz
        
        while time.time() - start_time < duration and self.is_walking:
            t = (time.time() - start_time) * step_frequency
            
            # Sinusoidal gait pattern
            left_hip = 20 * np.sin(2 * np.pi * t)
            right_hip = 20 * np.sin(2 * np.pi * t + np.pi)
            left_knee = max(0, 40 * np.sin(2 * np.pi * t))
            right_knee = max(0, 40 * np.sin(2 * np.pi * t + np.pi))
            
            # Apply direction offset
            direction_rad = np.radians(direction)
            hip_offset = 10 * np.sin(direction_rad)
            
            self.controller.set_joint_angle("left_hip", left_hip + hip_offset)
            self.controller.set_joint_angle("right_hip", right_hip - hip_offset)
            self.controller.set_joint_angle("left_knee", left_knee)
            self.controller.set_joint_angle("right_knee", right_knee)
            
            time.sleep(0.02)  # 50Hz update rate
        
        self.stop_motion(smooth=True)
    
    def stop_motion(self, smooth=True):
        """Stop walking motion"""
        self.is_walking = False
        if smooth:
            # Gradually return to neutral position
            for i in range(20):
                factor = (20 - i) / 20
                self.controller.set_joint_angle("left_hip", 0)
                self.controller.set_joint_angle("right_hip", 0)
                time.sleep(0.05)

# Usage
from joint_controller import JointController
controller = JointController()
walker = WalkController(controller)
walker.walk(speed=1.2, direction=0, duration=5.0)`,
      },
      {
        name: 'Balance Control',
        description: 'Maintain balance and center of mass',
        language: 'python',
        code: `import numpy as np
from scipy.spatial.transform import Rotation

class BalanceController:
    def __init__(self, imu_sensor):
        self.imu = imu_sensor
        self.com_offset = np.array([0.0, 0.0, 0.0])
        self.auto_balance = False
        self.kp = 0.5  # Proportional gain
        self.kd = 0.1  # Derivative gain
    
    def enable_auto_balance(self):
        """Enable automatic balance correction"""
        self.auto_balance = True
    
    def adjust_com(self, x=0, y=0, z=0):
        """Adjust center of mass offset in meters"""
        self.com_offset = np.array([x, y, z])
    
    def get_metrics(self):
        """Calculate balance stability metrics"""
        accel = self.imu.get_acceleration()
        gyro = self.imu.get_gyroscope()
        
        # Calculate tilt angle
        tilt = np.arctan2(accel[0], accel[2])
        tilt_deg = np.degrees(tilt)
        
        # Stability score (0-100)
        stability = max(0, 100 - abs(tilt_deg) * 5)
        
        return {
            "stability_score": stability,
            "tilt_angle": tilt_deg,
            "angular_velocity": np.linalg.norm(gyro),
            "com_offset": self.com_offset.tolist()
        }
    
    def compute_correction(self):
        """Compute ankle/hip angles for balance"""
        metrics = self.get_metrics()
        tilt = np.radians(metrics["tilt_angle"])
        
        # PD controller
        ankle_correction = -self.kp * tilt - self.kd * metrics["angular_velocity"]
        
        return {
            "ankle_pitch": ankle_correction,
            "hip_adjustment": self.com_offset[0] * 10
        }

# Usage
from imu_sensor import IMUSensor
imu = IMUSensor()
balance = BalanceController(imu)
balance.enable_auto_balance()
balance.adjust_com(x=0.02, y=0.0, z=0.05)
print(balance.get_metrics())`,
      },
      {
        name: 'Gait Control',
        description: 'Configure and control walking patterns',
        language: 'python',
        code: `from opanarchy_sdk import GaitController

gait = GaitController()

# Set gait parameters
gait.configure(
    step_height=0.05,
    step_length=0.3,
    frequency=1.5
)

# Apply custom gait pattern
gait.apply_pattern("humanoid_walk")`,
      },
    ],
  },
  {
    id: 'sensor',
    title: 'Sensor SDK',
    description: 'Unified drivers for depth cameras, LiDAR, touch sensors',
    icon: Activity,
    apis: [
      {
        name: 'Depth Camera',
        description: 'Access depth camera data streams',
        language: 'python',
        code: `import pyrealsense2 as rs
import numpy as np
import cv2

class DepthCamera:
    def __init__(self, device_id=0, width=640, height=480):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        self.align = rs.align(rs.stream.color)
        self.is_streaming = False
    
    def start_stream(self, fps=30):
        """Start depth camera streaming"""
        self.pipeline.start(self.config)
        self.is_streaming = True
        print(f"Streaming started at {fps} FPS")
    
    def get_depth_frame(self):
        """Get current depth frame"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return DepthFrame(depth_image, color_image, depth_frame)
    
    def stop_stream(self):
        """Stop streaming"""
        self.pipeline.stop()
        self.is_streaming = False

class DepthFrame:
    def __init__(self, depth_data, color_data, rs_frame):
        self.depth_data = depth_data
        self.color_data = color_data
        self.rs_frame = rs_frame
    
    def get_depth(self, x, y):
        """Get depth at pixel coordinates in meters"""
        return self.rs_frame.get_distance(x, y)
    
    def get_point_cloud(self):
        """Generate 3D point cloud"""
        pc = rs.pointcloud()
        points = pc.calculate(self.rs_frame)
        vertices = np.asanyarray(points.get_vertices())
        return vertices

# Usage
camera = DepthCamera()
camera.start_stream(fps=30)
frame = camera.get_depth_frame()
depth = frame.get_depth(320, 240)
print(f"Depth at center: {depth:.2f}m")
camera.stop_stream()`,
      },
      {
        name: 'LiDAR Scanner',
        description: 'Read LiDAR point cloud data',
        language: 'python',
        code: `from opanarchy_sdk import LidarScanner

lidar = LidarScanner()

# Start scanning
lidar.start_scan(resolution=0.25)  # degrees

# Get point cloud
points = lidar.get_point_cloud()
print(f"Points detected: {len(points)}")

# Filter by distance
close_objects = lidar.filter_by_distance(max_dist=2.0)`,
      },
      {
        name: 'Touch Sensors',
        description: 'Read touch and pressure sensors',
        language: 'python',
        code: `from opanarchy_sdk import TouchSensor

# Initialize touch sensors
touch = TouchSensor()

# Read all sensors
readings = touch.read_all()
for sensor_id, value in readings.items():
    print(f"{sensor_id}: {value}")

# Set up contact detection
touch.on_contact("left_hand", callback=handle_touch)`,
      },
      {
        name: 'Sensor Fusion',
        description: 'Combine multiple sensor streams',
        language: 'python',
        code: `from opanarchy_sdk import SensorFusion

fusion = SensorFusion()

# Add sensor streams
fusion.add_stream("camera", camera)
fusion.add_stream("lidar", lidar)

# Get fused data
fused_data = fusion.get_combined_data()
obstacles = fusion.detect_obstacles()`,
      },
    ],
  },
  {
    id: 'vision-perception',
    title: 'Vision + Perception SDK',
    description: 'Object detection, SLAM, pose estimation',
    icon: Eye,
    apis: [
      {
        name: 'Object Detection',
        description: 'Detect and classify objects in real-time',
        language: 'python',
        code: `from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence
    
    def detect(self, image):
        """
        Detect objects in image
        Returns list of detected objects with bounding boxes
        """
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    },
                    "center": ((x1+x2)/2, (y1+y2)/2)
                })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        for obj in detections:
            bbox = obj["bbox"]
            cv2.rectangle(image, 
                         (bbox["x1"], bbox["y1"]), 
                         (bbox["x2"], bbox["y2"]), 
                         (0, 255, 0), 2)
            
            label = f"{obj['label']}: {obj['confidence']:.2f}"
            cv2.putText(image, label, 
                       (bbox["x1"], bbox["y1"]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
        
        return image

# Usage
detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.5)
image = cv2.imread("scene.jpg")
objects = detector.detect(image)

for obj in objects:
    print(f"{obj['label']}: {obj['confidence']:.2f}")
    print(f"Location: {obj['bbox']}")
    print(f"Center: {obj['center']}")`,
      },
      {
        name: 'SLAM Module',
        description: 'Simultaneous localization and mapping',
        language: 'python',
        code: `from opanarchy_sdk import SLAM

slam = SLAM()

# Initialize SLAM
slam.initialize(initial_pose=[0, 0, 0])

# Update with sensor data
slam.update(camera_frame, lidar_scan)

# Get current map and position
map_data = slam.get_map()
position = slam.get_position()`,
      },
      {
        name: 'Pose Estimation',
        description: 'Estimate human and robot poses',
        language: 'python',
        code: `from opanarchy_sdk import PoseEstimator

pose = PoseEstimator()

# Detect human poses
humans = pose.detect_humans(image)

for human in humans:
    keypoints = human.get_keypoints()
    print(f"Head: {keypoints['head']}")
    print(f"Hands: {keypoints['hands']}")`,
      },
      {
        name: 'Camera Calibration',
        description: 'Calibrate camera intrinsics and extrinsics',
        language: 'python',
        code: `from opanarchy_sdk import CameraCalibration

calibration = CameraCalibration()

# Calibrate camera
calibration.calibrate(calibration_images)

# Get camera matrix
matrix = calibration.get_camera_matrix()
distortion = calibration.get_distortion_coefficients()`,
      },
    ],
  },
  {
    id: 'robotics-agent',
    title: 'Robotics Agent SDK',
    description: 'Framework for autonomous decision-making',
    icon: Brain,
    apis: [
      {
        name: 'Agent Framework',
        description: 'Create autonomous decision-making agents',
        language: 'python',
        code: `import asyncio
from typing import Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class Event:
    type: str
    data: Any
    timestamp: float

class RobotAgent:
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.event_handlers: Dict[str, list[Callable]] = {}
        self.state = {"position": [0, 0], "status": "idle"}
    
    def on_event(self, event_type: str):
        """Decorator to register event handlers"""
        def decorator(func: Callable):
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(func)
            return func
        return decorator
    
    async def emit_event(self, event_type: str, data: Any):
        """Emit event and call all registered handlers"""
        if event_type in self.event_handlers:
            event = Event(event_type, data, asyncio.get_event_loop().time())
            for handler in self.event_handlers[event_type]:
                await asyncio.create_task(handler(event))
    
    async def start_autonomous(self):
        """Start autonomous operation loop"""
        self.is_running = True
        print(f"Agent {self.name} started in autonomous mode")
        
        while self.is_running:
            # Simulated sensor check
            obstacle_detected = await self.check_sensors()
            
            if obstacle_detected:
                await self.emit_event("obstacle_detected", 
                                     {"distance": 0.5, "angle": 45})
            
            await asyncio.sleep(0.1)
    
    async def check_sensors(self):
        """Check sensors for obstacles"""
        # Placeholder - integrate with real sensors
        return False
    
    def stop(self):
        """Stop robot movement"""
        self.state["status"] = "stopped"
        print(f"Agent {self.name} stopped")
    
    def navigate_around(self, obstacle):
        """Navigate around detected obstacle"""
        print(f"Navigating around obstacle at {obstacle}")
        self.state["status"] = "navigating"

# Usage
agent = RobotAgent(name="explorer_bot")

@agent.on_event("obstacle_detected")
async def handle_obstacle(event):
    print(f"Obstacle detected: {event.data}")
    agent.stop()
    agent.navigate_around(event.data)

asyncio.run(agent.start_autonomous())`,
      },
      {
        name: 'Agent Communication',
        description: 'Multi-agent communication via X402 protocol',
        language: 'python',
        code: `from opanarchy_sdk import AgentComm

comm = AgentComm(agent_id="bot_001")

# Send message to another agent
comm.send_message(
    to="bot_002",
    message={"type": "task", "data": "patrol_area_A"}
)

# Listen for messages
@comm.on_message
def handle_message(msg):
    print(f"From {msg.sender}: {msg.data}")`,
      },
      {
        name: 'On-Device LLM',
        description: 'Run LLM models on robot hardware',
        language: 'python',
        code: `from opanarchy_sdk import OnDeviceLLM

llm = OnDeviceLLM(model="llama-7b")

# Process natural language commands
response = llm.process_command(
    "Pick up the red cube and place it on the table"
)

# Execute parsed actions
for action in response.actions:
    robot.execute(action)`,
      },
      {
        name: 'Task Planner',
        description: 'Plan and execute complex tasks',
        language: 'python',
        code: `from opanarchy_sdk import TaskPlanner

planner = TaskPlanner()

# Define task
task = planner.create_task(
    goal="clean_room",
    constraints=["avoid_furniture", "battery_>20%"]
)

# Execute task
planner.execute(task, callback=on_progress)`,
      },
    ],
  },
  {
    id: 'simulation',
    title: 'Simulation SDK',
    description: 'Integrations for Gazebo, Isaac, Web simulators',
    icon: Gamepad2,
    apis: [
      {
        name: 'Gazebo Integration',
        description: 'Connect to Gazebo simulator',
        language: 'python',
        code: `from opanarchy_sdk import GazeboSimulator

sim = GazeboSimulator()

# Launch simulation
sim.launch_world("warehouse.world")

# Spawn robot
robot = sim.spawn_robot(
    model="humanoid_v1",
    position=[0, 0, 1]
)

# Control simulated robot
robot.move_forward(speed=1.0)`,
      },
      {
        name: 'Isaac Sim',
        description: 'NVIDIA Isaac Sim integration',
        language: 'python',
        code: `from opanarchy_sdk import IsaacSimulator

sim = IsaacSimulator()

# Create simulation scene
scene = sim.create_scene("robotics_lab")

# Add robot and objects
robot = scene.add_robot("opanarchy_humanoid")
table = scene.add_object("table", position=[1, 0, 0])

# Run simulation
sim.run(steps=1000)`,
      },
      {
        name: 'Web Simulator',
        description: 'Browser-based 3D simulation',
        language: 'javascript',
        code: `import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

class WebSimulator {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.camera.position.set(5, 5, 5);
    this.controls.update();
    
    this.robot = null;
    this.isRunning = false;
    this.loader = new GLTFLoader();
    
    this.setupLighting();
    this.setupGround();
  }
  
  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);
  }
  
  setupGround() {
    const geometry = new THREE.PlaneGeometry(20, 20);
    const material = new THREE.MeshStandardMaterial({ color: 0x808080 });
    const ground = new THREE.Mesh(geometry, material);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);
  }
  
  async loadRobot(modelPath) {
    return new Promise((resolve, reject) => {
      this.loader.load(modelPath, (gltf) => {
        this.robot = gltf.scene;
        this.robot.position.set(0, 0, 0);
        this.robot.traverse((child) => {
          if (child.isMesh) child.castShadow = true;
        });
        this.scene.add(this.robot);
        resolve(this.robot);
      }, undefined, reject);
    });
  }
  
  createScenario({ environment, obstacles, lighting }) {
    // Add obstacles
    obstacles.forEach((type, i) => {
      const geometry = new THREE.BoxGeometry(1, 1, 1);
      const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
      const obstacle = new THREE.Mesh(geometry, material);
      obstacle.position.set(i * 2 - 2, 0.5, 2);
      obstacle.castShadow = true;
      this.scene.add(obstacle);
    });
  }
  
  animate() {
    if (!this.isRunning) return;
    requestAnimationFrame(() => this.animate());
    
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  
  start() {
    this.isRunning = true;
    this.animate();
  }
  
  stop() {
    this.isRunning = false;
  }
}

// Usage
const sim = new WebSimulator('canvas-id');
await sim.loadRobot('/models/humanoid.glb');
sim.createScenario({
  environment: 'office',
  obstacles: ['desk', 'chair'],
  lighting: 'natural'
});
sim.start();`,
      },
      {
        name: 'Test Scenarios',
        description: 'Prebuilt testing scenarios',
        language: 'python',
        code: `from opanarchy_sdk import TestScenarios

scenarios = TestScenarios()

# Run navigation test
results = scenarios.run_test(
    name="obstacle_navigation",
    robot_model="humanoid_v1",
    iterations=10
)

print(f"Success rate: {results.success_rate}%")`,
      },
    ],
  },
  {
    id: 'hardware-interface',
    title: 'Hardware Interface SDK',
    description: 'Unified API for servos, motors, actuators',
    icon: Cpu,
    apis: [
      {
        name: 'Servo Control',
        description: 'Control servo motors with precision',
        language: 'python',
        code: `import serial
import struct
import time

class ServoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        self.serial = serial.Serial(port, baudrate, timeout=0.5)
        self.servos = {}
    
    def set_position(self, servo_id, angle, duration=500):
        """
        Set servo position
        servo_id: 1-254
        angle: 0-180 degrees
        duration: movement time in ms
        """
        # Convert angle to servo position (500-2500 microseconds)
        position = int(500 + (angle / 180.0) * 2000)
        
        # Dynamixel-style packet
        packet = bytearray([
            0xFF, 0xFF,           # Header
            servo_id,             # ID
            0x07,                 # Length
            0x03,                 # Write command
            0x1E,                 # Goal position address
            position & 0xFF,      # Position low byte
            (position >> 8) & 0xFF,  # Position high byte
            duration & 0xFF,      # Duration low byte
            (duration >> 8) & 0xFF   # Duration high byte
        ])
        
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        
        self.serial.write(packet)
        time.sleep(0.01)
    
    def configure(self, servo_id, max_speed=100, acceleration=50):
        """Configure servo parameters"""
        self.servos[servo_id] = {
            'max_speed': max_speed,
            'acceleration': acceleration
        }
        
        # Send configuration
        speed_value = int(max_speed * 10.23)
        self._write_register(servo_id, 0x20, speed_value)
    
    def get_position(self, servo_id):
        """Read current servo position"""
        packet = bytearray([
            0xFF, 0xFF,
            servo_id,
            0x04,
            0x02,  # Read command
            0x24,  # Present position address
            0x02   # Read 2 bytes
        ])
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        
        self.serial.write(packet)
        response = self.serial.read(8)
        
        if len(response) >= 7:
            position = response[5] | (response[6] << 8)
            angle = (position - 500) / 2000.0 * 180.0
            return angle
        return None
    
    def _write_register(self, servo_id, address, value):
        """Write value to servo register"""
        packet = bytearray([
            0xFF, 0xFF, servo_id, 0x05, 0x03,
            address, value & 0xFF, (value >> 8) & 0xFF
        ])
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        self.serial.write(packet)

# Usage
servo = ServoController()
servo.configure(servo_id=1, max_speed=100, acceleration=50)
servo.set_position(servo_id=1, angle=90, duration=500)
time.sleep(0.6)
position = servo.get_position(servo_id=1)
print(f"Current position: {position:.1f}°")`,
      },
      {
        name: 'Motor Control',
        description: 'Control DC and stepper motors',
        language: 'python',
        code: `from opanarchy_sdk import MotorController

motor = MotorController()

# Set motor speed
motor.set_speed(motor_id="left_wheel", speed=50)

# Set motor direction
motor.set_direction(motor_id="left_wheel", direction="forward")

# Get motor telemetry
telemetry = motor.get_telemetry(motor_id="left_wheel")`,
      },
      {
        name: 'Actuator Interface',
        description: 'Generic actuator control',
        language: 'python',
        code: `from opanarchy_sdk import ActuatorInterface

actuator = ActuatorInterface()

# Register custom actuator
actuator.register(
    id="gripper",
    type="pneumatic",
    driver="custom_driver"
)

# Control actuator
actuator.activate("gripper", force=50)`,
      },
      {
        name: 'Third-Party Hardware',
        description: 'Plug-and-play third-party integrations',
        language: 'python',
        code: `from opanarchy_sdk import HardwareAdapter

adapter = HardwareAdapter()

# Auto-detect connected hardware
devices = adapter.scan_devices()

for device in devices:
    print(f"Found: {device.name} ({device.type})")
    adapter.connect(device)`,
      },
    ],
  },
  {
    id: 'safety-diagnostics',
    title: 'Safety & Diagnostics SDK',
    description: 'Error detection, motor health, power tracking',
    icon: Shield,
    apis: [
      {
        name: 'Error Detection',
        description: 'Real-time error monitoring and alerts',
        language: 'python',
        code: `import threading
import time
from datetime import datetime
from collections import deque
from typing import Callable, Dict

class ErrorDetector:
    def __init__(self, log_size=100):
        self.is_monitoring = False
        self.error_handlers: Dict[str, list[Callable]] = {}
        self.error_log = deque(maxlen=log_size)
        self.monitor_thread = None
        self.thresholds = {
            "motor_overheat": 70,  # °C
            "low_battery": 15,     # %
            "high_current": 10,    # A
            "connection_loss": 3   # seconds
        }
    
    def enable_monitoring(self):
        """Start error monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Error monitoring enabled")
    
    def on_error(self, error_type: str):
        """Decorator to register error handlers"""
        def decorator(func: Callable):
            if error_type not in self.error_handlers:
                self.error_handlers[error_type] = []
            self.error_handlers[error_type].append(func)
            return func
        return decorator
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            # Check various error conditions
            self._check_motor_temperature()
            self._check_battery()
            self._check_current()
            time.sleep(0.5)
    
    def _check_motor_temperature(self):
        """Check motor temperatures"""
        # Simulated temperature reading
        temps = {"motor_1": 65, "motor_2": 75}
        
        for motor_id, temp in temps.items():
            if temp > self.thresholds["motor_overheat"]:
                self._trigger_error("motor_overheat", {
                    "motor_id": motor_id,
                    "temperature": temp,
                    "threshold": self.thresholds["motor_overheat"]
                })
    
    def _check_battery(self):
        """Check battery level"""
        # Simulated battery reading
        battery_level = 85
        if battery_level < self.thresholds["low_battery"]:
            self._trigger_error("low_battery", {
                "level": battery_level,
                "threshold": self.thresholds["low_battery"]
            })
    
    def _check_current(self):
        """Check motor current draw"""
        pass
    
    def _trigger_error(self, error_type: str, data: dict):
        """Trigger error and call handlers"""
        error = {
            "type": error_type,
            "data": data,
            "timestamp": datetime.now(),
            "severity": "critical" if "overheat" in error_type else "warning"
        }
        
        self.error_log.append(error)
        
        if error_type in self.error_handlers:
            for handler in self.error_handlers[error_type]:
                handler(type('Error', (), error))
    
    def get_error_log(self, last_n=10):
        """Get recent errors"""
        return list(self.error_log)[-last_n:]
    
    def stop_monitoring(self):
        """Stop error monitoring"""
        self.is_monitoring = False

# Usage
detector = ErrorDetector()
detector.enable_monitoring()

@detector.on_error("motor_overheat")
def handle_overheat(error):
    print(f"⚠️ Motor {error.data['motor_id']} overheating!")
    print(f"Temperature: {error.data['temperature']}°C")
    # robot.stop_motor(error.data['motor_id'])

time.sleep(5)
errors = detector.get_error_log(last_n=10)
for err in errors:
    print(f"{err['timestamp']}: {err['type']} - {err['data']}")`,
      },
      {
        name: 'Motor Health',
        description: 'Monitor motor health and performance',
        language: 'python',
        code: `from opanarchy_sdk import MotorHealth

health = MotorHealth()

# Check motor health
status = health.check_motor("left_arm")

print(f"Health score: {status.health_score}")
print(f"Temperature: {status.temperature}°C")
print(f"Vibration: {status.vibration_level}")

# Get maintenance alerts
alerts = health.get_maintenance_alerts()`,
      },
      {
        name: 'Power Tracking',
        description: 'Monitor battery and power consumption',
        language: 'python',
        code: `from opanarchy_sdk import PowerMonitor

power = PowerMonitor()

# Get battery status
battery = power.get_battery_status()
print(f"Charge: {battery.percentage}%")
print(f"Voltage: {battery.voltage}V")
print(f"Time remaining: {battery.time_remaining}min")

# Monitor power consumption
consumption = power.get_consumption_by_component()`,
      },
      {
        name: 'Thermal Monitoring',
        description: 'Track component temperatures',
        language: 'python',
        code: `from opanarchy_sdk import ThermalMonitor

thermal = ThermalMonitor()

# Get temperature readings
temps = thermal.get_all_temperatures()

for component, temp in temps.items():
    if temp > 70:
        print(f"Warning: {component} at {temp}°C")

# Set temperature alerts
thermal.set_alert_threshold("cpu", max_temp=75)`,
      },
    ],
  },
  {
    id: 'communication',
    title: 'Communication SDK',
    description: 'Robot-to-Robot, Robot-to-Cloud messaging',
    icon: Radio,
    apis: [
      {
        name: 'Topic Messaging',
        description: 'ROS-like publish/subscribe topics',
        language: 'python',
        code: `import threading
import time
from typing import Callable, Dict, Any
from collections import defaultdict
import json

class TopicMessaging:
    def __init__(self):
        self.subscribers: Dict[str, list[Callable]] = defaultdict(list)
        self.message_queue = []
        self.lock = threading.Lock()
        self.is_running = True
        
        # Start message processing thread
        self.processor = threading.Thread(target=self._process_messages, daemon=True)
        self.processor.start()
    
    def subscribe(self, topic: str):
        """Decorator to subscribe to a topic"""
        def decorator(callback: Callable):
            with self.lock:
                self.subscribers[topic].append(callback)
            print(f"Subscribed to topic: {topic}")
            return callback
        return decorator
    
    def publish(self, topic: str, data: Any):
        """Publish message to topic"""
        message = {
            "topic": topic,
            "data": data,
            "timestamp": time.time()
        }
        
        with self.lock:
            self.message_queue.append(message)
    
    def _process_messages(self):
        """Process message queue"""
        while self.is_running:
            messages_to_process = []
            
            with self.lock:
                if self.message_queue:
                    messages_to_process = self.message_queue.copy()
                    self.message_queue.clear()
            
            for message in messages_to_process:
                self._deliver_message(message)
            
            time.sleep(0.01)
    
    def _deliver_message(self, message: dict):
        """Deliver message to all subscribers"""
        topic = message["topic"]
        
        # Direct topic match
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(type('Message', (), message))
                except Exception as e:
                    print(f"Error in callback: {e}")
        
        # Wildcard matching
        parts = topic.split('/')
        for sub_topic in self.subscribers:
            if self._topic_matches(topic, sub_topic):
                for callback in self.subscribers[sub_topic]:
                    try:
                        callback(type('Message', (), message))
                    except Exception as e:
                        print(f"Error in callback: {e}")
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports wildcards)"""
        if pattern.endswith('*'):
            return topic.startswith(pattern[:-1])
        return topic == pattern
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from topic"""
        with self.lock:
            if topic in self.subscribers and callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
    
    def shutdown(self):
        """Shutdown messaging system"""
        self.is_running = False

# Usage
messaging = TopicMessaging()

@messaging.subscribe("/robot/sensors/camera")
def on_camera_data(msg):
    print(f"Camera frame at {msg.timestamp}")
    print(f"Data: {msg.data}")

@messaging.subscribe("/robot/status")
def on_status(msg):
    print(f"Robot status: {msg.data}")

# Publish messages
messaging.publish("/robot/status", {"battery": 85, "status": "active"})
messaging.publish("/robot/sensors/camera", {"frame_id": 12345, "resolution": "1920x1080"})

time.sleep(1)`,
      },
      {
        name: 'Cloud Messaging',
        description: 'Robot to cloud communication',
        language: 'python',
        code: `from opanarchy_sdk import CloudMessaging

cloud = CloudMessaging(api_key="your_key")

# Send telemetry to cloud
cloud.send_telemetry({
    "robot_id": "bot_001",
    "position": [1.2, 3.4, 0.0],
    "status": "operational"
})

# Receive cloud commands
@cloud.on_command
def handle_command(cmd):
    robot.execute_command(cmd)`,
      },
      {
        name: 'Agent Messaging',
        description: 'Multi-agent communication layer',
        language: 'python',
        code: `from opanarchy_sdk import AgentMessaging

messaging = AgentMessaging(agent_id="bot_001")

# Broadcast to all agents
messaging.broadcast({
    "type": "alert",
    "message": "New task available"
})

# Direct message
messaging.send_to_agent(
    agent_id="bot_002",
    data={"task": "assist_me"}
)`,
      },
      {
        name: 'Decentralized Protocol',
        description: 'P2P communication using X402 protocol',
        language: 'python',
        code: `from opanarchy_sdk import DecentralizedComm

comm = DecentralizedComm()

# Join network
comm.join_network(network_id="opanarchy_mesh")

# Discover peers
peers = comm.discover_peers()

# Send encrypted message
comm.send_encrypted(
    peer_id="bot_003",
    message="Secure coordination data"
)`,
      },
    ],
  },
];

interface SDKCategoriesPanelProps {
  onSelectAPI: (api: { name: string; code: string; language: string; category: string }) => void;
}

export const SDKCategoriesPanel = ({ onSelectAPI }: SDKCategoriesPanelProps) => {
  return (
    <Card className="border-border h-[calc(100vh-14rem)] flex flex-col bg-card">
      <div className="px-4 py-3 border-b border-border">
        <h2 className="font-medium text-sm">API Reference</h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          8 categories, 32 APIs
        </p>
      </div>

      <ScrollArea className="flex-1">
        <Accordion type="multiple" className="w-full px-2 py-2">
          {sdkCategories.map((category) => {
            const Icon = category.icon;
            return (
              <AccordionItem key={category.id} value={category.id} className="border-border/50">
                <AccordionTrigger className="hover:no-underline px-2 py-2">
                  <div className="flex items-center gap-2.5 text-left">
                    <Icon className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="font-medium text-sm">{category.title.replace(' SDK', '')}</p>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="px-2 pb-2">
                  <div className="space-y-0.5 mt-1">
                    {category.apis.map((api, idx) => (
                      <Button
                        key={idx}
                        variant="ghost"
                        className="w-full justify-start text-left h-auto py-2 px-2 hover:bg-muted/50"
                        onClick={() => onSelectAPI({ 
                          ...api, 
                          category: category.title 
                        })}
                      >
                        <div className="flex items-center gap-2 w-full">
                          <ChevronRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                          <span className="text-sm">{api.name}</span>
                          <Badge variant="outline" className="text-[10px] ml-auto px-1.5 py-0">
                            {api.language}
                          </Badge>
                        </div>
                      </Button>
                    ))}
                  </div>
                </AccordionContent>
              </AccordionItem>
            );
          })}
        </Accordion>
      </ScrollArea>
    </Card>
  );
};