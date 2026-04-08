import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from interfaces_pkg.msg import LaneInfo
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"
TIMER = 0.1  # 모션 플랜 발행 주기 (초)
CYCLE_DURATION = 14.5  # 전진/후진 시간 (초)

def convert_steeringangle2command(max_target_angle, target_angle):
    f = lambda x: 7 / (max_target_angle**3) * (x**3)
    ret_direction = round(f(target_angle))
    ret_direction = 7 if ret_direction >= 7 else ret_direction
    ret_direction = -7 if ret_direction <= -7 else ret_direction
    return ret_direction

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # Parameter 설정
        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter('timer', TIMER).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # State 변수
        self.detection_data = None
        self.path_data_1 = None
        self.path_data_2 = None
        self.traffic_light_data = None
        self.lidar_data = None
        self.slope_1 = 0.0
        self.slope_2 = 0.0

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.max_accel_step = 10

        self.mode = 1  # 시작은 후진 (mode: 0=전진, 1=후진)
        self.start_time = None  # 주기 시작 시간
        self.data_ready = False


        # Subscription
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, 'path_planning_result_1', self.path1_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, 'path_planning_result_2', self.path2_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)
        self.lane_sub1 = self.create_subscription(LaneInfo, 'yolov8_lane_info_1', self.lane_1_callback, self.qos_profile)
        self.lane_sub2 = self.create_subscription(LaneInfo, 'yolov8_lane_info_2', self.lane_2_callback, self.qos_profile)

        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def lane_1_callback(self, msg: LaneInfo):
        self.slope_1 = msg.slope

    def lane_2_callback(self, msg: LaneInfo):
        self.slope_2 = msg.slope

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path1_callback(self, msg: PathPlanningResult):
        self.path_data_1 = list(zip(msg.x_points, msg.y_points))

    def path2_callback(self, msg: PathPlanningResult):
        self.path_data_2 = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def update_speed_smoothly(self, target, current):
        if current < target:
            return min(current + self.max_accel_step, target)
        elif current > target:
            return max(current - self.max_accel_step, target)
        else:
            return current

    def timer_callback(self):
        # start_time을 첫 callback 시점에 정확히 설정
        if not self.data_ready:
            if self.path_data_1 is not None and self.path_data_2 is not None:
                self.start_time = self.get_clock().now()
                self.data_ready = True
                self.get_logger().info("Start time initialized after data ready")
            else:
                # 아직 준비 안 됐으면 그냥 return
                return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        # 14초마다 mode 전환
        if elapsed >= CYCLE_DURATION:
            self.mode = 1 - self.mode  # 0 ↔ 1 toggle
            #self.mode = 0  # 0 ↔ 1 toggle
            self.start_time = self.get_clock().now()  # 정확히 현재 시각을 기준으로 다음 14초를 시작

        if self.lidar_data is not None and self.lidar_data.data is True:
            self.steering_command = 0
            target_left_speed = 0
            target_right_speed = 0

        elif self.traffic_light_data is not None and self.traffic_light_data.data == 'Red':
            for detection in self.detection_data.detections:
                if detection.class_name == 'traffic_light':
                    y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                    if y_max < 150:
                        self.steering_command = 0
                        target_left_speed = 0
                        target_right_speed = 0
                        break
        else:
            if self.path_data_1 is None or self.path_data_2 is None:
                self.steering_command = 0
                target_left_speed = 0
                target_right_speed = 0
            else:
                if self.mode == 1:
                    target_slope = DMFL.calculate_slope_between_points(self.path_data_1[-10], self.path_data_1[-1])
                    self.steering_command = convert_steeringangle2command(45, target_slope)
                    target_left_speed = 100
                    target_right_speed = 100
                else:
                    target_slope = DMFL.calculate_slope_between_points(self.path_data_2[-10], self.path_data_2[-1])
                    self.steering_command = -1 * convert_steeringangle2command(50, target_slope)
                    target_left_speed = -100
                    target_right_speed = -100

        self.current_left_speed = self.update_speed_smoothly(target_left_speed, self.current_left_speed)
        self.current_right_speed = self.update_speed_smoothly(target_right_speed, self.current_right_speed)

        self.left_speed_command = self.current_left_speed
        self.right_speed_command = self.current_right_speed

        self.get_logger().info(f"mode: {self.mode}, steering: {self.steering_command}, left_speed: {self.left_speed_command}, right_speed: {self.right_speed_command}")

        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
