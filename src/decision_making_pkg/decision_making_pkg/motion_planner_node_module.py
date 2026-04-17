import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import PathPlanningResult, MotionCommand
from .lib import decision_making_func_lib as DMFL

# ==========================================
# [1] Parameter Configuration Module
class MotionPlannerConfig:
    def __init__(self):
        # --- [Topic Names] ---
        self.sub_path_1_topic = "path_planning_result_1"
        self.sub_path_2_topic = "path_planning_result_2"
        self.pub_topic = "topic_control_signal"

        # --- [Time & Mode] ---
        self.timer_period = 0.1              # 모션 제어 주기 (초)
        self.forward_cycle_duration = 14.5   # 전진 시간 (초)
        self.backward_cycle_duration = 14.5  # 후진 시간 (초)

        # --- [Speed & Accel] ---
        self.forward_speed = 100        # 전진 목표 속도
        self.backward_speed = -100      # 후진 목표 속도
        self.max_accel_step = 10        # 속도가 부드럽게 변하는 단위
        # 이 값 이하로 속도가 줄어들면 '정지됐다'고 판단하고 방향 전환
        self.stop_threshold = 5

        # --- [Steering Limits] ---
        self.max_angle_forward = 45     # 전진 시 최대 조향각
        self.max_angle_backward = 50    # 후진 시 최대 조향각

        # --- [Calculation Indices] ---
        self.lookahead_index_near = -40
        self.lookahead_index_far = -1
# ==========================================

def convert_steeringangle2command(max_target_angle, target_angle):
    f = lambda x: 7 / (max_target_angle**3) * (x**3)
    ret_direction = round(f(target_angle))
    ret_direction = 7 if ret_direction >= 7 else ret_direction
    ret_direction = -7 if ret_direction <= -7 else ret_direction
    return ret_direction


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        self.cfg = MotionPlannerConfig()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.path_data_1 = None
        self.path_data_2 = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.current_left_speed = 0
        self.current_right_speed = 0

        self.mode = 1      # 0=전진(후방카메라), 1=후진(전방카메라)
        self.start_time = None
        self.data_ready = False

        # ── 부드러운 방향 전환을 위한 상태 ──────────────
        # False: 정상 주행 중
        # True: 감속 중 (목표 시간 경과 후 속도 0 될 때까지 대기)
        self.switching = False
        # ─────────────────────────────────────────────────

        self.path_sub_1 = self.create_subscription(
            PathPlanningResult, self.cfg.sub_path_1_topic, self.path1_callback, self.qos_profile)
        self.path_sub_2 = self.create_subscription(
            PathPlanningResult, self.cfg.sub_path_2_topic, self.path2_callback, self.qos_profile)

        self.publisher = self.create_publisher(
            MotionCommand, self.cfg.pub_topic, self.qos_profile)

        self.timer = self.create_timer(self.cfg.timer_period, self.timer_callback)
        self.last_path_time = self.get_clock().now()

    def path1_callback(self, msg: PathPlanningResult):
        self.last_path_time = self.get_clock().now()
        self.path_data_1 = list(zip(msg.x_points, msg.y_points))

    def path2_callback(self, msg: PathPlanningResult):
        self.last_path_time = self.get_clock().now()
        self.path_data_2 = list(zip(msg.x_points, msg.y_points))

    def update_speed_smoothly(self, target, current):
        if current < target:
            return min(current + self.cfg.max_accel_step, target)
        elif current > target:
            return max(current - self.cfg.max_accel_step, target)
        else:
            return current

    def timer_callback(self):
        if not self.data_ready:
            if self.path_data_1 is not None and self.path_data_2 is not None:
                self.start_time = self.get_clock().now()
                self.data_ready = True
                self.get_logger().info("경로 데이터 수신 완료. 주행 시작!")
            else:
                return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        current_duration = (
            self.cfg.forward_cycle_duration if self.mode == 0
            else self.cfg.backward_cycle_duration
        )

        # ── 방향 전환 타이밍 감지 → 감속 시작 ─────────────
        if elapsed >= current_duration and not self.switching:
            self.switching = True
            self.steering_command = 0  # 감속 중 조향 고정
            self.get_logger().info("목표 시간 경과. 감속 시작 후 방향 전환.")
        # ──────────────────────────────────────────────────

        # ── 감속 중: 속도가 충분히 줄면 방향 전환 ─────────
        if self.switching:
            target_left_speed = 0
            target_right_speed = 0
            self.current_left_speed = self.update_speed_smoothly(
                target_left_speed, self.current_left_speed)
            self.current_right_speed = self.update_speed_smoothly(
                target_right_speed, self.current_right_speed)

            # 거의 멈췄으면 방향 전환 실행
            if abs(self.current_left_speed) <= self.cfg.stop_threshold:
                self.current_left_speed = 0
                self.current_right_speed = 0
                self.mode = 1 - self.mode
                self.switching = False
                self.start_time = self.get_clock().now()
                self.get_logger().info(f"방향 전환 완료. 새 mode: {self.mode}")

            self.left_speed_command = self.current_left_speed
            self.right_speed_command = self.current_right_speed
            self._publish(self.steering_command,
                          self.left_speed_command,
                          self.right_speed_command)
            return
        # ──────────────────────────────────────────────────

        # ── 정상 주행 ──────────────────────────────────────
        data_age = (self.get_clock().now() - self.last_path_time).nanoseconds * 1e-9

        if data_age > 1.0:
            self.get_logger().warn(f"차선 데이터 유실! 안전 정지! (데이터 나이: {data_age:.2f}초)")
            self.steering_command = 0
            target_left_speed = 0
            target_right_speed = 0

        elif self.path_data_1 is None or self.path_data_2 is None:
            self.steering_command = 0
            target_left_speed = 0
            target_right_speed = 0

        else:
            if self.mode == 1:  # 후진 모드 (전방카메라 경로 사용)
                target_slope = DMFL.calculate_slope_between_points(
                    self.path_data_1[self.cfg.lookahead_index_near],
                    self.path_data_1[self.cfg.lookahead_index_far]
                )
                self.steering_command = convert_steeringangle2command(
                    self.cfg.max_angle_forward, target_slope)
                target_left_speed = self.cfg.forward_speed
                target_right_speed = self.cfg.forward_speed
            else:  # 전진 모드 (후방카메라 경로 사용)
                target_slope = DMFL.calculate_slope_between_points(
                    self.path_data_2[self.cfg.lookahead_index_near],
                    self.path_data_2[self.cfg.lookahead_index_far]
                )
                self.steering_command = -1 * convert_steeringangle2command(
                    self.cfg.max_angle_backward, target_slope)
                target_left_speed = self.cfg.backward_speed
                target_right_speed = self.cfg.backward_speed

        self.current_left_speed = self.update_speed_smoothly(
            target_left_speed, self.current_left_speed)
        self.current_right_speed = self.update_speed_smoothly(
            target_right_speed, self.current_right_speed)

        self.left_speed_command = self.current_left_speed
        self.right_speed_command = self.current_right_speed

        self.get_logger().info(
            f"mode: {self.mode}, steering: {self.steering_command}, "
            f"left_speed: {self.left_speed_command}, right_speed: {self.right_speed_command}"
        )
        self._publish(self.steering_command,
                      self.left_speed_command,
                      self.right_speed_command)

    def _publish(self, steering, left_speed, right_speed):
        msg = MotionCommand()
        msg.steering = steering
        msg.left_speed = left_speed
        msg.right_speed = right_speed
        self.publisher.publish(msg)


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