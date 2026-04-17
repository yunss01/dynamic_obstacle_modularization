import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult
import numpy as np
from scipy.interpolate import CubicSpline

# ==========================================
# [1] Parameter Configuration Module
class PathPlannerConfig:
    def __init__(self):
        # --- [Topic Names] ---
        self.sub_lane_1_topic = "yolov8_lane_info_1" # 전방 카메라
        self.sub_lane_2_topic = "yolov8_lane_info_2" # 후방 카메라
        self.pub_path_1_topic = "path_planning_result_1"
        self.pub_path_2_topic = "path_planning_result_2"

        # --- [Vehicle Parameters] ---
        # x: 이미지 상에서 차량 앞/뒤 범퍼 중심 픽셀 x좌표
        # y: 자동 계산 (target y 최댓값 + car_center_y_margin)
        # 전진할 때 오른쪽으로 붙이고 싶으면 이걸 더 작게
        self.car_center_x_1 = 280   # 전방 카메라 기준 차량 중심 x
        # 후진할 때 오른쪽으로 붙이고 싶으면 이걸 더 크게
        self.car_center_x_2 = 390   # 후방 카메라 기준 차량 중심 x

        # target y 최댓값에 더할 여유값 (너무 크면 경로가 급격히 꺾임, 20~40 권장)
        self.car_center_y_margin = 30

        # --- [Lateral Bias] ---
        # 실선에 붙는 현상 보정용 오프셋 (target_x에 더해짐)
        # 안 쓰니까 수정하지 마
        self.lateral_bias_1 = 0    # 전방 카메라 보정값 (픽셀)
        self.lateral_bias_2 = -0   # 후방 카메라 보정값 (픽셀)

        # --- [Algorithm Parameters] ---
        self.min_target_points = 3      # 경로 계획을 시작할 최소 타겟 점 개수
        self.spline_resolution = 100    # 스플라인으로 부드럽게 생성할 보간 점의 개수
# ==========================================

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        self.cfg = PathPlannerConfig()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.lane_sub1 = self.create_subscription(
            LaneInfo, self.cfg.sub_lane_1_topic, self.lane_1_callback, self.qos_profile)
        self.lane_sub2 = self.create_subscription(
            LaneInfo, self.cfg.sub_lane_2_topic, self.lane_2_callback, self.qos_profile)

        self.publisher_1 = self.create_publisher(
            PathPlanningResult, self.cfg.pub_path_1_topic, self.qos_profile)
        self.publisher_2 = self.create_publisher(
            PathPlanningResult, self.cfg.pub_path_2_topic, self.qos_profile)

    def lane_1_callback(self, msg: LaneInfo):
        self.generate_and_publish_path(
            msg.target_points, self.cfg.car_center_x_1, self.cfg.lateral_bias_1, self.publisher_1, cam_id="1")

    def lane_2_callback(self, msg: LaneInfo):
        self.generate_and_publish_path(
            msg.target_points, self.cfg.car_center_x_2, self.cfg.lateral_bias_2, self.publisher_2, cam_id="2")

    def generate_and_publish_path(self, target_points, car_center_x, lateral_bias, publisher, cam_id="?"):

        if len(target_points) < self.cfg.min_target_points:
            self.get_logger().warn("타겟 지점이 부족해서 경로를 만들 수 없습니다.")
            return

        x_points = [tp.target_x for tp in target_points]
        y_points = [tp.target_y for tp in target_points]

        # ── car_center y 자동 계산 ───────────────────────────
        auto_car_center_y = max(y_points) + self.cfg.car_center_y_margin
        self.get_logger().info(
            f"[cam{cam_id}] target y: {min(y_points)}~{max(y_points)}, "
            f"car_center: ({car_center_x}, {auto_car_center_y}), lateral_bias: {lateral_bias}"
        )
        # ─────────────────────────────────────────────────────

        x_points.append(car_center_x)
        y_points.append(auto_car_center_y)

        # y 값을 기준으로 정렬
        sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])
        y_points, x_points = zip(*sorted_points)

        # 중복 y값 제거
        seen_y = set()
        deduped = []
        for y, x in zip(y_points, x_points):
            if y not in seen_y:
                seen_y.add(y)
                deduped.append((y, x))

        if len(deduped) < self.cfg.min_target_points:
            self.get_logger().warn("중복 제거 후 타겟 지점 부족. 경로 생성 건너뜀.")
            return

        y_points, x_points = zip(*deduped)

        # 스플라인 보간 (차선 중앙 기준으로 정상 생성)
        cs = CubicSpline(y_points, x_points, bc_type='natural')
        y_new = np.linspace(min(y_points), max(y_points), self.cfg.spline_resolution)
        x_new = cs(y_new)

        # 경로 전체를 lateral_bias만큼 평행이동
        # 기울기(곡률)는 그대로 유지되고 위치만 이동됨
        x_new = x_new + lateral_bias

        path_msg = PathPlanningResult()
        path_msg.x_points = list(x_new)
        path_msg.y_points = list(y_new)
        publisher.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()