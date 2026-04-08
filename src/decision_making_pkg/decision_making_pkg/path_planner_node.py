import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#---------------Variable Setting---------------
SUB_LANE_TOPIC_NAME = "yolov8_lane_info"  # lane_info_extractor 노드에서 퍼블리시하는 타겟 지점 토픽
PUB_TOPIC_NAME = "path_planning_result"   # 경로 계획 결과 퍼블리시 토픽
CAR_CENTER_POINT1 = (320, 260) # 이미지 상에서 차량 앞 범퍼의 중심이 위치한 픽셀 좌표
CAR_CENTER_POINT2 = (320, 260)
#----------------------------------------------
class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # 파라미터 선언
        self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.car_center_point1 = self.declare_parameter('car_center_point1', CAR_CENTER_POINT1).value
        self.car_center_point2 = self.declare_parameter('car_center_point2', CAR_CENTER_POINT2).value
        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.target_points_1 = []  # 차선의 타겟 지점들 (차선 중앙)
        self.target_points_2 = []
        # 서브스크라이버 설정 (타겟 지점 구독)
        self.lane_sub1 = self.create_subscription(LaneInfo, 'yolov8_lane_info_1', self.lane_1_callback, self.qos_profile)
        self.lane_sub2 = self.create_subscription(LaneInfo, 'yolov8_lane_info_2', self.lane_2_callback, self.qos_profile)

        # 퍼블리셔 설정 (경로 계획 결과 퍼블리시)
        self.publisher_1 = self.create_publisher(PathPlanningResult, 'path_planning_result_1', self.qos_profile)
        self.publisher_2 = self.create_publisher(PathPlanningResult, 'path_planning_result_2', self.qos_profile)

    def lane_1_callback(self, msg: LaneInfo):
        
        # 타겟 지점 받아오기
        self.target_points_1 = msg.target_points
        
        # 타겟 지점이 3개 이상 모이면 경로 계획 시작
        if len(self.target_points_1) >= 3:
            self.plan_path_1()

    def lane_2_callback(self, msg: LaneInfo):
        
        # 타겟 지점 받아오기
        self.target_points_2 = msg.target_points
        
        # 타겟 지점이 3개 이상 모이면 경로 계획 시작
        if len(self.target_points_2) >= 3:
            self.plan_path_2()

    def plan_path_1(self):
        # self.target_points가 TargetPoint 객체들의 리스트라고 가정
        if not self.target_points_1:
            self.get_logger().warn("No target points available")
            return
        
        # TargetPoint 객체에서 x, y 값 추출
        x_points, y_points = zip(*[(tp.target_x, tp.target_y) for tp in self.target_points_1])

        #차량 앞 범퍼의 중심이 위치한 픽셀 좌표 추가
        y_points_list, x_points_list = list(y_points), list(x_points) 
        y_points_list.append(self.car_center_point1[1])
        x_points_list.append(self.car_center_point1[0])
        y_points, x_points = tuple(y_points_list), tuple(x_points_list)
        
        # y 값을 기준으로 정렬 (y가 증가하는 순서로 정렬)
        sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])

        # 정렬된 y, x 값을 다시 분리
        y_points, x_points = zip(*sorted_points)
        
        # 몇개의 점으로 경로 계획을 하는지 확인
        self.get_logger().info(f"Planning path with {len(y_points)} points")

        # 스플라인 보간법을 사용하여 경로 생성
        cs = CubicSpline(y_points, x_points, bc_type='natural')

        # 생성된 경로 점들 (추가적인 점들을 생성하여 부드러운 경로를 얻음)
        y_new = np.linspace(min(y_points), max(y_points), 100)
        x_new = cs(y_new)

        # 경로를 따라가는 정보 (PathPlanningResult 메시지로 발행)
        path_msg = PathPlanningResult()
        path_msg.x_points = list(x_new)
        path_msg.y_points = list(y_new)

        # 경로 퍼블리시
        self.publisher_1.publish(path_msg)

        # 타겟 지점 초기화 (다음 경로 계산을 위해)
        self.target_points_1.clear()

    def plan_path_2(self):
        # self.target_points가 TargetPoint 객체들의 리스트라고 가정
        if not self.target_points_2:
            self.get_logger().warn("No target points available")
            return
        
        # TargetPoint 객체에서 x, y 값 추출
        x_points, y_points = zip(*[(tp.target_x, tp.target_y) for tp in self.target_points_2])

        #차량 앞 범퍼의 중심이 위치한 픽셀 좌표 추가
        y_points_list, x_points_list = list(y_points), list(x_points) 
        y_points_list.append(self.car_center_point2[1])
        x_points_list.append(self.car_center_point2[0])
        y_points, x_points = tuple(y_points_list), tuple(x_points_list)
        
        # y 값을 기준으로 정렬 (y가 증가하는 순서로 정렬)
        sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])

        # 정렬된 y, x 값을 다시 분리
        y_points, x_points = zip(*sorted_points)
        
        # 몇개의 점으로 경로 계획을 하는지 확인
        self.get_logger().info(f"Planning path with {len(y_points)} points")

        # 스플라인 보간법을 사용하여 경로 생성
        cs = CubicSpline(y_points, x_points, bc_type='natural')

        # 생성된 경로 점들 (추가적인 점들을 생성하여 부드러운 경로를 얻음)
        y_new = np.linspace(min(y_points), max(y_points), 100)
        x_new = cs(y_new)

        # 경로를 따라가는 정보 (PathPlanningResult 메시지로 발행)
        path_msg = PathPlanningResult()
        path_msg.x_points = list(x_new)
        path_msg.y_points = list(y_new)

        # 경로 퍼블리시
        self.publisher_2.publish(path_msg)

        # 타겟 지점 초기화 (다음 경로 계산을 위해)
        self.target_points_2.clear()

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
