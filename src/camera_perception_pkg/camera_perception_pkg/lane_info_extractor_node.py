import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL
import numpy as np

# scikit-learn RANSACRegressor 및 LinearRegression 임포트
from sklearn.linear_model import RANSACRegressor, LinearRegression

#---------------Variable Setting---------------
SUB_TOPIC_NAME = "detections"
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"
SHOW_IMAGE = True
# 후방 카메라 차선 중심 오른쪽 편향 보정값 (픽셀)

#----------------------------------------------

class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.cv_bridge = CvBridge()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber_1 = self.create_subscription(DetectionArray, 'detections_1', self.yolov8_detections_1_callback, self.qos_profile)
        self.subscriber_2 = self.create_subscription(DetectionArray, 'detections_2', self.yolov8_detections_2_callback, self.qos_profile)
        self.publisher_1 = self.create_publisher(LaneInfo, 'yolov8_lane_info_1', self.qos_profile)
        self.publisher_2 = self.create_publisher(LaneInfo, 'yolov8_lane_info_2', self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

        self.src_mat_orig_ref_1 = [[238, 316],[402, 313], [501, 476], [155, 476]]  # 전방카메라 좌표
        self.src_mat_orig_ref_2 = [[169, 196], [434, 196], [605, 432], [0, 432]]  # 후방카메라 좌표

        # ── 카메라 1 실측 기준 ──────────────────────────────────────────
        # BEV ROI 기준: line(실선)이 오른쪽(x0≈500), dotted_line(점선)이 왼쪽(x0≈170)
        # ──────────────────────────────────────────────────────────────
        self.last_lane_width_1 = 338  # 마지막으로 측정된 차선 폭 (초기값: 실측 후 교체 권장)
        self.last_lane_width_2 = 330

        # 마지막으로 정상 검출된 각 차선의 x0 저장
        # 점선이 가려질 때 last_lane_width 추정 대신 이 값을 사용 → 더 안정적
        self.last_line_x0_1 = None
        self.last_dotted_x0_1 = None
        self.last_line_x0_2 = None
        self.last_dotted_x0_2 = None


    def _fit_line_in_roi(self, roi_image, line_type_name):
        """
        주어진 ROI 이미지에서 점들을 추출하고, RANSAC 알고리즘을 사용해 직선을 피팅한 후,
        선 파라미터 [vx, vy, x0, y0]를 반환합니다. (ROI 좌표계 기준)
        """
        if roi_image is None or roi_image.size == 0:
            self.get_logger().warn(f"ROI image for {line_type_name} is empty.")
            return None
            
        h_roi, w_roi = roi_image.shape[:2]
        points_y_coords, points_x_coords = np.where(roi_image > 0)

        if len(points_x_coords) < 2: 
            self.get_logger().info(f"Not enough points ({len(points_x_coords)}) for {line_type_name} RANSAC fitting in ROI.")
            return None

        X_data = points_y_coords.reshape(-1, 1)
        y_target = points_x_coords

        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=2,
                residual_threshold=None,
                max_trials=100,
                random_state=42
            )
            ransac.fit(X_data, y_target)

            inlier_mask = ransac.inlier_mask_
            if np.sum(inlier_mask) < 15:
                self.get_logger().info(
                    f"Too few inliers ({np.sum(inlier_mask)}) for {line_type_name}. 차선 미검출로 처리.")
                return None

            slope_m = ransac.estimator_.coef_[0]
            intercept_c = ransac.estimator_.intercept_

            x0 = intercept_c
            y0 = 0.0

            vx_unnormalized = slope_m
            vy_unnormalized = 1.0
            
            norm = np.sqrt(vx_unnormalized**2 + vy_unnormalized**2)
            if norm < 1e-9:
                self.get_logger().warn(f"Normalization factor is too small for {line_type_name} during RANSAC parameter conversion.")
                if abs(vx_unnormalized) > abs(vy_unnormalized):
                    vx = np.sign(vx_unnormalized)
                    vy = 0.0
                else:
                    vx = 0.0
                    vy = np.sign(vy_unnormalized)
            else:
                vx = vx_unnormalized / norm
                vy = vy_unnormalized / norm
            
            return (vx, vy, x0, y0)

        except ValueError as ve:
            self.get_logger().warn(f"RANSAC fitting failed for {line_type_name} in ROI (ValueError): {ve}")
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected error during RANSAC fitting for {line_type_name} in ROI: {e}")
            return None


    def _get_line_endpoints_in_image(self, params, h_img, w_img):
        if params is None:
            return None, None
        vx, vy, x0, y0 = params

        if abs(vy) > 1e-6:
            x_top = x0 - (y0 * vx / vy)
            x_bottom = x0 + ((h_img - 1 - y0) * vx / vy)
            x_top = int(np.clip(x_top, 0, w_img - 1))
            x_bottom = int(np.clip(x_bottom, 0, w_img - 1))
            pt1 = (x_top, 0)
            pt2 = (x_bottom, h_img - 1)
        else:
            x = int(x0)
            pt1 = (x, 0)
            pt2 = (x, h_img - 1)

        return pt1, pt2


    def _process_detections_and_visualize(self, detection_msg: DetectionArray, callback_id: str, src_mat_orig_ref):
        if len(detection_msg.detections) == 0:
            self.get_logger().info(f"detections_{callback_id}: No detections received.")
            return None, None, None

        line_edge_image_float = CPFL.draw_edges(detection_msg, cls_name='line', color=255)
        dotted_line_edge_image_float = CPFL.draw_edges(detection_msg, cls_name='dotted_line', color=255)

        if line_edge_image_float is None and dotted_line_edge_image_float is None:
            h_default, w_default = 480, 640
            all_lane_edge_combined_float = np.zeros((h_default, w_default), dtype=np.float32)
            self.get_logger().warn(f"detections_{callback_id}: Both line and dotted_line edge images are None. Creating empty base image.")
        elif line_edge_image_float is None:
            if dotted_line_edge_image_float is not None:
                all_lane_edge_combined_float = dotted_line_edge_image_float
            else:
                h_default, w_default = 480, 640
                all_lane_edge_combined_float = np.zeros((h_default, w_default), dtype=np.float32)
        elif dotted_line_edge_image_float is None:
            all_lane_edge_combined_float = line_edge_image_float
        else:
            all_lane_edge_combined_float = line_edge_image_float + dotted_line_edge_image_float

        all_lane_edge_image_uint8 = cv2.convertScaleAbs(all_lane_edge_combined_float)
        (h_orig, w_orig) = all_lane_edge_image_uint8.shape[:2]
        vis_image_orig_space = cv2.cvtColor(all_lane_edge_image_uint8, cv2.COLOR_GRAY2BGR)

        dst_mat_bev_ref = [[round(w_orig * 0.3), round(h_orig * 0.0)], [round(w_orig * 0.7), round(h_orig * 0.0)],
                           [round(w_orig * 0.7), h_orig], [round(w_orig * 0.3), h_orig]]

        M_bird_inv = cv2.getPerspectiveTransform(np.float32(dst_mat_bev_ref), np.float32(src_mat_orig_ref))
        cutting_idx = 250

        all_lane_bird_image_combined = CPFL.bird_convert(all_lane_edge_image_uint8, srcmat=src_mat_orig_ref, dstmat=dst_mat_bev_ref)

        roi_image_for_laneinfo = None

        if all_lane_bird_image_combined is not None:
            reference_roi_from_combined = CPFL.roi_rectangle_below(all_lane_bird_image_combined, cutting_idx=cutting_idx)
            if reference_roi_from_combined is not None and reference_roi_from_combined.size > 0:
                h_ref_roi, w_ref_roi = reference_roi_from_combined.shape[:2]
                roi_image_for_laneinfo = np.zeros((h_ref_roi, w_ref_roi), dtype=np.uint8)
            else:
                self.get_logger().warn(f"detections_{callback_id}: Combined ROI for laneinfo is empty.")
        else:
            self.get_logger().warn(f"detections_{callback_id}: Combined bird image for LaneInfo ROI reference is None.")

        if roi_image_for_laneinfo is None:
            h_default_roi, w_default_roi = 100, 100
            self.get_logger().warn(f"detections_{callback_id}: Defaulting to a small black ROI image for laneinfo due to previous errors.")
            roi_image_for_laneinfo = np.zeros((h_default_roi, w_default_roi), dtype=np.uint8)

        lane_types_to_process = [
            ("line", line_edge_image_float, (0, 255, 0), f"line_{callback_id}"),
            ("dotted_line", dotted_line_edge_image_float, (255, 0, 0), f"dotted_line_{callback_id}")
        ]
        fitted_results = {}

        for type_name, edge_float_img, color, log_prefix in lane_types_to_process:
            if edge_float_img is None or edge_float_img.size == 0:
                self.get_logger().warn(f"{log_prefix}: Edge image is empty. Skipping RANSAC for this type.")
                continue

            edge_uint8_img = cv2.convertScaleAbs(edge_float_img)
            individual_bird_image = CPFL.bird_convert(edge_uint8_img, srcmat=src_mat_orig_ref, dstmat=dst_mat_bev_ref)
            if individual_bird_image is None or individual_bird_image.size == 0:
                self.get_logger().warn(f"{log_prefix}: Individual Bird image is empty after conversion.")
                continue

            individual_roi_image_raw = CPFL.roi_rectangle_below(individual_bird_image, cutting_idx=cutting_idx)
            if individual_roi_image_raw is None or individual_roi_image_raw.size == 0:
                self.get_logger().warn(f"{log_prefix}: Individual ROI image for RANSAC is empty.")
                continue

            individual_roi_image_uint8 = cv2.convertScaleAbs(individual_roi_image_raw)

            if self.show_image:
                cv2.imshow(f'{log_prefix}_roi', individual_roi_image_uint8)

            params_roi = self._fit_line_in_roi(individual_roi_image_uint8, log_prefix + "_roi")

            if params_roi:
                h_roi_ind, w_roi_ind = individual_roi_image_uint8.shape[:2]
                pt1_roi_ind, pt2_roi_ind = self._get_line_endpoints_in_image(params_roi, h_roi_ind, w_roi_ind)

                if pt1_roi_ind and pt2_roi_ind:
                    x0_check = params_roi[2]
                    roi_center_x = w_roi_ind / 2

                    if callback_id == "1":
                        # Camera 1 (직진): 실선(line)=왼쪽, 점선(dotted_line)=오른쪽
                        if type_name == "line" and x0_check > roi_center_x:
                            self.get_logger().warn(
                                f"[1] line x0({x0_check:.1f}) > 중앙({roi_center_x:.1f}): 위치 이상(오른쪽으로 튐), 버림")
                            continue
                        if type_name == "dotted_line" and x0_check < roi_center_x:
                            self.get_logger().warn(
                                f"[1] dotted_line x0({x0_check:.1f}) < 중앙({roi_center_x:.1f}): 위치 이상(왼쪽으로 튐), 버림")
                            continue

                    elif callback_id == "2":
                        # Camera 2 (후진): 점선(dotted_line)=왼쪽, 실선(line)=오른쪽
                        if type_name == "dotted_line" and x0_check > roi_center_x:
                            self.get_logger().warn(
                                f"[2] dotted_line x0({x0_check:.1f}) > 중앙({roi_center_x:.1f}): 위치 이상(오른쪽으로 튐), 버림")
                            continue
                        if type_name == "line" and x0_check < roi_center_x:
                            self.get_logger().warn(
                                f"[2] line x0({x0_check:.1f}) < 중앙({roi_center_x:.1f}): 위치 이상(왼쪽으로 튐), 버림")
                            continue

                    # 원본 공간 시각화
                    pt1_bird_transform = (pt1_roi_ind[0], pt1_roi_ind[1] + cutting_idx)
                    pt2_bird_transform = (pt2_roi_ind[0], pt2_roi_ind[1] + cutting_idx)

                    points_to_transform = np.float32([[pt1_bird_transform, pt2_bird_transform]]).reshape(-1, 1, 2)
                    transformed_points = cv2.perspectiveTransform(points_to_transform, M_bird_inv)

                    if transformed_points is not None and len(transformed_points) == 2:
                        pt1_orig = tuple(map(int, transformed_points[0, 0]))
                        pt2_orig = tuple(map(int, transformed_points[1, 0]))
                        cv2.line(vis_image_orig_space, pt1_orig, pt2_orig, color, 2)
                    else:
                        self.get_logger().warn(f"{log_prefix}: Perspective transform failed for RANSAC line on original image.")

                    if roi_image_for_laneinfo is not None:
                        if roi_image_for_laneinfo.shape[0] == h_roi_ind and roi_image_for_laneinfo.shape[1] == w_roi_ind:
                            cv2.line(roi_image_for_laneinfo, pt1_roi_ind, pt2_roi_ind, 255, 1)
                        else:
                            self.get_logger().warn(f"{log_prefix}: Shape mismatch between individual ROI ({h_roi_ind},{w_roi_ind}) and combined ROI canvas ({roi_image_for_laneinfo.shape[:2]}).")
                            cv2.line(roi_image_for_laneinfo, pt1_roi_ind, pt2_roi_ind, 255, 1)

                    fitted_results[type_name] = (params_roi, individual_roi_image_uint8.shape)

        display_all_lane_bird_image = all_lane_bird_image_combined

        # --- 보정용 반대 차선 예측 ---
        if len(fitted_results) == 2:
            x0_line = fitted_results['line'][0][2]
            x0_dot  = fitted_results['dotted_line'][0][2]
            measured_width = abs(x0_line - x0_dot)

            lane_center_x = (x0_line + x0_dot) / 2
            self.get_logger().info(f"[{callback_id}] 차선 중앙 x: {lane_center_x:.1f}")

            lane_width_ref = self.last_lane_width_1 if callback_id == "1" else self.last_lane_width_2

            if callback_id == "1":
                # Camera 1 정상: x0_line(왼쪽) < x0_dot(오른쪽)
                # 역전: x0_line >= x0_dot → line이 오른쪽으로 튐 → line 버림
                if x0_line >= x0_dot:
                    self.get_logger().warn(f"[1] line 역전 감지 (line이 오른쪽으로 튐). 예측으로 대체.")
                    del fitted_results['line']
                elif lane_width_ref > 0 and measured_width < lane_width_ref * 0.7:
                    self.get_logger().warn(
                        f"[1] 측정 폭({measured_width:.1f}) < 기준 폭({lane_width_ref:.1f}) * 0.7. "
                        f"line 오염 의심. 예측으로 대체.")
                    del fitted_results['line']

            elif callback_id == "2":
                # Camera 2 정상: x0_dot(왼쪽) < x0_line(오른쪽)
                # 역전: x0_dot >= x0_line → dotted가 오른쪽으로 튐 → dotted 버림
                if x0_dot >= x0_line:
                    self.get_logger().warn(f"[2] dotted_line 역전 감지 (dotted가 오른쪽으로 튐). 예측으로 대체.")
                    del fitted_results['dotted_line']
                elif lane_width_ref > 0 and measured_width < lane_width_ref * 0.7:
                    self.get_logger().warn(
                        f"[2] 측정 폭({measured_width:.1f}) < 기준 폭({lane_width_ref:.1f}) * 0.7. "
                        f"dotted_line 오염 의심. 예측으로 대체.")
                    del fitted_results['dotted_line']

            # 정상일 때만 폭 업데이트
            if len(fitted_results) == 2:
                self.get_logger().info(f"[{callback_id}] 양쪽 차선 정상. 폭: {measured_width:.2f}")
                if 150 < measured_width < 600:
                    if callback_id == "1":
                        self.last_lane_width_1 = measured_width
                        self.last_line_x0_1 = fitted_results['line'][0][2]
                        self.last_dotted_x0_1 = fitted_results['dotted_line'][0][2]
                        self.get_logger().info(
                            f"[1] x0 저장: line={self.last_line_x0_1:.1f}, dotted={self.last_dotted_x0_1:.1f}")
                    else:
                        self.last_lane_width_2 = measured_width
                        self.last_line_x0_2 = fitted_results['line'][0][2]
                        self.last_dotted_x0_2 = fitted_results['dotted_line'][0][2]
                        self.get_logger().info(
                            f"[2] x0 저장: line={self.last_line_x0_2:.1f}, dotted={self.last_dotted_x0_2:.1f}")

        if len(fitted_results) == 1:
            existing_type = list(fitted_results.keys())[0]
            params_roi, (h_roi, w_roi) = fitted_results[existing_type]
            vx, vy, x0, y0 = params_roi
            lane_width = self.last_lane_width_1 if callback_id == "1" else self.last_lane_width_2

            if callback_id == "1":
                # Camera 1: line=왼쪽, dotted=오른쪽
                if existing_type == 'line':
                    # 점선이 안 보임 → 마지막으로 본 점선 위치 사용 (없으면 폭으로 추정)
                    x0_new = self.last_dotted_x0_1 if self.last_dotted_x0_1 is not None else x0 + lane_width
                else:
                    # 실선이 안 보임 → 마지막으로 본 실선 위치 사용
                    x0_new = self.last_line_x0_1 if self.last_line_x0_1 is not None else x0 - lane_width
            elif callback_id == "2":
                # Camera 2: dotted=왼쪽, line=오른쪽
                if existing_type == 'dotted_line':
                    # 실선이 안 보임 → 마지막으로 본 실선 위치 사용
                    x0_new = self.last_line_x0_2 if self.last_line_x0_2 is not None else x0 + lane_width
                else:
                    # 점선이 안 보임 → 마지막으로 본 점선 위치 사용
                    x0_new = self.last_dotted_x0_2 if self.last_dotted_x0_2 is not None else x0 - lane_width
            else:
                x0_new = x0  # fallback

            new_params = (vx, vy, x0_new, y0)
            pt1_pred, pt2_pred = self._get_line_endpoints_in_image(new_params, h_roi, w_roi)
            if pt1_pred and pt2_pred:
                cv2.line(roi_image_for_laneinfo, pt1_pred, pt2_pred, 255, 1)
                self.get_logger().info(
                    f"{existing_type}: Predicted opposite lane added with offset {lane_width} for callback_id={callback_id}."
                )

        # 두 차선 모두 검출 실패 시 발행하지 않음
        if len(fitted_results) == 0:
            self.get_logger().warn(f"[{callback_id}] 유효한 차선 없음. 이번 프레임 건너뜀.")
            return vis_image_orig_space, display_all_lane_bird_image, None

        return vis_image_orig_space, display_all_lane_bird_image, roi_image_for_laneinfo


    def yolov8_detections_1_callback(self, detection_msg: DetectionArray):
        vis_image_orig_space, all_lane_bird_image, roi_image_for_laneinfo = \
            self._process_detections_and_visualize(detection_msg, "1", self.src_mat_orig_ref_1)

        if roi_image_for_laneinfo is None:
            if self.show_image and vis_image_orig_space is not None:
                cv2.imshow('ransac_lines_on_original_1', vis_image_orig_space)
                if all_lane_bird_image is not None:
                    cv2.imshow('lane_bird_img_1_combined', all_lane_bird_image)
                cv2.waitKey(1)
            return

        if self.show_image:
            if vis_image_orig_space is not None:
                cv2.imshow('ransac_lines_on_original_1', vis_image_orig_space)
            if all_lane_bird_image is not None:
                cv2.imshow('lane_bird_img_1_combined', all_lane_bird_image)
            cv2.imshow('roi_img_1_for_laneinfo_RANSAC_lines', roi_image_for_laneinfo)
            cv2.waitKey(1)

        roi_image_uint8_final = roi_image_for_laneinfo
        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image_uint8_final, encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image for detections_1: {e}")

        grad = CPFL.dominant_gradient(roi_image_uint8_final, theta_limit=70)
        target_points = []
        if roi_image_uint8_final is not None and roi_image_uint8_final.shape[0] > 10:
            for target_point_y_roi in range(5, round(roi_image_uint8_final.shape[0] * 0.9), 50):
                target_point_x_roi = CPFL.get_lane_center(roi_image_uint8_final, detection_height=target_point_y_roi,
                                                    detection_thickness=10, road_gradient=grad, lane_width=300)
                if target_point_x_roi is not None:
                    target_point = TargetPoint()
                    target_point.target_x = round(float(target_point_x_roi))
                    target_point.target_y = round(float(target_point_y_roi))
                    target_points.append(target_point)
        else:
            self.get_logger().warn("ROI image for lane info calculation (detections_1) is too small or None.")

        lane = LaneInfo()
        lane.slope = float(grad) if grad is not None else 0.0
        lane.target_points = target_points
        self.publisher_1.publish(lane)

    def yolov8_detections_2_callback(self, detection_msg: DetectionArray):
        vis_image_orig_space, all_lane_bird_image, roi_image_for_laneinfo = \
            self._process_detections_and_visualize(detection_msg, "2", self.src_mat_orig_ref_2)

        if roi_image_for_laneinfo is None:
            if self.show_image and vis_image_orig_space is not None:
                cv2.imshow('ransac_lines_on_original_2', vis_image_orig_space)
                if all_lane_bird_image is not None:
                    cv2.imshow('lane_bird_img_2_combined', all_lane_bird_image)
                cv2.waitKey(1)
            return

        if self.show_image:
            if vis_image_orig_space is not None:
                cv2.imshow('ransac_lines_on_original_2', vis_image_orig_space)
            if all_lane_bird_image is not None:
                cv2.imshow('lane_bird_img_2_combined', all_lane_bird_image)
            cv2.imshow('roi_img_2_for_laneinfo_RANSAC_lines', roi_image_for_laneinfo)
            cv2.waitKey(1)

        roi_image_uint8_final = roi_image_for_laneinfo
        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image_uint8_final, encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image for detections_2: {e}")

        grad = CPFL.dominant_gradient(roi_image_uint8_final, theta_limit=70)
        target_points = []
        if roi_image_uint8_final is not None and roi_image_uint8_final.shape[0] > 10:
            for target_point_y_roi in range(5, round(roi_image_uint8_final.shape[0] * 0.9), 50):
                target_point_x_roi = CPFL.get_lane_center(roi_image_uint8_final, detection_height=target_point_y_roi,
                                                    detection_thickness=10, road_gradient=grad, lane_width=300)
                if target_point_x_roi is not None:
                    target_point = TargetPoint()
                    target_point.target_x = round(float(target_point_x_roi))
                    target_point.target_y = round(float(target_point_y_roi))
                    target_points.append(target_point)
        else:
            self.get_logger().warn("ROI image for lane info calculation (detections_2) is too small or None.")

        # target_points 시각화
        if self.show_image and roi_image_uint8_final is not None:
            vis_image_with_points = cv2.cvtColor(roi_image_uint8_final, cv2.COLOR_GRAY2BGR)
            for pt in target_points:
                cv2.circle(vis_image_with_points, (pt.target_x, pt.target_y), 3, (0, 0, 255), -1)
            cv2.imshow('roi_img_2_with_target_points', vis_image_with_points)
            cv2.waitKey(1)

        lane = LaneInfo()
        lane.slope = float(grad) if grad is not None else 0.0
        lane.target_points = target_points
        self.publisher_2.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()