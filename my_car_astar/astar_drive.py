#!/usr/bin/env python3
"""A* path planning + Pure Pursuit driving + YOLOv8 emergency stop.

The implementation follows the two supplied course pages:
* OccupancyGrid, current pose, and goal are combined into one A* driving node.
* The generated path is followed with the Pure Pursuit steering formula.
* Mapped obstacles receive robot-clearance inflation; unexpected LiDAR returns
    and selected YOLO detections force cmd_vel to zero.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


GridPoint = Tuple[int, int]
WorldPoint = Tuple[float, float]


def normalize_angle(angle: float) -> float:
    """Return an angle in the [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def front_obstacle_distance(
    scan: LaserScan,
    half_angle: float,
    pose: Optional[Tuple[float, float, float]] = None,
    map_message: Optional[OccupancyGrid] = None,
    occupied_threshold: int = 50,
    sensor_offset: float = 0.064,
    mapped_obstacle_tolerance: float = 0.12,
) -> float:
    """Return the nearest forward return not already represented as occupied."""
    if scan.angle_increment == 0.0:
        return math.inf

    distances = []
    for index, distance in enumerate(scan.ranges):
        angle = normalize_angle(scan.angle_min + index * scan.angle_increment)
        if abs(angle) > half_angle:
            continue
        if (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
        ):
            if pose is not None and map_message is not None:
                sensor_x = pose[0] - sensor_offset * math.cos(pose[2])
                sensor_y = pose[1] - sensor_offset * math.sin(pose[2])
                hit = (
                    sensor_x + distance * math.cos(pose[2] + angle),
                    sensor_y + distance * math.sin(pose[2] + angle),
                )
                info = map_message.info
                origin = (info.origin.position.x, info.origin.position.y)
                cell_x, cell_y = world_to_grid(
                    hit,
                    origin,
                    info.resolution,
                )
                if info.resolution > 0.0:
                    tolerance_cells = math.ceil(
                        mapped_obstacle_tolerance / info.resolution
                    )
                    occupied_nearby = False
                    for offset_y in range(
                        -tolerance_cells,
                        tolerance_cells + 1,
                    ):
                        for offset_x in range(
                            -tolerance_cells,
                            tolerance_cells + 1,
                        ):
                            if (
                                offset_x * offset_x + offset_y * offset_y
                                > tolerance_cells * tolerance_cells
                            ):
                                continue
                            neighbor_x = cell_x + offset_x
                            neighbor_y = cell_y + offset_y
                            if not (
                                0 <= neighbor_x < info.width
                                and 0 <= neighbor_y < info.height
                            ):
                                continue
                            cell_value = map_message.data[
                                neighbor_y * info.width + neighbor_x
                            ]
                            if cell_value >= occupied_threshold:
                                occupied_nearby = True
                                break
                        if occupied_nearby:
                            break
                    if occupied_nearby:
                        continue
            distances.append(distance)
    return min(distances, default=math.inf)


def front_obstacle_image_box(
    scan: LaserScan,
    nearest_distance: float,
    camera_info: CameraInfo,
    transform,
    half_angle: float,
    obstacle_height: float = 0.4,
    lidar_height: float = 0.121,
) -> Optional[Tuple[int, int, int, int]]:
    """Project the nearest LiDAR return cluster through TF and camera intrinsics."""
    nearby_returns = []
    for index, distance in enumerate(scan.ranges):
        angle = normalize_angle(scan.angle_min + index * scan.angle_increment)
        if (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
            and abs(angle) <= half_angle
            and abs(distance - nearest_distance) <= 0.12
        ):
            nearby_returns.append((distance, angle))
    if not nearby_returns:
        nearby_returns.append((nearest_distance, 0.0))

    rotation = transform.transform.rotation
    translation = transform.transform.translation
    qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
    camera_points = []
    for distance, angle in nearby_returns:
        scan_x = distance * math.cos(angle)
        scan_y = distance * math.sin(angle)
        for scan_z in (-lidar_height, obstacle_height - lidar_height):
            cross_x = 2.0 * (qy * scan_z - qz * scan_y)
            cross_y = 2.0 * (qz * scan_x - qx * scan_z)
            cross_z = 2.0 * (qx * scan_y - qy * scan_x)
            camera_x = (
                scan_x + qw * cross_x + qy * cross_z - qz * cross_y
                + translation.x
            )
            camera_y = (
                scan_y + qw * cross_y + qz * cross_x - qx * cross_z
                + translation.y
            )
            camera_z = (
                scan_z + qw * cross_z + qx * cross_y - qy * cross_x
                + translation.z
            )
            if camera_z <= 0.01:
                continue
            k = camera_info.k
            pixel_x = (k[0] * camera_x + k[1] * camera_y) / camera_z + k[2]
            pixel_y = (k[3] * camera_x + k[4] * camera_y) / camera_z + k[5]
            if (
                -camera_info.width <= pixel_x <= 2 * camera_info.width
                and -camera_info.height <= pixel_y <= 2 * camera_info.height
            ):
                camera_points.append((pixel_x, pixel_y))

    if not camera_points:
        return None

    padding = 8
    x1 = max(0, math.floor(min(point[0] for point in camera_points)) - padding)
    y1 = max(0, math.floor(min(point[1] for point in camera_points)) - padding)
    x2 = min(
        camera_info.width - 1,
        math.ceil(max(point[0] for point in camera_points)) + padding,
    )
    y2 = min(
        camera_info.height - 1,
        math.ceil(max(point[1] for point in camera_points)) + padding,
    )
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert a quaternion to the planar yaw used by the driving formula."""
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def world_to_grid(
    point: WorldPoint,
    origin: WorldPoint,
    resolution: float,
) -> GridPoint:
    """Convert a map-frame coordinate to an OccupancyGrid cell."""
    return (
        int(math.floor((point[0] - origin[0]) / resolution)),
        int(math.floor((point[1] - origin[1]) / resolution)),
    )


def grid_to_world(
    point: GridPoint,
    origin: WorldPoint,
    resolution: float,
) -> WorldPoint:
    """Return the map-frame coordinate at the center of a grid cell."""
    return (
        origin[0] + (point[0] + 0.5) * resolution,
        origin[1] + (point[1] + 0.5) * resolution,
    )


def _neighbors(point: GridPoint) -> Iterable[Tuple[GridPoint, float]]:
    """Yield the eight adjacent cells and their movement costs."""
    x, y = point
    diagonal = math.sqrt(2.0)
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, diagonal),
        (-1, 1, diagonal),
        (1, -1, diagonal),
        (1, 1, diagonal),
    )
    for dx, dy, cost in moves:
        yield (x + dx, y + dy), cost


def _heuristic(first: GridPoint, second: GridPoint) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def astar_path(
    data: Sequence[int],
    width: int,
    height: int,
    start: GridPoint,
    goal: GridPoint,
    occupied_threshold: int = 50,
    clearance_cells: int = 0,
) -> List[GridPoint]:
    """Find a minimum-cost path through an OccupancyGrid using heapq.

    Negative cells are unknown and treated as unavailable. Cells whose value is
    at least ``occupied_threshold`` are also unavailable.
    """
    if width <= 0 or height <= 0 or len(data) != width * height:
        return []

    def in_bounds(cell: GridPoint) -> bool:
        return 0 <= cell[0] < width and 0 <= cell[1] < height

    def is_free(cell: GridPoint) -> bool:
        if not in_bounds(cell):
            return False
        radius_squared = clearance_cells * clearance_cells
        for dy in range(-clearance_cells, clearance_cells + 1):
            for dx in range(-clearance_cells, clearance_cells + 1):
                if dx * dx + dy * dy > radius_squared:
                    continue
                neighbor = (cell[0] + dx, cell[1] + dy)
                if not in_bounds(neighbor):
                    return False
                value = data[neighbor[1] * width + neighbor[0]]
                if value < 0 or value >= occupied_threshold:
                    return False
        return True

    if not is_free(start) or not is_free(goal):
        return []

    open_heap: List[Tuple[float, float, GridPoint]] = []
    heapq.heappush(open_heap, (_heuristic(start, goal), 0.0, start))
    came_from: Dict[GridPoint, GridPoint] = {}
    cost_so_far: Dict[GridPoint, float] = {start: 0.0}
    closed = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)
        for neighbor, step_cost in _neighbors(current):
            if not is_free(neighbor) or neighbor in closed:
                continue

            # A diagonal move cannot cut through the corner of an obstacle.
            dx = neighbor[0] - current[0]
            dy = neighbor[1] - current[1]
            if dx != 0 and dy != 0:
                if not is_free((current[0] + dx, current[1])):
                    continue
                if not is_free((current[0], current[1] + dy)):
                    continue

            new_cost = current_cost + step_cost
            if new_cost >= cost_so_far.get(neighbor, math.inf):
                continue

            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            priority = new_cost + _heuristic(neighbor, goal)
            heapq.heappush(open_heap, (priority, new_cost, neighbor))

    return []


def select_lookahead_index(
    path: Sequence[WorldPoint],
    current: WorldPoint,
    lookahead_distance: float,
    previous_index: int = 0,
) -> int:
    """Choose the first path point at least the look-ahead distance away."""
    if not path:
        return 0

    start_index = max(0, min(previous_index, len(path) - 1))
    nearest_index = min(
        range(start_index, len(path)),
        key=lambda index: math.hypot(
            path[index][0] - current[0],
            path[index][1] - current[1],
        ),
    )

    for index in range(nearest_index, len(path)):
        distance = math.hypot(
            path[index][0] - current[0],
            path[index][1] - current[1],
        )
        if distance >= lookahead_distance:
            return index
    return len(path) - 1


def next_route_waypoint(
    goals: Sequence[WorldPoint],
    current_index: int,
) -> Optional[Tuple[int, WorldPoint]]:
    """Return the next ordered goal, or None when the route is complete."""
    next_index = current_index + 1
    if next_index >= len(goals):
        return None
    return next_index, goals[next_index]


class AStarDrive(Node):
    """Plan with A*, follow with Pure Pursuit, and stop on YOLO detection."""

    def __init__(self) -> None:
        super().__init__('astar_drive')

        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('linear_velocity', 0.1)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('robot_radius', 0.2)
        self.declare_parameter('obstacle_stop_distance', 0.45)
        self.declare_parameter('obstacle_front_angle', 0.35)
        self.declare_parameter('mapped_obstacle_tolerance', 0.12)
        self.declare_parameter('scan_timeout', 1.0)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('annotated_image_topic', '/yolo/annotated_image')
        self.declare_parameter('obstacle_bbox_height', 0.4)
        self.declare_parameter('lidar_height_above_ground', 0.121)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('show_detection', True)
        self.declare_parameter('require_detector', True)
        self.declare_parameter('drive_enabled', True)
        self.declare_parameter('control_period', 0.1)

        self.map_message: Optional[OccupancyGrid] = None
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.goal: Optional[WorldPoint] = None
        self.goals: List[WorldPoint] = []
        self.goal_index = 0
        self.path: List[WorldPoint] = []
        self.path_index = 0
        self.drive_was_enabled = True
        self.stop_object_visible = False
        self.latest_scan: Optional[LaserScan] = None
        self.last_scan_received_at: Optional[float] = None
        self.scan_stop_active = False
        self.latest_image_header = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_publisher = self.create_publisher(Path, '/planned_path', 10)
        annotated_topic = str(
            self.get_parameter('annotated_image_topic').value
        )
        self.annotated_image_publisher = self.create_publisher(
            Image,
            annotated_topic,
            qos_profile_sensor_data,
        )
        self.latest_frame = None
        self.bridge = None
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10,
        )
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(PoseArray, '/goal_poses', self.goals_callback, 10)
        image_topic = str(self.get_parameter('image_topic').value)
        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.cv2 = None
        self.detector = None
        self.camera = None
        self._initialize_detector()

        period = float(self.get_parameter('control_period').value)
        self.timer = self.create_timer(period, self.control_loop)
        self.get_logger().info(
            'A* + Pure Pursuit + YOLOv8 비상정지 노드가 시작되었습니다.'
        )

    @property
    def detector_ready(self) -> bool:
        return (
            self.cv2 is not None
            and self.detector is not None
            and self.bridge is not None
            and self.latest_frame is not None
        )

    def _initialize_detector(self) -> None:
        try:
            import cv2
            from cv_bridge import CvBridge
            from ultralytics import YOLO

            model_path = str(self.get_parameter('yolo_model').value)
            self.cv2 = cv2
            self.bridge = CvBridge()
            self.detector = YOLO(model_path)
            image_topic = str(self.get_parameter('image_topic').value)
            self.get_logger().info(
                f'YOLOv8={model_path}, camera image topic={image_topic} 준비 완료'
            )
        except Exception as error:
            self.cv2 = None
            self.detector = None
            self.bridge = None
            self.get_logger().error(f'객체 검출 초기화 실패: {error}')

    def image_callback(self, message: Image) -> None:
        if self.bridge is None:
            return
        try:
            self.latest_image_header = message.header
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
        except Exception as error:
            self.get_logger().warning(f'Gazebo 카메라 이미지 변환 실패: {error}')

    def scan_callback(self, message: LaserScan) -> None:
        self.latest_scan = message
        self.last_scan_received_at = time.monotonic()

    def camera_info_callback(self, message: CameraInfo) -> None:
        self.latest_camera_info = message

    def map_callback(self, message: OccupancyGrid) -> None:
        self.map_message = message
        self.plan_path()

    def pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        orientation = pose.orientation
        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.current_pose = (pose.position.x, pose.position.y, yaw)
        if self.goal is not None and not self.path:
            self.plan_path()

    def goal_callback(self, message: PoseStamped) -> None:
        self.goals = [(message.pose.position.x, message.pose.position.y)]
        self.goal_index = 0
        self.goal = self.goals[0]
        self.path = []
        self.path_index = 0
        self.get_logger().info(
            f'새 목표 수신: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}'
        )
        self.plan_path()

    def goals_callback(self, message: PoseArray) -> None:
        if not message.poses:
            self.get_logger().warning('빈 A* 경유점 목록은 무시합니다.')
            return

        self.goals = [
            (pose.position.x, pose.position.y)
            for pose in message.poses
        ]
        self.goal_index = 0
        self.goal = self.goals[0]
        self.path = []
        self.path_index = 0
        self.get_logger().info(
            f'A* 다중 경로 수신: {len(self.goals)}개 경유점'
        )
        self.plan_path()

    def plan_path(self) -> None:
        if self.map_message is None or self.current_pose is None or self.goal is None:
            return

        info = self.map_message.info
        origin = (info.origin.position.x, info.origin.position.y)
        start_cell = world_to_grid(
            (self.current_pose[0], self.current_pose[1]),
            origin,
            info.resolution,
        )
        goal_cell = world_to_grid(self.goal, origin, info.resolution)
        cells = astar_path(
            self.map_message.data,
            info.width,
            info.height,
            start_cell,
            goal_cell,
            int(self.get_parameter('occupied_threshold').value),
            clearance_cells=(
                math.ceil(
                    float(self.get_parameter('robot_radius').value)
                    / info.resolution
                    + math.sqrt(2.0) / 2.0
                )
                if float(self.get_parameter('robot_radius').value) > 0.0
                else 0
            ),
        )

        if not cells:
            self.path = []
            self.stop_robot()
            self.get_logger().error('A*가 시작점에서 목표점까지 경로를 찾지 못했습니다.')
            return

        self.path = [grid_to_world(cell, origin, info.resolution) for cell in cells]
        self.path_index = 0
        self.publish_path()
        self.get_logger().info(f'A* 경로 생성 완료: {len(self.path)}개 경로점')

    def publish_path(self) -> None:
        message = Path()
        message.header.frame_id = 'map'
        message.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.path_publisher.publish(message)

    def detect_stop_object(
        self,
        lidar_distance: Optional[float] = None,
        safety_warning: Optional[str] = None,
    ) -> bool:
        if not self.detector_ready or self.latest_frame is None:
            return False

        frame = self.latest_frame
        confidence = float(self.get_parameter('confidence_threshold').value)
        results = self.detector(frame, stream=True, conf=confidence, verbose=False)
        detected = False
        display_frame = frame.copy()
        detected_classes = set()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.detector.names[class_id]
                box_confidence = float(box.conf[0])
                x1, y1, x2, y2 = (
                    int(value) for value in box.xyxy[0].tolist()
                )
                label = f'{class_name} {box_confidence:.2f}'
                self.cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )
                self.cv2.putText(
                    display_frame,
                    label,
                    (x1, max(18, y1 - 6)),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    self.cv2.LINE_AA,
                )
                detected = True
                detected_classes.add(class_name)

        if lidar_distance is not None:
            box = None
            if self.latest_scan is not None and self.latest_camera_info is not None:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.latest_camera_info.header.frame_id,
                        self.latest_scan.header.frame_id,
                        Time(),
                    )
                    box = front_obstacle_image_box(
                        self.latest_scan,
                        lidar_distance,
                        self.latest_camera_info,
                        transform,
                        float(self.get_parameter('obstacle_front_angle').value),
                        float(self.get_parameter('obstacle_bbox_height').value),
                        float(self.get_parameter('lidar_height_above_ground').value),
                    )
                except TransformException as error:
                    self.get_logger().warning(
                        f'LiDAR bbox TF 변환 실패: {error}',
                        throttle_duration_sec=2.0,
                    )

            if box is not None:
                x1, y1, x2, y2 = box
                self.cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3,
                )
                label = f'OBSTACLE / EMERGENCY STOP {lidar_distance:.2f}m'
                self.cv2.putText(
                    display_frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    self.cv2.LINE_AA,
                )
            else:
                safety_warning = (
                    f'LIDAR OBSTACLE / EMERGENCY STOP {lidar_distance:.2f}m'
                )
        if safety_warning is not None:
            frame_height, frame_width = display_frame.shape[:2]
            self.cv2.rectangle(
                display_frame,
                (2, 2),
                (frame_width - 3, frame_height - 3),
                (0, 0, 255),
                3,
            )
            self.cv2.putText(
                display_frame,
                safety_warning,
                (12, 28),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                self.cv2.LINE_AA,
            )

        annotated_message = self.bridge.cv2_to_imgmsg(
            display_frame,
            encoding='bgr8',
        )
        if self.latest_image_header is not None:
            annotated_message.header = self.latest_image_header
        self.annotated_image_publisher.publish(annotated_message)

        if bool(self.get_parameter('show_detection').value):
            self.cv2.imshow('YOLOv8 emergency stop', display_frame)
            self.cv2.waitKey(1)

        if detected != self.stop_object_visible:
            if detected:
                class_summary = ', '.join(sorted(detected_classes))
                self.get_logger().warning(
                    f'객체 검출 ({class_summary}): 비상정지 cmd_vel을 발행합니다.'
                )
            else:
                self.get_logger().info('검출 객체가 화면에서 사라졌습니다.')
        self.stop_object_visible = detected
        return detected

    def control_loop(self) -> None:
        if not bool(self.get_parameter('drive_enabled').value):
            if self.drive_was_enabled:
                self.stop_robot()
                self.get_logger().info(
                    '수동 조종을 위해 A* 자동주행을 정지했습니다.'
                )
            self.drive_was_enabled = False
            self.detect_stop_object()
            return
        self.drive_was_enabled = True

        require_detector = bool(self.get_parameter('require_detector').value)
        if require_detector and not self.detector_ready:
            self.stop_robot()
            return

        scan_timeout = float(self.get_parameter('scan_timeout').value)
        if (
            self.latest_scan is None
            or self.last_scan_received_at is None
            or time.monotonic() - self.last_scan_received_at > scan_timeout
        ):
            self.stop_robot()
            self.detect_stop_object(safety_warning='LIDAR LOST - EMERGENCY STOP')
            self.get_logger().warning(
                '/scan 수신이 없거나 오래되어 안전 정지합니다.',
                throttle_duration_sec=2.0,
            )
            return

        obstacle_distance = front_obstacle_distance(
            self.latest_scan,
            float(self.get_parameter('obstacle_front_angle').value),
            self.current_pose,
            self.map_message,
            int(self.get_parameter('occupied_threshold').value),
            mapped_obstacle_tolerance=float(
                self.get_parameter('mapped_obstacle_tolerance').value
            ),
        )
        stop_distance = float(
            self.get_parameter('obstacle_stop_distance').value
        )
        if obstacle_distance <= stop_distance:
            self.stop_robot()
            self.scan_stop_active = True
            self.detect_stop_object(lidar_distance=obstacle_distance)
            self.get_logger().warning(
                f'전방 장애물 {obstacle_distance:.2f} m: 안전 정지합니다.',
                throttle_duration_sec=1.0,
            )
            return
        if self.scan_stop_active:
            self.scan_stop_active = False
            self.get_logger().info('전방 장애물이 사라져 주행을 재개합니다.')

        if self.detect_stop_object():
            self.stop_robot()
            return

        if self.current_pose is None or self.goal is None or not self.path:
            self.stop_robot()
            return

        current = (self.current_pose[0], self.current_pose[1])
        goal_distance = math.hypot(
            self.goal[0] - current[0],
            self.goal[1] - current[1],
        )
        if goal_distance < float(self.get_parameter('goal_tolerance').value):
            self.stop_robot()
            next_waypoint = next_route_waypoint(
                self.goals,
                self.goal_index,
            )
            if next_waypoint is not None:
                self.goal_index, self.goal = next_waypoint
                self.path = []
                self.path_index = 0
                self.get_logger().info(
                    f'A* 경유점 {self.goal_index}/{len(self.goals)} 도착. '
                    '다음 경유점 경로를 계산합니다.'
                )
                self.plan_path()
                return
            self.get_logger().info(
                '모든 A* 경유점에 도착했습니다.',
                throttle_duration_sec=2.0,
            )
            return

        lookahead = float(self.get_parameter('lookahead_distance').value)
        self.path_index = select_lookahead_index(
            self.path,
            current,
            lookahead,
            self.path_index,
        )
        target_x, target_y = self.path[self.path_index]
        target_angle = math.atan2(target_y - current[1], target_x - current[0])
        alpha = normalize_angle(target_angle - self.current_pose[2])

        linear_velocity = float(self.get_parameter('linear_velocity').value)
        angular_velocity = linear_velocity * (2.0 * math.sin(alpha)) / lookahead
        angular_limit = float(self.get_parameter('max_angular_velocity').value)
        angular_velocity = max(-angular_limit, min(angular_limit, angular_velocity))

        command = Twist()
        command.linear.x = linear_velocity
        command.angular.z = angular_velocity
        self.cmd_publisher.publish(command)

    def stop_robot(self) -> None:
        # SIGINT 처리로 ROS context가 먼저 닫힌 경우에는 발행하지 않습니다.
        if rclpy.ok():
            self.cmd_publisher.publish(Twist())

    def close(self) -> None:
        self.stop_robot()
        if self.cv2 is not None:
            self.cv2.destroyAllWindows()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AStarDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
