#!/usr/bin/env python3
"""A* path planning + Pure Pursuit driving + YOLOv8 emergency stop.

The implementation follows the two supplied course pages:
* OccupancyGrid, current pose, and goal are combined into one A* driving node.
* The generated path is followed with the Pure Pursuit steering formula.
* YOLOv8 reads a webcam frame and forces cmd_vel to zero when the selected
  object is detected.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node


GridPoint = Tuple[int, int]
WorldPoint = Tuple[float, float]


def normalize_angle(angle: float) -> float:
    """Return an angle in the [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


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
        value = data[cell[1] * width + cell[0]]
        return 0 <= value < occupied_threshold

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


class AStarDrive(Node):
    """Plan with A*, follow with Pure Pursuit, and stop on YOLO detection."""

    def __init__(self) -> None:
        super().__init__('astar_drive')

        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('linear_velocity', 0.1)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('stop_class', 'person')
        self.declare_parameter('show_detection', True)
        self.declare_parameter('require_detector', True)
        self.declare_parameter('control_period', 0.1)

        self.map_message: Optional[OccupancyGrid] = None
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.goal: Optional[WorldPoint] = None
        self.path: List[WorldPoint] = []
        self.path_index = 0
        self.stop_object_visible = False

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_publisher = self.create_publisher(Path, '/planned_path', 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10,
        )
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

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
            and self.camera is not None
            and self.camera.isOpened()
        )

    def _initialize_detector(self) -> None:
        try:
            import cv2
            from ultralytics import YOLO

            camera_index = int(self.get_parameter('camera_index').value)
            model_path = str(self.get_parameter('yolo_model').value)
            self.cv2 = cv2
            self.detector = YOLO(model_path)
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f'웹캠 {camera_index}을 열 수 없습니다.')
            self.get_logger().info(
                f'YOLOv8={model_path}, webcam={camera_index} 준비 완료'
            )
        except Exception as error:
            self.cv2 = None
            self.detector = None
            if self.camera is not None:
                self.camera.release()
            self.camera = None
            self.get_logger().error(f'객체 검출 초기화 실패: {error}')

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
        self.goal = (message.pose.position.x, message.pose.position.y)
        self.path = []
        self.path_index = 0
        self.get_logger().info(
            f'새 목표 수신: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}'
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

    def detect_stop_object(self) -> bool:
        if not self.detector_ready:
            return False

        success, frame = self.camera.read()
        if not success:
            self.get_logger().warning('웹캠 프레임을 읽지 못했습니다.')
            return False

        confidence = float(self.get_parameter('confidence_threshold').value)
        stop_class = str(self.get_parameter('stop_class').value)
        results = self.detector(frame, stream=True, conf=confidence, verbose=False)
        detected = False
        display_frame = frame

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.detector.names[class_id]
                if class_name == stop_class:
                    detected = True
            display_frame = result.plot()

        if bool(self.get_parameter('show_detection').value):
            self.cv2.imshow('YOLOv8 emergency stop', display_frame)
            self.cv2.waitKey(1)

        if detected != self.stop_object_visible:
            if detected:
                self.get_logger().warning(
                    f'{stop_class} 검출: 비상정지 cmd_vel을 발행합니다.'
                )
            else:
                self.get_logger().info(f'{stop_class}이 화면에서 사라졌습니다.')
        self.stop_object_visible = detected
        return detected

    def control_loop(self) -> None:
        require_detector = bool(self.get_parameter('require_detector').value)
        if require_detector and not self.detector_ready:
            self.stop_robot()
            return

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
            self.get_logger().info(
                '목표에 도착했습니다.',
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
        if self.camera is not None:
            self.camera.release()
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
