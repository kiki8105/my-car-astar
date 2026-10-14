# my-car-astar

ROS 2 Humble용 주행 실습 패키지입니다. 하나의 노드가 다음 순서로 동작합니다.

1. `/map`의 `OccupancyGrid`를 2차원 격자로 해석합니다.
2. `/amcl_pose`에서 현재 위치를 받고 `/goal_pose`에서 목표를 받습니다.
3. `heapq` 기반 A*로 현재 위치에서 목표까지의 최소 비용 경로를 만듭니다.
4. 생성한 경로를 `/planned_path`로 발행합니다.
5. Pure Pursuit 공식으로 경로를 추종하며 `/cmd_vel`을 발행합니다.
6. YOLOv8이 웹캠에서 `person`을 신뢰도 0.7 이상으로 검출하면 `/cmd_vel`을 0으로 만들어 비상정지합니다.

구현 범위는 강의 페이지의 [A*·Pure Pursuit 주행 과제](https://app.notion.com/p/1-3f9a63e50cdf80c482cdd07f8f3e4eac)와 [YOLOv8 웹캠 객체 검출 예제](https://app.notion.com/p/1-3f9a63e50cdf80acb1dfcf9f9b0a4a00)를 기준으로 했습니다.

## 주요 토픽

| 구분 | 이름 | 형식 |
|---|---|---|
| 구독 | `/map` | `nav_msgs/msg/OccupancyGrid` |
| 구독 | `/amcl_pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` |
| 구독 | `/goal_pose` | `geometry_msgs/msg/PoseStamped` |
| 발행 | `/planned_path` | `nav_msgs/msg/Path` |
| 발행 | `/cmd_vel` | `geometry_msgs/msg/Twist` |

## 설치

YOLOv8 예제에서 사용하는 `ultralytics`를 프로젝트 전용 가상환경에 설치합니다. CPU 실습 PC에서 CUDA 패키지를 받지 않도록 PyTorch CPU 저장소를 사용합니다.

```bash
cd /home/kiki/turtlebot3/src/my_car_astar
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install \
  'numpy<2' 'opencv-python>=4.7,<4.13' ultralytics
```

패키지를 빌드합니다.

```bash
cd /home/kiki/turtlebot3
source /opt/ros/humble/setup.bash
/home/kiki/turtlebot3/src/my_car_astar/.venv/bin/python -m colcon build \
  --packages-select my_car_astar --symlink-install
source install/setup.bash
```

첫 실행에서는 `yolov8n.pt`가 자동으로 내려받아집니다.

## 실행

먼저 지도 서버와 AMCL이 실행 중이어야 합니다. RViz 또는 기존 컨트롤러에서 `Initial Pose`를 지정하면 `/amcl_pose`가 들어옵니다.

```bash
export ROS_DOMAIN_ID=64
ros2 launch my_car_astar astar_drive.launch.py
```

RViz 또는 기존 컨트롤러에서 목표를 지정해 `/goal_pose`를 발행합니다. 정상 동작 시 터미널에 `A* 경로 생성 완료`가 출력되고 로봇이 Pure Pursuit 방식으로 경로를 추종합니다.

웹캠 앞에 사람이 나타나 YOLOv8의 `person` 검출 확률이 0.7 이상이 되면 다음 로그와 함께 정지합니다.

```text
person 검출: 비상정지 cmd_vel을 발행합니다.
```

사람이 화면에서 사라지면 기존 A* 경로 추종을 계속합니다.

## 파라미터

설정 파일은 `config/astar_drive.yaml`입니다.

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `lookahead_distance` | `0.5` | Pure Pursuit 전방 주시 거리(m) |
| `linear_velocity` | `0.1` | 선속도(m/s) |
| `goal_tolerance` | `0.2` | 목표 도착 허용 거리(m) |
| `max_angular_velocity` | `1.0` | 최대 각속도(rad/s) |
| `occupied_threshold` | `50` | A*가 장애물로 처리할 지도 값 |
| `camera_index` | `0` | `cv2.VideoCapture()`에 전달할 웹캠 번호 |
| `yolo_model` | `yolov8n.pt` | YOLOv8 모델 파일 |
| `confidence_threshold` | `0.7` | 객체 검출 최소 확신도 |
| `stop_class` | `person` | 검출 시 정지할 클래스 이름 |
| `show_detection` | `true` | YOLO 검출 창 표시 여부 |
| `require_detector` | `true` | 검출기를 준비하지 못하면 주행 금지 |

## 확인 명령

```bash
ros2 topic echo /planned_path
ros2 topic echo /cmd_vel
```

테스트는 다음과 같이 실행합니다.

```bash
cd /home/kiki/turtlebot3/src/my_car_astar
source /opt/ros/humble/setup.bash
python3 -m pytest -q
```
