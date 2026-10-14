# my-car-astar

ROS 2 Humble용 주행 실습 패키지입니다. 하나의 노드가 다음 순서로 동작합니다.

1. `/map`의 `OccupancyGrid`를 2차원 격자로 해석합니다.
2. `/amcl_pose`에서 현재 위치를 받고 `/goal_pose`에서 목표를 받습니다.
3. `heapq` 기반 A*로 현재 위치에서 목표까지의 최소 비용 경로를 만듭니다.
4. 생성한 경로를 `/planned_path`로 발행합니다.
5. Pure Pursuit 공식으로 경로를 추종하며 `/cmd_vel`을 발행합니다.
6. A*는 장애물 주변에 Waffle Pi 차체 여유를 두고 경로를 계획합니다. 지도상 자유/미지 공간에서 LiDAR가 새 장애물을 감지하거나 YOLOv8이 `person`을 검출하면 `/cmd_vel`을 0으로 만들어 정지합니다.

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

## Gazebo 테스트 (Waffle Pi)

각 명령은 별도의 터미널에서 실행합니다. 런처가 ROS 2 Humble, 워크스페이스, `ROS_DOMAIN_ID=64`, `TURTLEBOT3_MODEL=waffle_pi`를 설정합니다. 코드와 설정을 변경한 뒤에는 한 번 빌드합니다.

```bash
cd /home/kiki/turtlebot3
source /opt/ros/humble/setup.bash
colcon build --packages-select my_car_astar --symlink-install
```

터미널 1에서 Gazebo 하우스 월드와 Waffle Pi를 실행합니다.

```bash
cd /home/kiki/turtlebot3
./scripts/astar_gazebo_64
```

터미널 2에서 저장 지도를 불러오고 AMCL 로컬라이제이션만 실행합니다. A*가 직접 `/cmd_vel`을 제어하므로 Nav2 주행 컨트롤러는 함께 실행하지 않습니다. 다른 지도를 쓸 때는 `MAP_FILE`을 절대 경로로 지정합니다.

```bash
cd /home/kiki/turtlebot3
./scripts/astar_localization_64
# 또는 MAP_FILE=/절대경로/map.yaml ./scripts/astar_localization_64
```

터미널 3에서 A* 주행과 Gazebo 카메라 YOLO 비상 정지를 실행합니다.

```bash
cd /home/kiki/turtlebot3
./scripts/astar_yolo_64
```

터미널 4에서 기존 컨트롤러를 실행합니다. 컨트롤러의 지도 패널이 실시간 `/map`을 표시하며, RViz 없이 초기 위치와 A* 목표를 지정할 수 있습니다.

```bash
cd /home/kiki/turtlebot3
./scripts/control_64
```

컨트롤러 지도 패널에서 `초기 위치 지정`을 선택하고 지도 위를 클릭한 채 드래그해 로봇 방향을 설정합니다. AMCL 위치가 표시되면 `목표 / 경유점 추가`를 선택하고 목표를 클릭한 채 드래그한 다음 `Goal 전송`을 누릅니다. 이 버튼은 A*가 구독하는 `/goal_pose`를 발행합니다. A* 터미널에 `A* 경로 생성 완료`가 출력되면 로봇이 경로를 추종합니다.

이 테스트에서는 `경로점 순회 시작`을 누르지 마세요. 이 기능은 Nav2 `FollowWaypoints` 액션을 사용하며, A* 테스트 경로는 `Goal 전송`으로 보내야 합니다. 컨트롤러의 수동 조종 버튼/화살표 키도 A* 주행 중에는 사용하지 마세요.

지도에 들어 있는 우편함, 벽 등은 A*가 차체 여유를 두고 우회합니다. 지도상 자유 공간, 미지 공간 또는 지도 밖에서 새로 감지된 물체는 `/scan`이 전방 0.45 m 이내에서 검출하면 정지합니다. 물체가 감지 영역을 벗어나면 기존 경로 추종을 재개합니다. 이는 동적 재계획이 아니라 충돌 방지 정지입니다.

YOLO가 Waffle Pi 영상의 `person`을 검출해도 정지합니다.

```text
person 검출: 비상정지 cmd_vel을 발행합니다.
```

사람이 화면에서 사라지면 기존 A* 경로 추종을 계속합니다. 영상 입력은 `/camera/image_raw`이며, 검출 창 표시는 GUI 환경에서만 사용할 수 있습니다. `/scan` 수신이 1초 넘게 끊겨도 fail-safe로 정지합니다.

## 파라미터

설정 파일은 `config/astar_drive.yaml`입니다.

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `lookahead_distance` | `0.5` | Pure Pursuit 전방 주시 거리(m) |
| `linear_velocity` | `0.1` | 선속도(m/s) |
| `goal_tolerance` | `0.2` | 목표 도착 허용 거리(m) |
| `max_angular_velocity` | `1.0` | 최대 각속도(rad/s) |
| `occupied_threshold` | `50` | A*가 장애물로 처리할 지도 값 |
| `robot_radius` | `0.18` | A* 점유 셀 확장에 사용할 로봇 반경(m) |
| `obstacle_stop_distance` | `0.45` | 지도상 미점유 전방 장애물 정지 거리(m) |
| `obstacle_front_angle` | `0.35` | 전방 감지 반각(rad) |
| `scan_timeout` | `1.0` | LiDAR 스캔 시간 초과 시 정지 시간(s) |
| `image_topic` | `/camera/image_raw` | Gazebo Waffle Pi 카메라 이미지 토픽 |
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
