"""MediaPipe Pose + Hands로 양팔 3자유도 각도와 양손 그립(0~255)을 동시에 추출한다.

arm_pose_control.py의 필터/각도 계산 로직을 그대로 재사용하고, 여기에 양팔 동시
추적과 손 그립 인식을 더한 것이 이 스크립트다.

출력 각도/값 (0.1초마다 한 줄, CSV)
  L_flex, L_abd, L_elbow, L_grip, R_flex, R_abd, R_elbow, R_grip
  - flex/abd/elbow: arm_pose_control.py와 동일한 정의(도)
  - grip: 0(편 상태) ~ 255(완전히 쥔 상태)

실행 예시:
  python robot_control.py --camera 1

Iriun Webcam의 카메라 번호는 PC 환경마다 다르므로 0, 1, 2를 차례로 시험한다.
화면 표시는 거울처럼 반전하며, 좌우 라벨(팔의 model_side)을 반대로 읽어
반전을 보정하므로 계산 결과는 실제 좌우와 일치한다.

손은 화면 전체에서 따로 찾지 않는다. 팔이 추적된 손목 주변만 잘라서 그
크롭 안에서만 손을 찾으므로, 언제나 해당 팔에 달린 손만 인식되고 팔과
손이 서로 다른 대상으로 따로 인식되는 일이 없다.

최초 실행 시 pose_landmarker_full.task 외에 hand_landmarker.task도 자동으로
내려받는다(약 8MB).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from arm_pose_control import (
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    ArmAngles,
    RobustAngleFilter,
    angle_between,
    calculate_arm_angles,
    draw_arm,
    ensure_model,
    point,
    tracked_well,
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# MediaPipe Hands의 고정 랜드마크 번호. (MCP, PIP, TIP) 세 점으로 굽힘각을 잰다.
# 엄지는 다른 손가락과 관절 구조/각도 범위가 달라 평균에서 제외한다.
FINGER_JOINTS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

# 프레임을 거울처럼 반전해서 인식하므로, 실제 좌우는 모델이 읽는 좌우의 반대다.
SIDE_OPPOSITE = {"left": "right", "right": "left"}

# MediaPipe Hands의 21개 랜드마크를 잇는 고정 연결선.
# 이 mediapipe 설치본은 Tasks API만 포함하고 mp.solutions가 없어 직접 정의한다.
HAND_CONNECTIONS = (
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


class RobustValueFilter:
    """RobustAngleFilter와 동일한 중앙값 제거 + 속도 제한 + EMA를 스칼라 값에 적용한다."""

    def __init__(
        self,
        alpha: float = 0.25,
        median_window: int = 5,
        max_speed: float = 600.0,
    ) -> None:
        self.alpha = alpha
        self.samples: deque[float] = deque(maxlen=median_window)
        self.max_speed = max_speed
        self.value: float | None = None
        self.last_time: float | None = None

    def update(self, raw: float, now: float) -> float:
        if not math.isfinite(raw):
            raise ValueError("그립 값이 유효하지 않습니다.")

        self.samples.append(raw)
        median = float(np.median(self.samples))

        if self.value is None:
            self.value = median
        else:
            dt = max(now - (self.last_time or now), 1.0 / 60.0)
            max_delta = self.max_speed * min(dt, 0.1) + 4.0
            delta = median - self.value
            limited_delta = max(-max_delta, min(max_delta, delta))
            safe_value = self.value + limited_delta
            self.value = self.alpha * safe_value + (1.0 - self.alpha) * self.value

        self.last_time = now
        return self.value


def finger_curl_angle(world_landmarks, mcp: int, pip: int, tip: int) -> float:
    proximal = point(world_landmarks, pip) - point(world_landmarks, mcp)
    distal = point(world_landmarks, tip) - point(world_landmarks, pip)
    return angle_between(proximal, distal)


def calculate_grip(hand_world_landmarks) -> float:
    """4손가락 굽힘각 평균. 편 상태 0도에 가깝고, 주먹을 쥘수록 커진다."""
    angles = [
        finger_curl_angle(hand_world_landmarks, mcp, pip, tip)
        for mcp, pip, tip in FINGER_JOINTS.values()
    ]
    return float(np.mean(angles))


def grip_angle_to_byte(angle: float, open_angle: float, closed_angle: float) -> float:
    ratio = (angle - open_angle) / (closed_angle - open_angle)
    return float(np.clip(ratio, 0.0, 1.0) * 255.0)


def landmark_px(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def hand_crop_box(
    wrist_px: tuple[int, int], elbow_px: tuple[int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """손목 주변에 손이 통째로 들어올 만큼의 정사각형 크롭 영역을 구한다.

    팔꿈치-손목 픽셀 거리를 척도로 써서 카메라와의 거리가 달라져도 손 크기에
    맞춰 크롭 크기가 함께 커지거나 작아진다. 손은 손목에서 팔꿈치 반대쪽으로
    더 뻗어 있으므로, 중심을 손목보다 살짝 바깥쪽으로 밀어 손 전체가 잘리지
    않게 한다.
    """
    dx, dy = wrist_px[0] - elbow_px[0], wrist_px[1] - elbow_px[1]
    forearm_len = math.hypot(dx, dy)
    if forearm_len < 5:
        return None

    half = int(np.clip(forearm_len * 1.3, 70, 260))
    ux, uy = dx / forearm_len, dy / forearm_len
    cx = int(wrist_px[0] + ux * half * 0.5)
    cy = int(wrist_px[1] + uy * half * 0.5)

    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, width), min(cy + half, height)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    return x0, y0, x1, y1


def ensure_hand_model() -> Path:
    if HAND_MODEL_PATH.exists():
        return HAND_MODEL_PATH
    print("Hand Landmarker 모델을 처음 한 번 다운로드합니다...")
    try:
        urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as exc:
        raise SystemExit(
            f"모델 다운로드 실패: {exc}\n"
            f"브라우저에서 다음 주소를 내려받아 {HAND_MODEL_PATH.name}으로 저장하세요:\n{HAND_MODEL_URL}"
        ) from exc
    return HAND_MODEL_PATH


def draw_hand(frame: np.ndarray, landmarks, origin: tuple[int, int], crop_size: tuple[int, int]) -> None:
    x0, y0 = origin
    crop_w, crop_h = crop_size
    points = [(int(x0 + item.x * crop_w), int(y0 + item.y * crop_h)) for item in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (220, 180, 80), 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 3, (30, 80, 255), -1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MediaPipe 기반 양팔 3자유도 + 양손 그립(0~255) 로봇 제어값 추정"
    )
    parser.add_argument("--camera", type=int, default=0, help="Iriun 카메라 번호 (기본값: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--alpha", type=float, default=0.25, help="EMA 계수. 작을수록 부드러움")
    parser.add_argument("--visibility", type=float, default=0.7, help="팔 랜드마크 최소 신뢰도")
    parser.add_argument("--median-window", type=int, default=5, help="중앙값 필터 프레임 수")
    parser.add_argument("--max-speed", type=float, default=240.0, help="팔 각도 허용 최대 각속도(deg/s)")
    parser.add_argument("--grip-open-angle", type=float, default=15.0, help="편 손의 손가락 굽힘각 기준(도)")
    parser.add_argument("--grip-closed-angle", type=float, default=120.0, help="쥔 손의 손가락 굽힘각 기준(도)")
    parser.add_argument(
        "--grip-max-speed", type=float, default=600.0, help="그립 값(0~255) 허용 최대 변화 속도(단위/s)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha <= 1.0:
        raise SystemExit("--alpha는 0보다 크고 1 이하여야 합니다.")
    if args.median_window < 1 or args.median_window % 2 == 0:
        raise SystemExit("--median-window는 1 이상의 홀수여야 합니다.")
    if args.max_speed <= 0:
        raise SystemExit("--max-speed는 0보다 커야 합니다.")
    if args.grip_max_speed <= 0:
        raise SystemExit("--grip-max-speed는 0보다 커야 합니다.")
    if args.grip_closed_angle <= args.grip_open_angle:
        raise SystemExit("--grip-closed-angle은 --grip-open-angle보다 커야 합니다.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}를 열 수 없습니다. --camera 번호를 바꿔 보세요.")

    window_name = "Robot Control - Dual Arm + Grip"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    if not hasattr(mp, "tasks"):
        raise SystemExit("현재 mediapipe 설치본에 Tasks API가 없습니다: python -m pip install -U mediapipe")

    vision = mp.tasks.vision
    pose_model_data = ensure_model().read_bytes()
    hand_model_data = ensure_hand_model().read_bytes()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=pose_model_data),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    def build_hand_options() -> vision.HandLandmarkerOptions:
        # 크롭 안에는 그 팔의 손 하나만 들어오므로 num_hands=1.
        return vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=hand_model_data),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    angle_filters = {
        "left": RobustAngleFilter(args.alpha, args.median_window, args.max_speed),
        "right": RobustAngleFilter(args.alpha, args.median_window, args.max_speed),
    }
    grip_filters = {
        "left": RobustValueFilter(args.alpha, args.median_window, args.grip_max_speed),
        "right": RobustValueFilter(args.alpha, args.median_window, args.grip_max_speed),
    }

    # 추적이 잠시 끊겨도 마지막 값을 유지해 로봇이 갑자기 원점으로 튀지 않게 한다.
    last_arm_angles = {"left": ArmAngles(0.0, 0.0, 0.0), "right": ArmAngles(0.0, 0.0, 0.0)}
    last_grip = {"left": 0.0, "right": 0.0}

    last_print_time = 0.0
    start_time = time.monotonic()
    last_timestamp_ms = -1

    try:
        with vision.PoseLandmarker.create_from_options(pose_options) as pose, vision.HandLandmarker.create_from_options(
            build_hand_options()
        ) as hand_left, vision.HandLandmarker.create_from_options(build_hand_options()) as hand_right:
            hand_landmarkers = {"left": hand_left, "right": hand_right}
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(int((time.monotonic() - start_time) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                pose_results = pose.detect_for_video(mp_image, timestamp_ms)
                now = time.monotonic()

                # --- 팔: 어깨/팔꿈치 각도. 팔이 추적된 경우에만 그 손목 주변을
                # 잘라 손을 찾으므로, 손은 항상 그 팔에 달린 손만 인식된다. ---
                arm_tracked = {"left": False, "right": False}
                grip_tracked = {"left": False, "right": False}
                if pose_results.pose_landmarks and pose_results.pose_world_landmarks:
                    pose_landmarks = pose_results.pose_landmarks[0]
                    pose_world_landmarks = pose_results.pose_world_landmarks[0]
                    height, width = frame.shape[:2]
                    for real_side, model_side in SIDE_OPPOSITE.items():
                        if not tracked_well(pose_landmarks, model_side, args.visibility):
                            continue
                        try:
                            raw_angles = calculate_arm_angles(pose_world_landmarks, model_side)
                            last_arm_angles[real_side] = angle_filters[real_side].update(raw_angles, now)
                            arm_tracked[real_side] = True
                            draw_arm(frame, pose_landmarks, model_side)
                        except ValueError:
                            continue

                        wrist_index = RIGHT_WRIST if model_side == "right" else LEFT_WRIST
                        elbow_index = RIGHT_ELBOW if model_side == "right" else LEFT_ELBOW
                        wrist_px = landmark_px(pose_landmarks[wrist_index], width, height)
                        elbow_px = landmark_px(pose_landmarks[elbow_index], width, height)
                        box = hand_crop_box(wrist_px, elbow_px, width, height)
                        if box is None:
                            continue
                        x0, y0, x1, y1 = box
                        crop_rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                        crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                        hand_result = hand_landmarkers[real_side].detect_for_video(crop_image, timestamp_ms)
                        if not (hand_result.hand_landmarks and hand_result.hand_world_landmarks):
                            continue
                        try:
                            raw_angle = calculate_grip(hand_result.hand_world_landmarks[0])
                            byte_value = grip_angle_to_byte(raw_angle, args.grip_open_angle, args.grip_closed_angle)
                            last_grip[real_side] = grip_filters[real_side].update(byte_value, now)
                            grip_tracked[real_side] = True
                            draw_hand(frame, hand_result.hand_landmarks[0], (x0, y0), (x1 - x0, y1 - y0))
                        except ValueError:
                            pass

                # 시리얼 전송을 붙일 때 이 출력부를 로봇으로의 전송으로 교체하면 된다.
                if now - last_print_time >= 0.1:
                    left = last_arm_angles["left"]
                    right = last_arm_angles["right"]
                    print(
                        f"{left.shoulder_flexion:.1f},{left.shoulder_abduction:.1f},{left.elbow_flexion:.1f},"
                        f"{last_grip['left']:.0f},"
                        f"{right.shoulder_flexion:.1f},{right.shoulder_abduction:.1f},{right.elbow_flexion:.1f},"
                        f"{last_grip['right']:.0f}"
                    )
                    last_print_time = now

                display = frame
                left = last_arm_angles["left"]
                right = last_arm_angles["right"]
                lines = [
                    (f"L-Arm  Flex {left.shoulder_flexion:6.1f}  Abd {left.shoulder_abduction:6.1f}  "
                     f"Elbow {left.elbow_flexion:6.1f}", arm_tracked["left"]),
                    (f"R-Arm  Flex {right.shoulder_flexion:6.1f}  Abd {right.shoulder_abduction:6.1f}  "
                     f"Elbow {right.elbow_flexion:6.1f}", arm_tracked["right"]),
                    (f"L-Hand Grip {last_grip['left']:5.0f} / 255", grip_tracked["left"]),
                    (f"R-Hand Grip {last_grip['right']:5.0f} / 255", grip_tracked["right"]),
                ]
                for row, (text, is_tracked) in enumerate(lines):
                    cv2.putText(
                        display,
                        text,
                        (20, 40 + row * 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if is_tracked else (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    display,
                    "Q: quit",
                    (20, 40 + len(lines) * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
