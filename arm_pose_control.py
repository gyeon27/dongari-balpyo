"""MediaPipe Pose로 로봇 팔의 3자유도 목표 각도를 계산한다.

출력 각도
  shoulder_flexion : 팔을 앞(+)/뒤(-)로 드는 각도
  shoulder_abduction: 팔을 몸 옆으로 벌리는 각도(안쪽은 음수)
  elbow_flexion    : 팔꿈치 굽힘. 완전히 편 상태가 0도

실행 예시:
  python arm_pose_control.py --camera 1 --side right

Iriun Webcam의 카메라 번호는 PC 환경마다 다르므로 0, 1, 2를 차례로
시험한다. 화면 표시만 거울처럼 반전하며 각도 계산에는 원본 영상을 쓴다.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")

# MediaPipe Pose의 고정 랜드마크 번호
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

# 선택한 팔만 화면에 표시한다.
ARM_CONNECTIONS = {
    "left": ((LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST)),
    "right": ((RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST)),
}


@dataclass
class ArmAngles:
    shoulder_flexion: float
    shoulder_abduction: float
    elbow_flexion: float


class RobustAngleFilter:
    """중앙값 제거 + 각속도 제한 + EMA를 결합한 로봇 제어용 필터."""

    def __init__(
        self,
        alpha: float = 0.25,
        median_window: int = 5,
        max_speed: float = 240.0,
    ) -> None:
        self.alpha = alpha
        self.samples: deque[np.ndarray] = deque(maxlen=median_window)
        self.max_speed = max_speed
        self.value: np.ndarray | None = None
        self.last_time: float | None = None
        self.was_limited = False

    def update(self, angles: ArmAngles, now: float) -> ArmAngles:
        current = np.array(
            [angles.shoulder_flexion, angles.shoulder_abduction, angles.elbow_flexion],
            dtype=float,
        )
        if not np.all(np.isfinite(current)):
            raise ValueError("각도 계산 결과가 유효하지 않습니다.")

        self.samples.append(current)
        # 순간적으로 한 프레임만 튀는 값은 중앙값에서 제거된다.
        median = np.median(np.stack(self.samples), axis=0)

        if self.value is None:
            self.value = median
            self.was_limited = False
        else:
            dt = max(now - (self.last_time or now), 1.0 / 60.0)
            # 프레임 간 허용 변화량. 4도 여유를 두어 작은 움직임은 지연시키지 않는다.
            max_delta = self.max_speed * min(dt, 0.1) + 4.0
            delta = median - self.value
            limited_delta = np.clip(delta, -max_delta, max_delta)
            self.was_limited = bool(np.any(np.abs(delta) > max_delta))
            safe_value = self.value + limited_delta
            self.value = self.alpha * safe_value + (1.0 - self.alpha) * self.value

        self.last_time = now
        return ArmAngles(*self.value.tolist())


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("길이가 0인 벡터는 정규화할 수 없습니다.")
    return vector / norm


def angle_between(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    cosine = float(np.dot(unit(vector_a), unit(vector_b)))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def point(landmarks, index: int) -> np.ndarray:
    item = landmarks[index]
    return np.array([item.x, item.y, item.z], dtype=float)


def calculate_arm_angles(world_landmarks, side: str) -> ArmAngles:
    """미터 단위 world landmark로 몸통 기준 관절각을 계산한다."""
    lm = world_landmarks

    left_shoulder = point(lm, LEFT_SHOULDER)
    right_shoulder = point(lm, RIGHT_SHOULDER)
    left_hip = point(lm, LEFT_HIP)
    right_hip = point(lm, RIGHT_HIP)

    if side == "right":
        shoulder = right_shoulder
        elbow = point(lm, RIGHT_ELBOW)
        wrist = point(lm, RIGHT_WRIST)
        outward_sign = -1.0
    else:
        shoulder = left_shoulder
        elbow = point(lm, LEFT_ELBOW)
        wrist = point(lm, LEFT_WRIST)
        outward_sign = 1.0

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    # 몸 기준축: 왼쪽, 위, 앞. 카메라가 약간 기울어도 몸을 따라 회전한다.
    body_left = unit(left_shoulder - right_shoulder)
    body_up_raw = unit(shoulder_mid - hip_mid)
    body_forward = unit(np.cross(body_left, body_up_raw))
    body_up = unit(np.cross(body_forward, body_left))

    upper_arm = unit(elbow - shoulder)
    forearm = unit(wrist - elbow)

    down = -float(np.dot(upper_arm, body_up))
    forward = float(np.dot(upper_arm, body_forward))
    outward = outward_sign * float(np.dot(upper_arm, body_left))

    # 각 평면으로 투영한 signed angle. 팔을 자연스럽게 내리면 두 값 모두 0도다.
    shoulder_flexion = math.degrees(math.atan2(forward, down))
    shoulder_abduction = math.degrees(math.atan2(outward, down))

    # 두 벡터가 같은 방향(팔을 곧게 폄)이면 굽힘 0도다.
    elbow_flexion = angle_between(upper_arm, forearm)

    return ArmAngles(shoulder_flexion, shoulder_abduction, elbow_flexion)


def tracked_well(pose_landmarks, side: str, threshold: float) -> bool:
    required = (
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        if side == "right"
        else (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    )
    return all(
        (pose_landmarks[item].visibility or 0.0) >= threshold
        and (pose_landmarks[item].presence or 0.0) >= threshold
        for item in required
    )


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print("Pose Landmarker 모델을 처음 한 번 다운로드합니다...")
    try:
        urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise SystemExit(
            f"모델 다운로드 실패: {exc}\n"
            f"브라우저에서 다음 주소를 내려받아 {MODEL_PATH.name}으로 저장하세요:\n{MODEL_URL}"
        ) from exc
    return MODEL_PATH


def draw_arm(frame: np.ndarray, landmarks, side: str) -> None:
    height, width = frame.shape[:2]
    points = [(int(item.x * width), int(item.y * height)) for item in landmarks]
    connections = ARM_CONNECTIONS[side]
    for start, end in connections:
        cv2.line(frame, points[start], points[end], (80, 220, 80), 2, cv2.LINE_AA)
    for index in {item for connection in connections for item in connection}:
        cv2.circle(frame, points[index], 4, (30, 80, 255), -1, cv2.LINE_AA)


def choose_side(side_argument: str | None) -> str:
    if side_argument is not None:
        return side_argument

    while True:
        answer = input("추적할 팔을 선택하세요 [R: 오른팔 / L: 왼팔]: ").strip().lower()
        if answer in ("r", "right", "오른팔", "오른쪽"):
            return "right"
        if answer in ("l", "left", "왼팔", "왼쪽"):
            return "left"
        print("R 또는 L을 입력해 주세요.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe 기반 3자유도 로봇 팔 제어값 추정")
    parser.add_argument("--camera", type=int, default=0, help="Iriun 카메라 번호 (기본값: 0)")
    parser.add_argument(
        "--side",
        choices=("right", "left"),
        default=None,
        help="추적할 팔. 생략하면 실행할 때 선택",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--alpha", type=float, default=0.25, help="EMA 계수. 작을수록 부드러움")
    parser.add_argument("--visibility", type=float, default=0.7, help="최소 랜드마크 신뢰도")
    parser.add_argument("--median-window", type=int, default=5, help="중앙값 필터 프레임 수")
    parser.add_argument("--max-speed", type=float, default=240.0, help="허용 최대 각속도(deg/s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_side = choose_side(args.side)
    if not 0.0 < args.alpha <= 1.0:
        raise SystemExit("--alpha는 0보다 크고 1 이하여야 합니다.")
    if args.median_window < 1 or args.median_window % 2 == 0:
        raise SystemExit("--median-window는 1 이상의 홀수여야 합니다.")
    if args.max_speed <= 0:
        raise SystemExit("--max-speed는 0보다 커야 합니다.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}를 열 수 없습니다. --camera 번호를 바꿔 보세요.")

    window_name = "Robot Arm Control - 3DoF Pose"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    if not hasattr(mp, "tasks"):
        raise SystemExit("현재 mediapipe 설치본에 Tasks API가 없습니다: python -m pip install -U mediapipe")

    vision = mp.tasks.vision
    # MediaPipe의 Windows 네이티브 코드가 한글 경로를 열지 못하는 경우가 있다.
    # Python에서 파일을 읽어 바이트로 넘기면 경로 인코딩 문제를 피할 수 있다.
    model_data = ensure_model().read_bytes()
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    angle_filter = RobustAngleFilter(args.alpha, args.median_window, args.max_speed)
    last_print_time = 0.0
    start_time = time.monotonic()
    last_timestamp_ms = -1

    try:
        with vision.PoseLandmarker.create_from_options(options) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    break

                # 셀카/거울 모드: 표시뿐 아니라 이 반전된 프레임 자체를 인식한다.
                frame = cv2.flip(frame, 1)
                # 입력을 반전하면 MediaPipe의 좌우 랜드마크 번호도 서로 바뀐다.
                model_side = "left" if selected_side == "right" else "right"

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(int((time.monotonic() - start_time) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                results = pose.detect_for_video(mp_image, timestamp_ms)

                status = "Pose not detected"
                angles: ArmAngles | None = None
                now = time.monotonic()
                if (
                    results.pose_landmarks
                    and results.pose_world_landmarks
                    and tracked_well(results.pose_landmarks[0], model_side, args.visibility)
                ):
                    try:
                        angles = angle_filter.update(
                            calculate_arm_angles(results.pose_world_landmarks[0], model_side),
                            now,
                        )
                        status = (
                            f"{selected_side.upper()} | Flex {angles.shoulder_flexion:6.1f}  "
                            f"Abd {angles.shoulder_abduction:6.1f}  "
                            f"Elbow {angles.elbow_flexion:6.1f}"
                        )
                        if angle_filter.was_limited:
                            status += "  [stabilizing]"
                    except ValueError:
                        status = "Invalid landmark geometry"

                    draw_arm(frame, results.pose_landmarks[0], model_side)

                # 시리얼 전송을 붙일 때 이 출력부를 "F,A,E\n" 전송으로 교체하면 된다.
                if angles is not None and now - last_print_time >= 0.1:
                    print(
                        f"{angles.shoulder_flexion:.1f},"
                        f"{angles.shoulder_abduction:.1f},"
                        f"{angles.elbow_flexion:.1f}"
                    )
                    last_print_time = now

                display = frame
                cv2.putText(
                    display,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if angles is not None else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "R: right arm  L: left arm  Q: quit",
                    (20, 75),
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
                if key == ord("r") and selected_side != "right":
                    selected_side = "right"
                    angle_filter = RobustAngleFilter(args.alpha, args.median_window, args.max_speed)
                    print("오른팔 추적으로 변경했습니다.")
                elif key == ord("l") and selected_side != "left":
                    selected_side = "left"
                    angle_filter = RobustAngleFilter(args.alpha, args.median_window, args.max_speed)
                    print("왼팔 추적으로 변경했습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
