import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class LandmarkState:
    """Track current and previous landmarks for simple motion prediction."""

    current: np.ndarray
    previous: np.ndarray


class HandTracker:
    def __init__(
        self,
        camera_index: int = 0,
        max_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        focal_length_px: float = 950.0,
    ) -> None:
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_index}")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.focal_length_px = focal_length_px
        self.landmark_state: Dict[int, LandmarkState] = {}
        self.fps_history: Deque[float] = deque(maxlen=20)
        self.last_time = time.perf_counter()

        # 平均成人食指 MCP 到小指 MCP 的实际宽度约 8cm，可用于粗略测距
        self.assumed_hand_width_cm = 8.0

    @staticmethod
    def _landmarks_to_numpy(hand_landmarks, frame_shape) -> np.ndarray:
        h, w, _ = frame_shape
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h, lm.z * w])
        return np.array(pts, dtype=np.float32)

    def _estimate_distance_cm(self, points: np.ndarray) -> float:
        # 5: index_mcp, 17: pinky_mcp
        hand_width_px = np.linalg.norm(points[5, :2] - points[17, :2])
        if hand_width_px < 1e-6:
            return 0.0
        return (self.assumed_hand_width_cm * self.focal_length_px) / hand_width_px

    @staticmethod
    def _predict_next(points_now: np.ndarray, points_prev: np.ndarray) -> np.ndarray:
        # 常速度模型: p_next = p_now + (p_now - p_prev)
        velocity = points_now - points_prev
        return points_now + velocity

    def _draw_predicted_skeleton(self, frame: np.ndarray, points: np.ndarray) -> None:
        connections = self.mp_hands.HAND_CONNECTIONS
        for start_idx, end_idx in connections:
            x1, y1 = points[start_idx, :2].astype(int)
            x2, y2 = points[end_idx, :2].astype(int)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 180, 0), 1, cv2.LINE_AA)

        for point in points:
            x, y = point[:2].astype(int)
            cv2.circle(frame, (x, y), 3, (255, 180, 0), -1, cv2.LINE_AA)

    def _update_fps(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    def run(self) -> None:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("读取摄像头帧失败，程序结束。")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                    points_now = self._landmarks_to_numpy(hand_lms, frame.shape)
                    state = self.landmark_state.get(hand_idx)
                    if state is None:
                        state = LandmarkState(current=points_now.copy(), previous=points_now.copy())
                        self.landmark_state[hand_idx] = state
                    else:
                        state.previous = state.current
                        state.current = points_now.copy()

                    dist_cm = self._estimate_distance_cm(state.current)
                    cv2.putText(
                        frame,
                        f"Distance: {dist_cm:.1f} cm",
                        (20, 40 + hand_idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (30, 255, 30),
                        2,
                        cv2.LINE_AA,
                    )

                    predicted = self._predict_next(state.current, state.previous)
                    self._draw_predicted_skeleton(frame, predicted)
                    cv2.putText(
                        frame,
                        "Predicted skeleton (orange)",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 180, 0),
                        2,
                        cv2.LINE_AA,
                    )

            fps = self._update_fps()
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-time Hand Skeleton Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时手掌骨架识别 + 距离估算 + 骨架预测")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--max-hands", type=int, default=1, help="最多跟踪手数")
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=950.0,
        help="相机焦距(像素)，影响距离估算精度",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = HandTracker(
        camera_index=args.camera,
        max_hands=args.max_hands,
        focal_length_px=args.focal_length_px,
    )
    tracker.run()


if __name__ == "__main__":
    main()
