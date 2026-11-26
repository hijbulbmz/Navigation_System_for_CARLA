"""Navigation HUD overlay for CARLA autonomous driving.

Displays live navigation state, speed targets, remaining distance, route progress,
and destination arrival confirmation on the OpenCV display frame.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from navigation.arrival_manager import NavigationState

# Color definitions (BGR for OpenCV)
COLOR_NAVIGATING = (255, 220, 0)      # Bright Cyan / Azure
COLOR_APPROACHING = (0, 215, 255)     # Amber / Yellow
COLOR_STOPPING = (0, 140, 255)        # Orange
COLOR_ARRIVED = (0, 255, 50)          # Bright Green
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_DIM = (180, 180, 180)
COLOR_BG_BOX = (20, 20, 20)           # Dark charcoal background box


def get_state_color(state: NavigationState) -> Tuple[int, int, int]:
    """Return BGR color corresponding to the navigation state."""
    if state == NavigationState.NAVIGATING:
        return COLOR_NAVIGATING
    if state == NavigationState.APPROACHING:
        return COLOR_APPROACHING
    if state == NavigationState.STOPPING:
        return COLOR_STOPPING
    if state == NavigationState.ARRIVED:
        return COLOR_ARRIVED
    return COLOR_TEXT_WHITE


def draw_navigation_hud(
    frame: "np.ndarray",
    state: NavigationState,
    current_speed_kmh: float,
    target_speed_kmh: float,
    dist_to_destination_m: float,
    total_distance_m: float,
    progress_pct: float,
    is_arrived: bool = False,
    top_left: Tuple[int, int] = (8, 55),
    alpha: float = 0.65,
) -> "np.ndarray":
    """Renders a clean, compact navigation HUD panel on the display frame.

    Args:
        frame: BGR image from camera feed / OpenCV display frame.
        state: Current NavigationState enum value.
        current_speed_kmh: Vehicle actual speed in km/h.
        target_speed_kmh: Current active target speed command in km/h.
        dist_to_destination_m: Remaining Euclidean distance to destination in meters.
        total_distance_m: Planned route total distance in meters.
        progress_pct: Route progress percentage (0.0 to 100.0).
        is_arrived: Boolean indicating if arrival confirmation has locked.
        top_left: (x, y) coordinates for HUD panel origin.
        alpha: Opacity factor for the background box.

    Returns:
        Frame with HUD rendered.
    """
    if frame is None or cv2 is None or np is None:
        return frame

    x0, y0 = top_left
    box_w = 260
    box_h = 76 if not is_arrived else 96

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(x0, frame_w - box_w))
    y1 = max(0, min(y0, frame_h - box_h))
    x2 = min(frame_w, x1 + box_w)
    y2 = min(frame_h, y1 + box_h)

    # Semi-transparent background box
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        bg = np.full_like(roi, COLOR_BG_BOX, dtype=np.uint8)
        cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0.0, roi)
        frame[y1:y2, x1:x2] = roi

    # Accent left border colored by state
    state_color = get_state_color(state)
    cv2.line(frame, (x1, y1), (x1, y2), state_color, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)

    # Line 1: State header with colored pill indicator
    cv2.circle(frame, (x1 + 14, y1 + 16), 5, state_color, -1, cv2.LINE_AA)
    state_text = f"NAV: {state.value}"
    cv2.putText(
        frame,
        state_text,
        (x1 + 26, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        state_color,
        1,
        cv2.LINE_AA,
    )

    # Line 2: Speed and Target Speed
    speed_text = f"Speed: {current_speed_kmh:4.1f} / {target_speed_kmh:4.1f} km/h"
    cv2.putText(
        frame,
        speed_text,
        (x1 + 12, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )

    # Line 3: Distance to destination and progress
    dist_text = f"To Dest: {dist_to_destination_m:5.1f}m ({progress_pct:4.1f}%)"
    cv2.putText(
        frame,
        dist_text,
        (x1 + 12, y1 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_DIM,
        1,
        cv2.LINE_AA,
    )

    # Line 4 (if arrived): Big banner
    if is_arrived:
        arrived_box_y = y1 + 68
        cv2.rectangle(
            frame,
            (x1 + 6, arrived_box_y),
            (x2 - 6, y2 - 4),
            (0, 80, 0),
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (x1 + 6, arrived_box_y),
            (x2 - 6, y2 - 4),
            COLOR_ARRIVED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "* DESTINATION REACHED *",
            (x1 + 16, arrived_box_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_ARRIVED,
            1,
            cv2.LINE_AA,
        )

    return frame

