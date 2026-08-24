"""Non-blocking manipulation base-motion ownership for candidate workers."""

from __future__ import annotations

import os
import time

from rclpy.callback_groups import ReentrantCallbackGroup

from supermarket_interfaces.msg import MotionMode
from supermarket_interfaces.srv import SetFootprintProfile, SetMotionMode


class ManipulationMotionLease:
    """Acquire STOP -> footprint -> MANIPULATION without blocking callbacks."""

    def __init__(self, node) -> None:
        self.node = node
        self.owner = f"worker-{os.getpid()}-manipulation"
        self.mode = MotionMode.STOP
        self.mode_owner = ""
        self.target_profile = ""
        self.state = "idle"
        self.future = None
        self.deadline = 0.0
        group = ReentrantCallbackGroup()
        self.mode_client = node.create_client(
            SetMotionMode, "/motion/set_mode", callback_group=group
        )
        self.footprint_client = node.create_client(
            SetFootprintProfile,
            "/motion/set_footprint_profile",
            callback_group=group,
        )
        node.create_subscription(
            MotionMode, "/motion/mode", self._mode_cb, 10, callback_group=group
        )

    def _mode_cb(self, message: MotionMode) -> None:
        self.mode = int(message.mode)
        self.mode_owner = str(message.owner)

    def ready(self, profile: str) -> bool:
        profile = str(profile)
        if (
            self.mode == MotionMode.MANIPULATION
            and self.mode_owner == self.owner
            and self.target_profile == profile
            and self.state == "ready"
        ):
            return True
        if profile != self.target_profile:
            self.target_profile = profile
            self.state = "request_stop"
            self.future = None
        self._advance()
        return (
            self.mode == MotionMode.MANIPULATION
            and self.mode_owner == self.owner
            and self.state == "ready"
        )

    def _advance(self) -> None:
        now = time.monotonic()
        if self.future is not None:
            if not self.future.done():
                if now >= self.deadline:
                    self.node.get_logger().error(
                        f"motion lease timed out in state={self.state}")
                    self.future = None
                    self.state = "failed"
                return
            try:
                response = self.future.result()
            except Exception as exc:
                self.future = None
                self._fail(f"motion lease service transport failed: {exc}")
                return
            self.future = None
            if response is None:
                self.state = "failed"
                return
            if self.state == "waiting_stop":
                if not response.accepted:
                    self._fail(response.detail)
                    return
                self.state = "request_footprint"
            elif self.state == "waiting_footprint":
                if not response.applied:
                    self._fail(response.detail)
                    return
                self.state = "request_manipulation"
            elif self.state == "waiting_manipulation":
                if not response.accepted:
                    self._fail(response.detail)
                    return
                self.state = "ready"
                self.node.get_logger().info(
                    f"manipulation motion lease acquired profile={self.target_profile}"
                )
                return

        if self.state in {"idle", "failed"}:
            self.state = "request_stop"
        if self.state == "request_stop":
            if not self.mode_client.service_is_ready():
                return
            request = SetMotionMode.Request()
            request.mode = MotionMode.STOP
            request.owner = self.owner
            request.reset_emergency_stop = False
            self.future = self.mode_client.call_async(request)
            self.state = "waiting_stop"
            self.deadline = now + 1.0
        elif self.state == "request_footprint":
            if not self.footprint_client.service_is_ready():
                return
            request = SetFootprintProfile.Request()
            request.profile = self.target_profile
            self.future = self.footprint_client.call_async(request)
            self.state = "waiting_footprint"
            self.deadline = now + 1.0
        elif self.state == "request_manipulation":
            if not self.mode_client.service_is_ready():
                return
            request = SetMotionMode.Request()
            request.mode = MotionMode.MANIPULATION
            request.owner = self.owner
            request.reset_emergency_stop = False
            self.future = self.mode_client.call_async(request)
            self.state = "waiting_manipulation"
            self.deadline = now + 1.0

    def _fail(self, detail: str) -> None:
        self.state = "failed"
        self.node.get_logger().error(
            f"manipulation motion lease rejected: {detail}"
        )
