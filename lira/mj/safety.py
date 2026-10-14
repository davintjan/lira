"""The safety filter between every nominal command and the motors: gathers the arm's clearance rows
(self-collision, table, the T, a moving obstacle) at the live state and runs the CBF-QP on them."""
from dataclasses import dataclass

import numpy as np

from core.cbf import cbf_qp
from core.geometry import capsule_capsule_distance_batch, capsule_set_rows, clearance_rows, moving_obstacle_rows
from mj.mj_interface import point_jacobian, site_jacobian


@dataclass(frozen=True)
class CBFConfig:
    alpha: float = 5.0           # class-K gain: higher = closer/faster approach allowed before braking
    margin: float = 0.03         # m: clearance held beyond the capsules' actual surfaces
    table_margin: float = 0.005  # m: much tighter -- several links rest just mm above the table by design
    # m past its margin: farther hazards get no row (no Jacobian, smaller QP). One that far could only
    # bind if closing faster than alpha * activation (0.75 m/s) -- raise it if links move faster
    activation: float = 0.15
    joint_vel_max: float = 10.0  # rad/s: box constraint on the solution (and the servos' ctrlrange)


class SafetyFilter:
    def __init__(self, arm, cfg=CBFConfig()):
        self.arm = arm
        self.cfg = cfg

    def _jacobian(self, body_id, point):
        return point_jacobian(self.arm.model, self.arm.data, body_id, point)

    def filter(self, u_des, caps=None, obstacle=None, others=None, metric=None, touch_block=False):
        """The safe joint velocity closest to u_des (see core.cbf.cbf_qp; metric: what "closest" means).
        caps: this tick's RobotGeometry.world(data), if the caller already has it.
        obstacle: MovingObstacle.update's capsule this tick, or None. The whole arm, rod included,
          stays clear of it.
        others: capsules of something else that moves, where they are now -- another arm's links
          (its RobotGeometry.link_capsules_world). This arm keeps its share of each clearance (see
          core.geometry.capsule_set_rows); the other arm, filtered the same way, keeps the rest.
        touch_block: let the rod touch the T, but only as much as u_des asks for (that's the pushing, see
          _task_contact_rows); every other link stays fully guarded."""
        arm, cfg = self.arm, self.cfg
        caps = caps or arm.geometry.world(arm.data)
        object_boxes = arm.env.update(arm.data)
        env_capsules = caps["env"]
        if touch_block:  # the rod's rows against the T come from _task_contact_rows instead
            env_capsules = [dict(cap, guard_objects=False) if cap["name"] == "pusher" else cap for cap in env_capsules]
        rows = clearance_rows(
            arm.flange_pose()[0], site_jacobian(arm.model, arm.data)[:3], caps["self"], caps["pairs"],
            env_capsules, object_boxes, arm.env.table_height, self._jacobian,
            margin=cfg.margin, table_margin=cfg.table_margin, activation=cfg.activation,
        )
        if touch_block:
            rows += self._task_contact_rows(caps["env"], object_boxes, u_des)
        if obstacle is not None:
            rows += self._obstacle_rows(caps["env"], obstacle)
        if others:
            rows += capsule_set_rows(caps["env"], others, self._jacobian, margin=cfg.margin, activation=cfg.activation)
        return cbf_qp(u_des, rows, alpha=cfg.alpha, u_max=cfg.joint_vel_max, metric=metric)

    def _task_contact_rows(self, env_capsules, object_boxes, u_des):
        """The rod against the T, when the task means to push it: the filter may not bring the rod
        closer to the T than u_des itself would. Each row asks
            grad . u >= min(grad . u_des, -alpha * h)
        -- the usual CBF bound, or the task's own approach rate if that's faster. So pushing works
        (the task's approach is always allowed), but dodging something else can only move the rod
        along or away from the T, never into it. As cbf_qp's grad . u >= -alpha * h', that's
        h' = max(h, -grad . u_des / alpha)."""
        cfg = self.cfg
        rod = [dict(cap, guard_table=False) for cap in env_capsules if cap["name"] == "pusher"]  # table: already guarded
        rows = clearance_rows(
            None, None, [], [], rod, object_boxes, self.arm.env.table_height, self._jacobian,
            margin=cfg.margin, table_margin=cfg.table_margin, activation=cfg.activation,
        )
        return [(max(h, -(grad @ u_des) / cfg.alpha), grad) for h, grad in rows]

    def _obstacle_rows(self, env_capsules, obstacle):
        """Rows keeping every robot capsule clear of the moving obstacle. Its own motion changes the
        clearance too: grad . u + h_dot_obstacle >= -alpha * h, which is cbf_qp's grad . u >= -alpha * h'
        with h' = h + h_dot_obstacle / alpha."""
        cfg = self.cfg
        return [(h + h_dot_obstacle / cfg.alpha, grad) for h, grad, h_dot_obstacle in moving_obstacle_rows(
            env_capsules, obstacle, self._jacobian, margin=cfg.margin, activation=cfg.activation, alpha=cfg.alpha)]


class ObstacleMonitor:
    """Tracks how close the moving obstacle gets to the arm, and whether MuJoCo ever has them in
    contact -- the proof the CBF is dodging, not just lucky."""

    def __init__(self, model):
        self.model = model
        self.obstacle_body = model.body("obstacle").id
        self.min_clearance = np.inf  # m
        self.contact_ticks = 0

    def update(self, data, env_capsules, obstacle):
        """Records this tick; returns True on the first tick the obstacle touches the robot."""
        A = np.array([c["a"] for c in env_capsules])
        clearance = capsule_capsule_distance_batch(
            A, np.array([c["b"] for c in env_capsules]), np.array([c["radius"] for c in env_capsules]),
            np.broadcast_to(obstacle["a"], A.shape), np.broadcast_to(obstacle["b"], A.shape),
            np.full(len(env_capsules), obstacle["radius"]))[0].min()
        self.min_clearance = min(self.min_clearance, clearance)
        robot_bodies = {c["body_id"] for c in env_capsules}
        body_of = self.model.geom_bodyid
        touching = any(
            self.obstacle_body in (body_of[c.geom1], body_of[c.geom2])
            and {body_of[c.geom1], body_of[c.geom2]} & robot_bodies
            for c in data.contact[:data.ncon])
        first_hit = touching and self.contact_ticks == 0
        self.contact_ticks += touching
        return first_hit

    def summary(self, so_far=False):
        return (f"obstacle: closest {self.min_clearance * 100:.1f} cm{' so far' if so_far else ''}, "
                f"{self.contact_ticks} contact ticks")
