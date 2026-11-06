"""ContactMonitor: how close two sets of bodies come during a run -- the proof the CBF keeps them apart,
not just luck. One arm against the obstacle, two arms against each other, both against the obstacle."""
import numpy as np

from core.geometry import capsule_capsule_distance


class ContactMonitor:
    """The closest two sides' capsules come, and on how many ticks MuJoCo has their bodies in contact.

    bodies_a, bodies_b: the two sides' body ids in `model`, the model whose contacts update() reads."""

    def __init__(self, model, bodies_a, bodies_b):
        self.body_of_geom = model.geom_bodyid
        self.bodies_a = set(bodies_a)
        self.bodies_b = set(bodies_b)
        self.clearance = np.inf      # m, at the last update
        self.min_clearance = np.inf  # m, over the whole run
        self.contact_ticks = 0

    def update(self, data, caps_a, caps_b):
        """Records this tick: how close the world-frame capsules caps_a and caps_b are, and whether the
        two sides touch. Returns True on the first tick they do."""
        self.clearance = closest_distance(caps_a, caps_b)
        self.min_clearance = min(self.min_clearance, self.clearance)
        touching = any(self._between_sides(contact) for contact in data.contact[:data.ncon])
        first_touch = touching and self.contact_ticks == 0
        self.contact_ticks += touching
        return first_touch

    def _between_sides(self, contact):
        bodies = {self.body_of_geom[contact.geom1], self.body_of_geom[contact.geom2]}
        return bool(bodies & self.bodies_a) and bool(bodies & self.bodies_b)

    def summary(self, so_far=False):
        closest = f"closest {self.min_clearance * 100:.1f} cm"
        if so_far:
            closest += " so far"
        return f"{closest}, {self.contact_ticks} contact ticks"


def closest_distance(caps_a, caps_b):
    """The smallest distance (m) between any capsule of caps_a and any of caps_b."""
    pairs = [(cap_a, cap_b) for cap_a in caps_a for cap_b in caps_b]
    first = [cap_a for cap_a, _ in pairs]
    second = [cap_b for _, cap_b in pairs]
    distances, _, _ = capsule_capsule_distance(
        np.array([cap["a"] for cap in first]), np.array([cap["b"] for cap in first]),
        np.array([cap["radius"] for cap in first]),
        np.array([cap["a"] for cap in second]), np.array([cap["b"] for cap in second]),
        np.array([cap["radius"] for cap in second]))
    return distances.min()
