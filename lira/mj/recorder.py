"""Offscreen video of the sim: one mp4 per camera (world.xml's front and top by default).

Frames are taken on sim time -- fps frames per simulated second -- so the video plays at true speed
however fast or slow the sim really ran (planning, policy inference and rendering itself all stall
the loop). Needs MUJOCO_GL=egl, set before mujoco is first imported: the default GLFW backend can't
make an offscreen context on this desktop."""
import pathlib

import imageio.v2 as imageio
import mujoco


class Recorder:
    def __init__(self, model, path, cameras=("front", "top"), fps=30, width=1280, height=720):
        """path: out.mp4 -> out_front.mp4, out_top.mp4, ... one per camera."""
        path = pathlib.Path(path)
        self.paths = {cam: path.with_name(f"{path.stem}_{cam}{path.suffix or '.mp4'}") for cam in cameras}
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.writers = {cam: imageio.get_writer(p, fps=fps, codec="libx264", quality=8)
                        for cam, p in self.paths.items()}
        self.period = 1.0 / fps
        self.next_time = 0.0

    def capture(self, data, draw_overlays=None):
        """Call every tick: renders a frame from each camera whenever one is due.
        draw_overlays(scene): adds extra geoms (targets, capsules) to the rendered scene."""
        if data.time < self.next_time:
            return
        self.next_time += self.period
        for cam, writer in self.writers.items():
            self.renderer.update_scene(data, camera=cam)
            if draw_overlays is not None:
                draw_overlays(self.renderer.scene)
            writer.append_data(self.renderer.render())

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.renderer.close()
        print("Recorded " + ", ".join(str(p) for p in self.paths.values()))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
