"""Render a short social animation for the Rune article.

Run:
  /opt/homebrew/bin/blender --background --python scripts/blender_social_helix_animation.py

Outputs:
  .cache/social_helix_frames/frame_####.png
"""
from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector


REPO = Path(__file__).resolve().parent.parent
FRAME_DIR = REPO / ".cache" / "social_helix_frames"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.35,
    metallic: float = 0.0,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = emission
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def make_curve(name: str, points: list[Vector], bevel_depth: float, mat: bpy.types.Material) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 24
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 8
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (co.x, co.y, co.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def add_sphere(name: str, loc: Vector, radius: float, mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=20, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def helix_point(index: float, radius: float = 1.55, height: float = 7.8, turns: float = 7.0) -> Vector:
    t = index / 79.0
    theta = 2 * math.pi * turns * t
    return Vector((radius * math.cos(theta), height * (t - 0.5), radius * math.sin(theta)))


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    clear_scene()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old_frame in FRAME_DIR.glob("frame_*.png"):
        old_frame.unlink()

    helix_mat = material("deep teal helix", (0.03, 0.36, 0.39, 1), roughness=0.28, metallic=0.12)
    point_mat = material(
        "gold phase points",
        (0.94, 0.62, 0.16, 1),
        roughness=0.2,
        emission=(0.95, 0.52, 0.12, 1),
        emission_strength=0.18,
    )
    active_mat = material(
        "active value",
        (0.86, 0.22, 0.42, 1),
        roughness=0.22,
        emission=(0.85, 0.08, 0.18, 1),
        emission_strength=0.65,
    )
    ghost_mat = material("phase projection", (0.15, 0.38, 0.58, 1), roughness=0.45)
    plane_mat = material("phase plane", (0.92, 0.89, 0.80, 0.62), roughness=0.78)

    radius = 1.55
    height = 7.8
    turns = 7.0
    pts = []
    for i in range(1200):
        t = i / 1199
        theta = 2 * math.pi * turns * t
        pts.append(Vector((radius * math.cos(theta), height * (t - 0.5), radius * math.sin(theta))))
    make_curve("integer helix", pts, 0.026, helix_mat)

    for n in range(0, 80, 5):
        add_sphere(f"phase point {n}", helix_point(n), 0.07, point_mat)

    active = add_sphere("selected integer", helix_point(18), 0.18, active_mat)
    projection = make_curve("projection ray", [Vector((0, active.location.y, 0)), active.location], 0.012, ghost_mat)

    bpy.ops.mesh.primitive_circle_add(vertices=128, radius=radius, fill_type="TRIFAN", location=(0, active.location.y, 0))
    disc = bpy.context.object
    disc.name = "phase disc"
    disc.rotation_euler[0] = math.pi / 2
    disc.data.materials.append(plane_mat)

    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0, -4.25, 0))
    floor = bpy.context.object
    floor.name = "warm matte floor"
    floor.data.materials.append(material("warm paper", (0.79, 0.76, 0.66, 1), roughness=0.82))

    bpy.ops.object.light_add(type="AREA", location=(-3.5, 5.5, 4.0))
    key = bpy.context.object
    key.data.energy = 560
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(3.0, 1.5, -2.5))
    rim = bpy.context.object
    rim.data.energy = 95
    rim.data.color = (0.78, 0.9, 1.0)

    bpy.ops.object.camera_add(location=(5.0, 2.9, 6.2))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 60
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 7.2
    cam.data.dof.aperture_fstop = 7.0

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 48
    scene.frame_set(1)
    scene.render.fps = 12
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 64
    scene.world.color = (0.98, 0.96, 0.9)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.image_settings.file_format = "PNG"

    def update_frame(frame: int) -> None:
        u = (frame - scene.frame_start) / (scene.frame_end - scene.frame_start)
        value = 18 + 43 * u
        loc = helix_point(value)
        active.location = loc
        disc.location.y = loc.y
        projection.data.splines[0].points[0].co = (0, loc.y, 0, 1)
        projection.data.splines[0].points[1].co = (loc.x, loc.y, loc.z, 1)

        angle = math.radians(39 + 30 * u)
        cam.location = Vector((5.1 * math.cos(angle), 2.55 + 0.35 * math.sin(math.pi * u), 5.1 * math.sin(angle) + 1.35))
        look_at(cam, Vector((0, 0.05, 0)))

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        update_frame(frame)
        active.keyframe_insert(data_path="location", frame=frame)
        disc.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        scene.render.filepath = str(FRAME_DIR / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
