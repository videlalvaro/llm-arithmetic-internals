"""Blender render for the ACM Queue/Rune article helix hero.

Run:
  /opt/homebrew/bin/blender --background --python scripts/blender_article_helix_render.py

Output:
  docs/article_figures/blender_integer_helix.png
"""
from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "article_figures" / "blender_integer_helix.png"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def material(name: str, color: tuple[float, float, float, float], roughness: float = 0.35,
             metallic: float = 0.0, emission: tuple[float, float, float, float] | None = None,
             emission_strength: float = 0.0) -> bpy.types.Material:
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
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def main() -> None:
    clear_scene()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    helix_mat = material("deep teal helix", (0.03, 0.36, 0.39, 1), roughness=0.28, metallic=0.12)
    point_mat = material("gold phase points", (0.94, 0.62, 0.16, 1), roughness=0.2,
                         emission=(0.95, 0.52, 0.12, 1), emission_strength=0.22)
    active_mat = material("active value", (0.77, 0.22, 0.30, 1), roughness=0.22,
                          emission=(0.85, 0.08, 0.18, 1), emission_strength=0.55)
    ghost_mat = material("projection ghost", (0.15, 0.38, 0.58, 0.45), roughness=0.45)

    radius = 1.55
    height = 7.8
    turns = 7.0
    pts = []
    for i in range(1300):
        t = i / 1299
        theta = 2 * math.pi * turns * t
        pts.append(Vector((radius * math.cos(theta), height * (t - 0.5), radius * math.sin(theta))))
    make_curve("integer helix", pts, 0.025, helix_mat)

    for n in range(0, 80, 5):
        t = n / 79
        theta = 2 * math.pi * turns * t
        loc = Vector((radius * math.cos(theta), height * (t - 0.5), radius * math.sin(theta)))
        add_sphere(f"phase point {n}", loc, 0.07, point_mat)

    active_n = 37
    t = active_n / 79
    theta = 2 * math.pi * turns * t
    active = Vector((radius * math.cos(theta), height * (t - 0.5), radius * math.sin(theta)))
    add_sphere("selected integer", active, 0.18, active_mat)
    make_curve("phase projection", [Vector((0, active.y, 0)), active], 0.012, ghost_mat)

    bpy.ops.mesh.primitive_circle_add(vertices=128, radius=radius, fill_type="TRIFAN", location=(0, active.y, 0))
    disc = bpy.context.object
    disc.name = "phase disc"
    disc.rotation_euler[0] = math.pi / 2
    disc.data.materials.append(material("phase plane", (0.94, 0.91, 0.82, 0.22), roughness=0.7))

    # Floor and lights.
    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0, -4.25, 0))
    floor = bpy.context.object
    floor.name = "warm matte floor"
    floor.data.materials.append(material("warm paper", (0.79, 0.76, 0.66, 1), roughness=0.82))

    bpy.ops.object.light_add(type="AREA", location=(-3.5, 5.5, 4.0))
    key = bpy.context.object
    key.name = "large soft key"
    key.data.energy = 550
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(3.0, 1.5, -2.5))
    rim = bpy.context.object
    rim.name = "small rim"
    rim.data.energy = 80
    rim.data.color = (0.78, 0.9, 1.0)

    bpy.ops.object.camera_add(location=(5.2, 3.0, 6.4), rotation=(math.radians(63), 0, math.radians(43)))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    direction = Vector((0, 0.1, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 58
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 7.4
    cam.data.dof.aperture_fstop = 6.5

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.world.color = (0.98, 0.96, 0.9)
    scene.render.resolution_x = 1800
    scene.render.resolution_y = 1200
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.filepath = str(OUT)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
