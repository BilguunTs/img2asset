"""
cleanup_script.py — Headless Blender mesh cleanup script

Run via:
    blender --background --python cleanup_script.py -- \
        --input  path/to/raw.glb \
        --output path/to/clean.glb \
        --poly-target 8000 \
        --smooth-iterations 2 \
        --texture-size 1024

This script is called by modules/cleanup.py. Do not run directly unless debugging.
"""

import sys
import argparse
import bpy


def parse_args():
    # Args after "--" are passed to the script; before are Blender's own
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",             required=True)
    parser.add_argument("--output",            required=True)
    parser.add_argument("--poly-target",       type=int, default=8000)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--texture-size",      type=int, default=1024)
    parser.add_argument("--fill-holes",        action="store_true", default=True)
    parser.add_argument("--unwrap-uvs",        action="store_true", default=True)
    parser.add_argument("--bake-texture",      action="store_true", default=True)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: str):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "glb" or ext == "gltf":
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Return the imported mesh objects
    return [o for o in bpy.context.selected_objects if o.type == "MESH"]


def join_objects(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def fix_normals(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def fill_holes(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.object.mode_set(mode="OBJECT")


def smooth(obj, iterations: int):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type="SMOOTH")
    mod = obj.modifiers["Smooth"]
    mod.iterations = iterations
    bpy.ops.object.modifier_apply(modifier=mod.name)


def decimate(obj, poly_target: int):
    current_polys = len(obj.data.polygons)
    if current_polys <= poly_target:
        print(f"  Poly count ({current_polys}) already under target ({poly_target}), skipping decimate.")
        return

    ratio = poly_target / current_polys
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type="DECIMATE")
    mod = obj.modifiers["Decimate"]
    mod.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  Decimated: {current_polys} → {len(obj.data.polygons)} polys")


def unwrap_uvs(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_texture(obj, texture_size: int):
    """Bake vertex colors or existing material to a new texture."""
    mat = bpy.data.materials.new(name="BakedMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add image texture node (bake target)
    img = bpy.data.images.new("BakedTexture", width=texture_size, height=texture_size)
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.select = True
    nodes.active = tex_node

    # Add principled BSDF + output
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Set up renderer for baking
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 32

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_selected_to_active=False)

    # Save baked image alongside output
    img.filepath_raw = "//baked_texture.png"
    img.file_format = "PNG"
    img.save()

    print(f"  Texture baked: {texture_size}x{texture_size}px")


def export_mesh(obj, path: str):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "glb" or ext == "gltf":
        bpy.ops.export_scene.gltf(
            filepath=path,
            use_selection=True,
            export_format="GLB",
        )
    elif ext == "obj":
        bpy.ops.wm.obj_export(filepath=path, use_selection=True)
    else:
        raise ValueError(f"Unsupported export format: {ext}")

    print(f"  Exported: {path}")


def main():
    args = parse_args()

    print(f"\n[img2asset] Blender cleanup starting")
    print(f"  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Poly target: {args.poly_target}")

    clear_scene()

    print("  Importing mesh...")
    objects = import_mesh(args.input)

    if not objects:
        print("ERROR: No mesh objects found in input file.")
        sys.exit(1)

    print(f"  Found {len(objects)} mesh object(s), joining...")
    obj = join_objects(objects) if len(objects) > 1 else objects[0]

    print("  Fixing normals...")
    fix_normals(obj)

    if args.fill_holes:
        print("  Filling holes...")
        fill_holes(obj)

    print("  Smoothing...")
    smooth(obj, args.smooth_iterations)

    print("  Decimating...")
    decimate(obj, args.poly_target)

    if args.unwrap_uvs:
        print("  Unwrapping UVs...")
        unwrap_uvs(obj)

    if args.bake_texture:
        print("  Baking texture...")
        bake_texture(obj, args.texture_size)

    print("  Exporting...")
    export_mesh(obj, args.output)

    print(f"\n[img2asset] Done: {args.output}\n")


main()
