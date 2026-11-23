# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""Hanging-sheet bone generator for open, closed and solidified meshes."""

import json
import math

import bmesh
import bpy
import gpu
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

LEVELS = 4
CONTROL_CHAIN_COLORS = (
    (0.78, 1.00, 0.10, 1.0),
    (0.10, 0.90, 0.24, 1.0),
    (0.10, 0.42, 1.0, 1.0),
    (0.86, 0.16, 1.0, 1.0),
)
EPS = 1.0e-8
MIN_WEIGHT = 0.001
PARENT_KEY = "__AUTO_SHEET_PARENT__"

PROP_COLLECTION = "auto_sheet_control_collection"
PROP_TARGETS = "auto_sheet_target_meshes"
PROP_REGIONS = "auto_sheet_regions"
PROP_INDICES = "auto_sheet_selected_indices"
PROP_PREFIX = "auto_sheet_prefix"
PROP_MODE = "auto_sheet_group_mode"
PROP_KIND = "auto_sheet_generated_kind"
KIND_CONTROL = "SHEET_CONTROL"
KIND_PROXY = "SHEET_PROXY"

_DRAW_HANDLE = None
_MIRROR_GUARD = False
_LIVE_PROXY_CACHE = {
    "signature": None,
    "triangles": [],
    "lines": [],
}


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def smoothstep(value):
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def safe_normalized(value, fallback=(0.0, 0.0, 1.0)):
    vector = Vector(value)
    if vector.length_squared <= EPS:
        vector = Vector(fallback)
    if vector.length_squared <= EPS:
        vector = Vector((0.0, 0.0, 1.0))
    return vector.normalized()


def average(points):
    points = list(points)
    if not points:
        return Vector((0.0, 0.0, 0.0))
    result = Vector((0.0, 0.0, 0.0))
    for point in points:
        result += Vector(point)
    return result / len(points)


def coordinate_median(points):
    points = list(points)
    if not points:
        return Vector((0.0, 0.0, 0.0))

    def median(values):
        values = sorted(float(value) for value in values)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) * 0.5

    return Vector((
        median(point.x for point in points),
        median(point.y for point in points),
        median(point.z for point in points),
    ))


def validate_prefix(prefix):
    if not str(prefix).strip("_ "):
        raise ValueError("命名前缀不能为空，也不能只包含下划线")


def prefix_stem(prefix):
    return str(prefix).rstrip("_") or "AutoSheet"


def control_collection_name(prefix, mesh_name):
    return f"{prefix_stem(prefix)}Controls_{mesh_name}"


def proxy_collection_name(prefix, mesh_name):
    return f"{prefix_stem(prefix)}Proxy_{mesh_name}"


def control_name(prefix, region, row, column):
    return f"{prefix}CTRL_P{region:02d}_V{row:02d}_U{column:02d}"


def bone_name(prefix, region, column, row, separate, reverse, columns):
    logical_column = columns - 1 - column if reverse else column
    region_part = f"P{region:02d}_" if separate else ""
    return f"{prefix}{region_part}C{logical_column:02d}_R{row:02d}"


def get_or_create_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(collection)
    return collection


def poll_mesh(_self, obj):
    return obj is not None and obj.type == 'MESH'


def poll_armature(_self, obj):
    return obj is not None and obj.type == 'ARMATURE'


def edit_meshes(context):
    meshes = [
        obj for obj in context.selected_objects
        if obj.type == 'MESH' and obj.mode == 'EDIT' and obj.name in context.view_layer.objects
    ]
    active = context.active_object
    if active in meshes:
        meshes.remove(active)
        meshes.insert(0, active)
    return meshes


def read_vertex_record(mesh_obj, vertex):
    matrix_world = mesh_obj.matrix_world
    normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
    return {
        "mesh": mesh_obj.name,
        "index": int(vertex.index),
        "world": matrix_world @ vertex.co,
        "normal": safe_normalized(normal_matrix @ vertex.normal),
    }


def detect_free_end_tip(indices, selected, adjacency, mesh_obj):
    """Detect a pinched lower end from geometry, independent of topology type.

    A solidified sheet can have one low tip on each skin. Those nearby tips are
    treated as one geometric tip, while an ordinary wide lower edge is left
    unchanged.
    """
    matrix_world = mesh_obj.matrix_world
    points = {index: matrix_world @ selected[index].co for index in indices}
    top = max(point.z for point in points.values())
    bottom = min(point.z for point in points.values())
    height = top - bottom
    if height <= EPS:
        return None

    xs = [point.x for point in points.values()]
    ys = [point.y for point in points.values()]
    overall_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    tolerance = max(height * 0.01, overall_span * 0.002, 1.0e-6)
    lowest = {
        index for index, point in points.items()
        if point.z <= bottom + tolerance
    }
    if not lowest:
        return None

    # A pinched end may contain duplicated/separated skin vertices, but it
    # should still be a small cluster rather than a complete horizontal hem.
    maximum_tip_vertices = max(8, int(math.ceil(len(indices) * 0.08)))
    if len(lowest) > maximum_tip_vertices:
        return None
    lowest_points = [points[index] for index in lowest]
    tip_span = math.hypot(
        max(point.x for point in lowest_points) - min(point.x for point in lowest_points),
        max(point.y for point in lowest_points) - min(point.y for point in lowest_points),
    )
    if overall_span > EPS and tip_span > max(overall_span * 0.12, tolerance * 2.0):
        return None

    outer_neighbors = {
        neighbor
        for index in lowest
        for neighbor in adjacency.get(index, ())
        if neighbor in indices and neighbor not in lowest
    }
    raised_neighbors = [
        points[index]
        for index in outer_neighbors
        if points[index].z >= bottom + height * 0.025
    ]
    if len(raised_neighbors) < 2:
        return None

    neighbor_span = math.hypot(
        max(point.x for point in raised_neighbors) - min(point.x for point in raised_neighbors),
        max(point.y for point in raised_neighbors) - min(point.y for point in raised_neighbors),
    )
    if overall_span > EPS and neighbor_span < overall_span * 0.18:
        return None
    return coordinate_median(lowest_points)

def collect_components(mesh_obj):
    """Collect selected point-cloud components without topology rejection."""
    if mesh_obj.mode != 'EDIT':
        return []
    bm = bmesh.from_edit_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.index_update()
    bm.normal_update()
    selected = {
        int(vertex.index): vertex
        for vertex in bm.verts
        if vertex.select and not vertex.hide
    }
    if not selected:
        return []

    # Connectivity is used only when the user requests per-component output.
    # It never decides whether a selection is a sheet, skirt, hole or solid.
    adjacency = {index: set() for index in selected}
    for edge in bm.edges:
        a, b = int(edge.verts[0].index), int(edge.verts[1].index)
        if a in selected and b in selected:
            adjacency[a].add(b)
            adjacency[b].add(a)

    faces_by_vertex = {}
    for face in bm.faces:
        ids = tuple(v.index for v in face.verts)
        if all(i in selected for i in ids):
            faces_by_vertex.setdefault(ids[0], []).append(ids)

    components = []
    remaining = set(selected)
    while remaining:
        seed = remaining.pop()
        indices = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    indices.add(neighbor)
                    stack.append(neighbor)
        records = [
            read_vertex_record(mesh_obj, selected[index])
            for index in sorted(indices)
        ]
        components.append({
            "key": f"{mesh_obj.name}:{min(indices)}",
            "mesh": mesh_obj.name,
            "indices": sorted(indices),
            "records": records,
            "faces": [face for i in indices for face in faces_by_vertex.get(i, ())],
            "tip_point": detect_free_end_tip(
                indices, selected, adjacency, mesh_obj
            ),
        })
    return components

def collect_selection(context, settings):
    meshes = edit_meshes(context)
    if not meshes:
        raise ValueError("请在多对象编辑模式中选择需要处理的垂落面片顶点")
    components = []
    for mesh_obj in meshes:
        object_components = collect_components(mesh_obj)
        if not object_components:
            raise ValueError(f"网格“{mesh_obj.name}”没有选中顶点")
        components.extend(object_components)

    if settings.group_mode == 'MERGED':
        regions = [components]
    else:
        regions = [[component] for component in components]

    for region in regions:
        count = sum(len(component["records"]) for component in region)
        if count < 4:
            if settings.group_mode == 'COMPONENTS':
                name = region[0]["mesh"]
                raise ValueError(
                    f"网格“{name}”存在少于 4 个选中顶点的连通块；"
                    "请补充选区或改用“合并处理”"
                )
            raise ValueError("至少需要 4 个选中顶点来拟合面片控制网格")
    return meshes, regions


def width_axis(records):
    center = average(record["world"] for record in records)
    xx = yy = xy = 0.0
    for record in records:
        delta = record["world"] - center
        xx += delta.x * delta.x
        yy += delta.y * delta.y
        xy += delta.x * delta.y
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    axis = Vector((math.cos(angle), math.sin(angle), 0.0))
    if axis.length_squared <= EPS:
        normal = average(record["normal"] for record in records)
        axis = normal.cross(Vector((0.0, 0.0, 1.0)))
    axis = safe_normalized(axis, (1.0, 0.0, 0.0))
    if axis.x < -1.0e-6 or (abs(axis.x) <= 1.0e-6 and axis.y < 0.0):
        axis.negate()
    return center, axis


def weighted_candidate_point(records, parameters, target_u, target_v, level, levels=LEVELS):
    scored = []
    for record, (u, v) in zip(records, parameters):
        du = abs(u - target_u) / 0.22
        if level == 0:
            dv = v / 0.12
        elif level == levels - 1:
            dv = (1.0 - v) / 0.12
        else:
            dv = abs(v - target_v) / 0.20
        score = du * du + dv * dv
        scored.append((score, record["world"]))
    scored.sort(key=lambda item: item[0])
    count = max(3, min(20, max(3, len(scored) // 10)))
    chosen = scored[:count]
    total = 0.0
    point = Vector((0.0, 0.0, 0.0))
    for score, candidate in chosen:
        weight = 1.0 / (0.025 + score)
        point += candidate * weight
        total += weight
    return point / total if total > EPS else Vector(chosen[0][1])


def fit_region_grid(components):
    all_records = [record for component in components for record in component["records"]]
    if len(all_records) < 4:
        raise ValueError("至少需要 4 个有效顶点来拟合面片控制网格")
    center, axis = width_axis(all_records)
    projections = [(record["world"] - center).dot(axis) for record in all_records]
    minimum, maximum = min(projections), max(projections)
    top = max(record["world"].z for record in all_records)
    bottom = min(record["world"].z for record in all_records)
    if maximum - minimum <= EPS:
        raise ValueError("面片横向宽度过小，无法生成多条骨骼链")
    if top - bottom <= EPS:
        raise ValueError("选区缺少从上到下的 Z 高度差，无法拟合垂落面片")

    component_grids = []
    for component in components:
        records = component["records"]
        parameters = []
        for record in records:
            projection = (record["world"] - center).dot(axis)
            u = clamp((projection - minimum) / (maximum - minimum))
            v = clamp((top - record["world"].z) / (top - bottom))
            parameters.append((u, v))
        grid = []
        tip_point = component.get("tip_point")
        for row in range(LEVELS):
            target_v = row / (LEVELS - 1)
            if row == LEVELS - 1 and tip_point is not None:
                grid.append([Vector(tip_point) for _column in range(LEVELS)])
            else:
                grid.append([
                    weighted_candidate_point(
                        records,
                        parameters,
                        column / (LEVELS - 1),
                        target_v,
                        row,
                    )
                    for column in range(LEVELS)
                ])
        component_grids.append(grid)

    grid = []
    for row in range(LEVELS):
        grid.append([
            coordinate_median(component_grid[row][column] for component_grid in component_grids)
            for column in range(LEVELS)
        ])
    diagonal = max((record["world"] - center).length for record in all_records) * 2.0
    return grid, max(diagonal * 0.02, (top - bottom) * 0.015, 0.002)


def delete_controls(prefix, primary_name):
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.get(PROP_KIND) == KIND_CONTROL
            and obj.get("auto_sheet_target") == primary_name
            and obj.name.startswith(prefix)
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    expected = control_collection_name(prefix, primary_name)
    collection = bpy.data.collections.get(expected)
    if collection is not None and not collection.objects and not collection.children:
        bpy.data.collections.remove(collection)
    return removed


def create_controls(primary, settings, region_grids, display_size):
    old_prefix = str(primary.get(PROP_PREFIX, settings.prefix))
    delete_controls(old_prefix, primary.name)
    if old_prefix != settings.prefix:
        delete_controls(settings.prefix, primary.name)
    collection = get_or_create_collection(control_collection_name(settings.prefix, primary.name))
    for region, grid in enumerate(region_grids):
        for row in range(LEVELS):
            for column in range(LEVELS):
                obj = bpy.data.objects.new(control_name(settings.prefix, region, row, column), None)
                collection.objects.link(obj)
                obj.empty_display_type = 'SPHERE'
                obj.empty_display_size = display_size
                obj.location = grid[row][column]
                obj.show_in_front = True
                obj.color = CONTROL_CHAIN_COLORS[column % len(CONTROL_CHAIN_COLORS)]
                obj[PROP_KIND] = KIND_CONTROL
                obj["auto_sheet_target"] = primary.name
                obj["auto_sheet_region"] = region
                obj["auto_sheet_row"] = row
                obj["auto_sheet_column"] = column
                set_locked(obj, True)
    return collection


def find_controls(primary, prefix, require=True):
    collection_name = str(primary.get(PROP_COLLECTION, control_collection_name(prefix, primary.name)))
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        if require:
            raise ValueError("找不到面片控制点集合，请先生成控制点")
        return [], None
    controls = {}
    for obj in collection.objects:
        if obj.get(PROP_KIND) != KIND_CONTROL or obj.get("auto_sheet_target") != primary.name:
            continue
        key = (int(obj.get("auto_sheet_region", -1)), int(obj.get("auto_sheet_row", -1)), int(obj.get("auto_sheet_column", -1)))
        controls[key] = obj
    region_count = 0
    if controls:
        region_count = max(key[0] for key in controls) + 1
    grids = []
    missing = []
    for region in range(region_count):
        grid = []
        for row in range(LEVELS):
            grid_row = []
            for column in range(LEVELS):
                obj = controls.get((region, row, column))
                if obj is None:
                    missing.append(control_name(prefix, region, row, column))
                grid_row.append(obj)
            grid.append(grid_row)
        grids.append(grid)
    if require and (not grids or missing):
        raise ValueError("面片控制点不完整：" + ", ".join(missing[:4]))
    return grids, collection


def flatten_controls(grids):
    return [obj for grid in grids for row in grid for obj in row if obj is not None]


def set_locked(obj, locked):
    value = bool(locked)
    obj.hide_select = value
    obj.lock_location = (value, value, value)
    obj.lock_rotation = (value, value, value)
    obj.lock_scale = (value, value, value)


def position_map(grids):
    return {
        obj.name: [float(obj.location.x), float(obj.location.y), float(obj.location.z)]
        for obj in flatten_controls(grids)
    }


def parse_dict(raw):
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def vector_value(value):
    try:
        return Vector((float(value[0]), float(value[1]), float(value[2])))
    except (TypeError, ValueError, IndexError):
        return None


def restore_positions(grids, raw):
    snapshot = parse_dict(raw)
    for obj in flatten_controls(grids):
        value = vector_value(snapshot.get(obj.name))
        if value is not None:
            obj.location = value


def primary_from_settings(settings):
    obj = settings.target_mesh
    if obj is None and settings.edit_mesh_name:
        obj = bpy.data.objects.get(settings.edit_mesh_name)
    return obj if obj is not None and obj.type == 'MESH' else None


def capture_hide_state(context, controls):
    names = {obj.name for obj in controls}
    return json.dumps({obj.name: bool(obj.hide_select) for obj in context.view_layer.objects if obj.name not in names}, separators=(",", ":"))


def enter_edit(context, settings, collection=None):
    primary = primary_from_settings(settings)
    if primary is None:
        raise ValueError("找不到面片生成器的主网格")
    grids, found_collection = find_controls(primary, settings.prefix, require=True)
    collection = collection or found_collection
    controls = flatten_controls(grids)
    for grid in grids:
        for row in range(len(grid)):
            for column in range(len(grid[row])):
                grid[row][column].color = CONTROL_CHAIN_COLORS[
                    column % len(CONTROL_CHAIN_COLORS)
                ]
    if not controls:
        raise ValueError("没有可编辑的面片控制点")
    skirt = getattr(context.scene, "auto_skirt_settings", None)
    if skirt is not None and skirt.control_edit_active:
        raise ValueError("请先完成裙子控制点编辑")
    settings.edit_mesh_name = primary.name
    settings.edit_snapshot = json.dumps(position_map(grids), separators=(",", ":"))
    settings.edit_last_positions = settings.edit_snapshot
    settings.edit_hide_state = capture_hide_state(context, controls)
    settings.edit_active = True
    _LIVE_PROXY_CACHE["signature"] = None
    if context.active_object is not None and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in controls:
        obj.hide_set(False, view_layer=context.view_layer)
    names = {obj.name for obj in controls}
    for obj in context.view_layer.objects:
        obj.hide_select = obj.name not in names
        try:
            obj.select_set(obj == controls[0])
        except RuntimeError:
            pass
    for obj in controls:
        set_locked(obj, False)
    context.view_layer.objects.active = controls[0]
    for area in context.screen.areas if context.screen else []:
        area.tag_redraw()


def leave_edit(context, settings, reset=False):
    primary = primary_from_settings(settings)
    try:
        grids, _collection = find_controls(primary, settings.prefix, require=True)
        if reset:
            restore_positions(grids, settings.edit_snapshot)
        for obj in flatten_controls(grids):
            set_locked(obj, True)
            obj.select_set(False)
    except (ValueError, RuntimeError, ReferenceError):
        pass
    for name, hidden in parse_dict(settings.edit_hide_state).items():
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.name in context.view_layer.objects:
            obj.hide_select = bool(hidden)
    if primary is not None and primary.name in context.view_layer.objects:
        context.view_layer.objects.active = primary
        primary.select_set(True)
    settings.edit_active = False
    settings.edit_snapshot = ""
    settings.edit_last_positions = ""
    settings.edit_hide_state = ""
    _LIVE_PROXY_CACHE["signature"] = None
    _LIVE_PROXY_CACHE["triangles"] = []
    _LIVE_PROXY_CACHE["lines"] = []
    for area in context.screen.areas if context.screen else []:
        area.tag_redraw()


def changed_names(current, previous):
    changed = set()
    for name, value in current.items():
        a, b = vector_value(value), vector_value(previous.get(name))
        if a is not None and b is not None and (a - b).length_squared > 1.0e-7:
            changed.add(name)
    return changed


def mirror_point(point, center, width_axis):
    point = Vector(point)
    return point - 2.0 * width_axis * (point - center).dot(width_axis)


def choose_source(obj_a, obj_b, changed, active_name):
    if active_name == obj_a.name and obj_a.name in changed:
        return obj_a, obj_b
    if active_name == obj_b.name and obj_b.name in changed:
        return obj_b, obj_a
    if obj_a.name in changed and obj_b.name not in changed:
        return obj_a, obj_b
    if obj_b.name in changed and obj_a.name not in changed:
        return obj_b, obj_a
    if obj_a.name in changed and obj_b.name in changed:
        return obj_a, obj_b
    return None, None


def apply_local_mirror(context):
    global _MIRROR_GUARD
    scene = context.scene
    settings = getattr(scene, "auto_sheet_settings", None)
    if _MIRROR_GUARD or settings is None or not settings.edit_active or not settings.local_symmetry:
        return
    primary = primary_from_settings(settings)
    try:
        grids, _collection = find_controls(primary, settings.prefix, require=True)
    except ValueError:
        return
    current = position_map(grids)
    previous = parse_dict(settings.edit_last_positions)
    changed = changed_names(current, previous)
    if not changed:
        return
    snapshot = parse_dict(settings.edit_snapshot)
    active_name = context.active_object.name if context.active_object else ""
    _MIRROR_GUARD = True
    try:
        for grid in grids:
            for row in range(LEVELS):
                snapshot_points = [vector_value(snapshot.get(obj.name)) for obj in grid[row]]
                if any(point is None for point in snapshot_points):
                    continue
                center = average(snapshot_points)
                width = safe_normalized(snapshot_points[-1] - snapshot_points[0], (1.0, 0.0, 0.0))
                for first, second in ((0, 3), (1, 2)):
                    source, target = choose_source(grid[row][first], grid[row][second], changed, active_name)
                    if source is not None:
                        target.location = mirror_point(source.location, center, width)
        settings.edit_last_positions = json.dumps(position_map(grids), separators=(",", ":"))
    finally:
        _MIRROR_GUARD = False


@persistent
def depsgraph_update(scene, depsgraph):
    del depsgraph
    settings = getattr(scene, "auto_sheet_settings", None)
    if settings is not None and settings.edit_active:
        apply_local_mirror(bpy.context)


def draw_header(self, context):
    settings = getattr(context.scene, "auto_sheet_settings", None)
    if settings is None or not settings.edit_active:
        return
    layout = self.layout
    layout.separator_spacer()
    row = layout.row(align=True)
    row.label(text="面片控制点编辑", icon='EMPTY_AXIS')
    row.prop(settings, "local_symmetry", text="局部横向对称")
    row.prop(settings, "live_proxy_preview", text="实时代理")
    row.operator("auto_sheet.reverse_horizontal", text="反转横向")
    row.operator("auto_sheet.reset_edit", text="复原", icon='LOOP_BACK')
    row.operator("auto_sheet.cancel_edit", text="取消", icon='CANCEL')
    row.operator("auto_sheet.finish_edit", text="完成", icon='CHECKMARK')


def stored_targets(primary):
    try:
        names = json.loads(primary.get(PROP_TARGETS, "[]"))
    except (TypeError, json.JSONDecodeError):
        names = []
    return [obj for name in names if (obj := bpy.data.objects.get(str(name))) is not None and obj.type == 'MESH']


def marker_positions():
    settings = getattr(bpy.context.scene, "auto_sheet_settings", None)
    if settings is None or not settings.edit_active:
        return [], []
    primary = primary_from_settings(settings)
    if primary is None:
        return [], []
    grids, _collection = find_controls(primary, settings.prefix, require=False)
    control_chains = [
        [
            grid[row][column].matrix_world.translation.copy()
            for row in range(LEVELS)
            if grid[row][column] is not None
        ]
        for grid in grids
        for column in range(LEVELS)
    ]
    selected = []
    for mesh_obj in stored_targets(primary):
        try:
            indices = json.loads(mesh_obj.get(PROP_INDICES, "[]"))
        except (TypeError, json.JSONDecodeError):
            indices = []
        selected.extend(
            mesh_obj.matrix_world @ mesh_obj.data.vertices[int(index)].co
            for index in indices
            if 0 <= int(index) < len(mesh_obj.data.vertices)
        )
    return control_chains, selected


def live_proxy_signature(settings, primary, grids):
    values = [
        primary.name,
        settings.group_mode,
        int(settings.bones_per_chain),
        int(settings.chains),
    ]
    for grid in grids:
        for row in range(LEVELS):
            for column in range(LEVELS):
                point = grid[row][column].matrix_world.translation
                values.extend(round(float(component), 6) for component in point)
    return tuple(values)


def live_proxy_geometry():
    """Cache the final open-sheet proxy topology while controls are edited."""
    settings = getattr(bpy.context.scene, "auto_sheet_settings", None)
    if settings is None or not settings.edit_active or not settings.live_proxy_preview:
        _LIVE_PROXY_CACHE["signature"] = None
        _LIVE_PROXY_CACHE["triangles"] = []
        _LIVE_PROXY_CACHE["lines"] = []
        return [], []
    primary = primary_from_settings(settings)
    if primary is None:
        return [], []
    grids, _collection = find_controls(primary, settings.prefix, require=True)
    signature = live_proxy_signature(settings, primary, grids)
    if signature == _LIVE_PROXY_CACHE["signature"]:
        return _LIVE_PROXY_CACHE["triangles"], _LIVE_PROXY_CACHE["lines"]

    triangles = []
    lines = []
    for grid in grids:
        control_grid = [
            [obj.matrix_world.translation.copy() for obj in row]
            for row in grid
        ]
        nodes = build_proxy_nodes(
            control_grid,
            settings.bones_per_chain,
            settings.chains,
        )
        vertices, faces, _vertex_uv = proxy_topology(nodes, triangulated=False)
        edges = set()
        for face in faces:
            if len(face) >= 3:
                for offset in range(1, len(face) - 1):
                    triangles.extend((
                        vertices[face[0]],
                        vertices[face[offset]],
                        vertices[face[offset + 1]],
                    ))
            for offset, first in enumerate(face):
                second = face[(offset + 1) % len(face)]
                edges.add(tuple(sorted((int(first), int(second)))))
        for first, second in edges:
            lines.extend((vertices[first], vertices[second]))

    _LIVE_PROXY_CACHE["signature"] = signature
    _LIVE_PROXY_CACHE["triangles"] = triangles
    _LIVE_PROXY_CACHE["lines"] = lines
    return triangles, lines


def chain_display_color(index):
    return CONTROL_CHAIN_COLORS[index % len(CONTROL_CHAIN_COLORS)]


def draw_markers():
    try:
        control_chains, selected = marker_positions()
        proxy_triangles, proxy_lines = live_proxy_geometry()
        if not control_chains and not selected and not proxy_lines:
            return
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        shader.bind()

        if proxy_triangles:
            gpu.state.depth_test_set('LESS_EQUAL')
            shader.uniform_float("color", (0.10, 0.72, 1.0, 0.16))
            batch_for_shader(
                shader, 'TRIS', {"pos": proxy_triangles}
            ).draw(shader)
        if proxy_lines:
            gpu.state.depth_test_set('LESS_EQUAL')
            try:
                gpu.state.line_width_set(1.5)
            except (AttributeError, RuntimeError):
                pass
            shader.uniform_float("color", (0.15, 0.82, 1.0, 0.82))
            batch_for_shader(shader, 'LINES', {"pos": proxy_lines}).draw(shader)

        gpu.state.depth_test_set('NONE')
        gpu.state.point_size_set(15.0)
        for chain_index, positions in enumerate(control_chains):
            if not positions:
                continue
            color = chain_display_color(chain_index)
            shader.uniform_float("color", color)
            batch_for_shader(shader, 'POINTS', {"pos": positions}).draw(shader)
            if len(positions) > 1:
                chain_lines = []
                for index in range(len(positions) - 1):
                    chain_lines.extend((positions[index], positions[index + 1]))
                batch_for_shader(shader, 'LINES', {"pos": chain_lines}).draw(shader)
        if selected:
            gpu.state.point_size_set(8.0)
            shader.uniform_float("color", (1.0, 0.28, 0.04, 1.0))
            batch_for_shader(shader, 'POINTS', {"pos": selected}).draw(shader)
    except (ReferenceError, RuntimeError, ValueError, KeyError):
        pass
    finally:
        try:
            gpu.state.point_size_set(1.0)
            gpu.state.line_width_set(1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.blend_set('NONE')
        except (AttributeError, RuntimeError):
            pass

def install_draw_handler():
    global _DRAW_HANDLE
    remove_draw_handler()
    _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(draw_markers, (), 'WINDOW', 'POST_VIEW')


def remove_draw_handler():
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, 'WINDOW')
        except (ReferenceError, RuntimeError):
            pass
        _DRAW_HANDLE = None

# ---------------------------------------------------------------------------
# Open control surface, proxy, bones and weights
# ---------------------------------------------------------------------------


def interpolate_four(points, value):
    value = clamp(value)
    scaled = value * (LEVELS - 1)
    segment = min(int(math.floor(scaled)), LEVELS - 2)
    factor = smoothstep(scaled - segment)
    return Vector(points[segment]).lerp(Vector(points[segment + 1]), factor)


def _centripetal_interval(first, second):
    """Return a non-zero alpha=0.5 Catmull-Rom knot interval."""
    return max((Vector(second) - Vector(first)).length ** 0.5, 1.0e-5)


def _knot_lerp(first, second, first_t, second_t, target_t):
    denominator = second_t - first_t
    if abs(denominator) <= EPS:
        return Vector(first).lerp(Vector(second), 0.5)
    factor = (target_t - first_t) / denominator
    return Vector(first) * (1.0 - factor) + Vector(second) * factor


def interpolate_horizontal_curve(points, value):
    """Open centripetal Catmull-Rom interpolation through four controls.

    Endpoint neighbours are extrapolated instead of wrapped. The curve passes
    through every control, remains open, and preserves middle-segment curvature.
    """
    if len(points) != LEVELS:
        raise ValueError("横向曲线必须包含 4 个控制点")
    value = clamp(value)
    if value <= EPS:
        return Vector(points[0])
    if value >= 1.0 - EPS:
        return Vector(points[-1])

    scaled = value * (LEVELS - 1)
    segment = min(int(math.floor(scaled)), LEVELS - 2)
    factor = scaled - segment
    first = Vector(points[segment])
    second = Vector(points[segment + 1])
    before = Vector(points[segment - 1]) if segment > 0 else first * 2.0 - second
    after = Vector(points[segment + 2]) if segment + 2 < LEVELS else second * 2.0 - first

    t0 = 0.0
    t1 = t0 + _centripetal_interval(before, first)
    t2 = t1 + _centripetal_interval(first, second)
    t3 = t2 + _centripetal_interval(second, after)
    target = t1 + (t2 - t1) * factor

    a1 = _knot_lerp(before, first, t0, t1, target)
    a2 = _knot_lerp(first, second, t1, t2, target)
    a3 = _knot_lerp(second, after, t2, t3, target)
    b1 = _knot_lerp(a1, a2, t0, t2, target)
    b2 = _knot_lerp(a2, a3, t1, t3, target)
    return _knot_lerp(b1, b2, t1, t2, target)


def evaluate_patch(grid, u, v):
    horizontal = [interpolate_horizontal_curve(grid[row], u) for row in range(LEVELS)]
    return interpolate_four(horizontal, v)


def resample_polyline(points, count):
    points = [Vector(point) for point in points]
    if count <= 1:
        return [points[0].copy()]
    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + (points[index] - points[index - 1]).length)
    total = cumulative[-1]
    if total <= EPS:
        return [points[0].copy() for _index in range(count)]
    result = []
    source = 0
    for index in range(count):
        target = total * index / (count - 1)
        while source < len(points) - 2 and cumulative[source + 1] < target:
            source += 1
        start, end = cumulative[source], cumulative[source + 1]
        factor = (target - start) / (end - start) if end - start > EPS else 0.0
        result.append(points[source].lerp(points[source + 1], factor))
    return result


def row_width(points):
    return max(
        ((Vector(first) - Vector(second)).length for first in points for second in points),
        default=0.0,
    )


def is_collapsed_free_end(nodes):
    if len(nodes) < 2 or not nodes[0]:
        return False
    maximum_width = max(row_width(row) for row in nodes)
    return maximum_width > EPS and row_width(nodes[-1]) <= max(maximum_width * 0.02, 1.0e-6)


def build_proxy_nodes(control_grid, bones_per_chain, chains):
    maximum_turn = 0.0
    for row in control_grid:
        for index in range(1, LEVELS - 1):
            incoming = safe_normalized(row[index] - row[index - 1], (1.0, 0.0, 0.0))
            outgoing = safe_normalized(row[index + 1] - row[index], incoming)
            maximum_turn = max(
                maximum_turn,
                math.acos(clamp(incoming.dot(outgoing), -1.0, 1.0)),
            )
    curvature_scale = 1.0 + maximum_turn / math.pi
    dense_u = min(2049, max(65, chains * 16 + 1, int(65 * curvature_scale)))
    dense_v = max(49, bones_per_chain * 10 + 1)
    horizontal_rows = []
    for v_index in range(dense_v):
        v = v_index / (dense_v - 1)
        curve = [evaluate_patch(control_grid, u_index / (dense_u - 1), v) for u_index in range(dense_u)]
        horizontal_rows.append(resample_polyline(curve, chains))
    columns = []
    for column in range(chains):
        curve = [horizontal_rows[row][column] for row in range(dense_v)]
        columns.append(resample_polyline(curve, bones_per_chain + 1))
    nodes = [
        [columns[column][row] for column in range(chains)]
        for row in range(bones_per_chain + 1)
    ]
    if is_collapsed_free_end(nodes):
        tip = average(nodes[-1])
        nodes[-1] = [tip.copy() for _column in range(chains)]
    return nodes


def proxy_normals(nodes):
    rows = len(nodes)
    columns = len(nodes[0]) if rows else 0
    collapsed = is_collapsed_free_end(nodes)
    result = []
    for row in range(rows):
        before_row, after_row = max(0, row - 1), min(rows - 1, row + 1)
        row_normals = []
        for column in range(columns):
            before_column, after_column = max(0, column - 1), min(columns - 1, column + 1)
            horizontal_row = row - 1 if collapsed and row == rows - 1 else row
            horizontal = nodes[horizontal_row][after_column] - nodes[horizontal_row][before_column]
            vertical = nodes[after_row][column] - nodes[before_row][column]
            row_normals.append(safe_normalized(horizontal.cross(vertical), (0.0, 1.0, 0.0)))
        result.append(row_normals)
    return result


def proxy_topology(nodes, triangulated=False):
    rows = len(nodes)
    columns = len(nodes[0]) if rows else 0
    collapsed = is_collapsed_free_end(nodes)
    if collapsed:
        vertices = [
            Vector(nodes[row][column])
            for row in range(rows - 1)
            for column in range(columns)
        ]
        vertex_uv = [
            Vector((column / (columns - 1), row / (rows - 1)))
            for row in range(rows - 1)
            for column in range(columns)
        ]
        vertices.append(average(nodes[-1]))
        vertex_uv.append(Vector((0.5, 1.0)))
        tip_index = len(vertices) - 1
    else:
        vertices = [Vector(nodes[row][column]) for row in range(rows) for column in range(columns)]
        vertex_uv = [
            Vector((column / (columns - 1), row / (rows - 1)))
            for row in range(rows)
            for column in range(columns)
        ]
        tip_index = -1

    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            if collapsed and row == rows - 2:
                faces.append((a, b, tip_index))
                continue
            d = (row + 1) * columns + column
            c = d + 1
            if triangulated:
                faces.extend(((a, b, c), (a, c, d)))
            else:
                faces.append((a, b, c, d))
    return vertices, faces, vertex_uv


def proxy_triangles(nodes):
    vertices, triangles, vertex_uv = proxy_topology(nodes, triangulated=True)
    triangle_uv = [tuple(vertex_uv[index] for index in triangle) for triangle in triangles]
    return vertices, triangles, triangle_uv


def make_projector(nodes):
    vertices, triangles, triangle_uv = proxy_triangles(nodes)
    if not triangles:
        raise ValueError("代理网格没有有效面")
    tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    return tree, vertices, triangles, triangle_uv


def barycentric(point, a, b, c):
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= EPS:
        return 1.0, 0.0, 0.0
    second = (d11 * d20 - d01 * d21) / denominator
    third = (d00 * d21 - d01 * d20) / denominator
    return 1.0 - second - third, second, third


def project_uv(point, projector):
    tree, vertices, triangles, triangle_uv = projector
    nearest = tree.find_nearest(Vector(point))
    if nearest is None or nearest[0] is None or nearest[2] is None:
        return 0.0, 0.0
    location, _normal, triangle_index, _distance = nearest
    triangle = triangles[triangle_index]
    weights = barycentric(location, *(vertices[index] for index in triangle))
    uv_values = triangle_uv[triangle_index]
    uv = uv_values[0] * weights[0] + uv_values[1] * weights[1] + uv_values[2] * weights[2]
    return clamp(uv.x), clamp(uv.y)


def compute_weights(u, v, settings, region):
    bones = settings.bones_per_chain
    columns = settings.chains
    separate = settings.group_mode == 'COMPONENTS'
    row_value = clamp(v * bones - 0.5, 0.0, bones - 1.0)
    row0 = int(math.floor(row_value))
    row1 = min(row0 + 1, bones - 1)
    row_t = row_value - row0
    row1_weight = (1.0 - math.cos(math.pi * row_t)) * 0.5
    row0_weight = 1.0 - row1_weight
    column_value = clamp(u) * (columns - 1)
    column0 = int(math.floor(column_value))
    column1 = min(column0 + 1, columns - 1)
    column_t = smoothstep(column_value - column0)
    contributions = (
        (column0, row0, (1.0 - column_t) * row0_weight),
        (column1, row0, column_t * row0_weight),
        (column0, row1, (1.0 - column_t) * row1_weight),
        (column1, row1, column_t * row1_weight),
    )
    weights = {}
    for column, row, weight in contributions:
        name = bone_name(settings.prefix, region, column, row, separate, settings.horizontal_reversed, columns)
        weights[name] = weights.get(name, 0.0) + weight
    parent_weight = 0.0
    if v < 0.15:
        parent_weight = 1.0 - smoothstep(v / 0.15)
        weights = {name: weight * (1.0 - parent_weight) for name, weight in weights.items()}
    weights = {name: weight for name, weight in weights.items() if weight >= MIN_WEIGHT}
    slots = 3 if parent_weight >= MIN_WEIGHT else 4
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:slots])
    if parent_weight >= MIN_WEIGHT:
        weights[PARENT_KEY] = parent_weight
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()} if total > EPS else {}


def apply_weights(mesh_obj, indices, weight_map, settings):
    indices = sorted(set(int(index) for index in indices))
    for group in list(mesh_obj.vertex_groups):
        if group.name.startswith(settings.prefix):
            mesh_obj.vertex_groups.remove(group)
    needed = {
        name for weights in weight_map.values() for name in weights
        if name != PARENT_KEY
    }
    groups = {}
    for name in needed:
        groups[name] = mesh_obj.vertex_groups.get(name) or mesh_obj.vertex_groups.new(name=name)
    parent_group = mesh_obj.vertex_groups.get(settings.parent_bone)
    if parent_group is None:
        parent_group = mesh_obj.vertex_groups.new(name=settings.parent_bone)
    try:
        parent_group.remove(indices)
    except RuntimeError:
        pass
    assignments = 0
    for index in indices:
        for name, weight in weight_map.get(index, {}).items():
            if weight < MIN_WEIGHT:
                continue
            if name == PARENT_KEY:
                parent_group.add([index], weight, 'REPLACE')
            else:
                groups[name].add([index], weight, 'REPLACE')
            assignments += 1
    return assignments


def ensure_modifier(mesh_obj, armature):
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object == armature:
            return False
    modifier = mesh_obj.modifiers.new(name=f"{prefix_stem(settings_name(armature))} Armature", type='ARMATURE')
    modifier.object = armature
    return True


def settings_name(armature):
    return armature.name if armature is not None else "Armature"


def align_bone_roll(bone, target_z):
    target_z = safe_normalized(target_z, (1.0, 0.0, 0.0))
    direction = safe_normalized(bone.tail - bone.head, (0.0, 1.0, 0.0))
    target_z -= direction * target_z.dot(direction)
    if target_z.length_squared <= EPS:
        target_z = direction.cross(Vector((0.0, 0.0, 1.0)))
    if target_z.length_squared <= EPS:
        target_z = direction.cross(Vector((1.0, 0.0, 0.0)))
    bone.align_roll(safe_normalized(target_z, (1.0, 0.0, 0.0)))


def get_or_create_bone_collection(armature_data, name):
    name = str(name).strip()
    if not name:
        return None
    return armature_data.collections.get(name) or armature_data.collections.new(name)


def cleanup_bones(armature, prefix):
    removed = 0
    for bone in list(armature.data.edit_bones):
        if bone.name.startswith(prefix):
            armature.data.edit_bones.remove(bone)
            removed += 1
    return removed


def create_bones(armature, region_nodes, settings):
    edit_bones = armature.data.edit_bones
    parent = edit_bones.get(settings.parent_bone)
    if parent is None:
        raise ValueError(f"找不到父骨骼：{settings.parent_bone}")
    collection = get_or_create_bone_collection(armature.data, settings.bone_collection)
    inverse = armature.matrix_world.inverted()
    normal_to_local = armature.matrix_world.to_3x3().transposed()
    separate = settings.group_mode == 'COMPONENTS'
    created = 0
    for region, nodes in enumerate(region_nodes):
        normals = proxy_normals(nodes)
        for column in range(settings.chains):
            previous = None
            for row in range(settings.bones_per_chain):
                name = bone_name(
                    settings.prefix, region, column, row, separate,
                    settings.horizontal_reversed, settings.chains,
                )
                if edit_bones.get(name) is not None:
                    raise ValueError(f"已存在同名面片骨骼：{name}")
                bone = edit_bones.new(name)
                bone.head = inverse @ nodes[row][column]
                bone.tail = inverse @ nodes[row + 1][column]
                if (bone.tail - bone.head).length_squared <= EPS:
                    raise ValueError("控制网格产生了零长度骨骼，请调整重叠控制点")
                bone.parent = parent if row == 0 else previous
                bone.use_connect = row > 0
                bone.use_deform = True
                if collection is not None:
                    collection.assign(bone)
                target = normals[row][column]
                if not settings.flip_bone_axis:
                    target = -target
                align_bone_roll(bone, normal_to_local @ target)
                previous = bone
                created += 1
    return created


def delete_proxies(prefix, primary_name):
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.get(PROP_KIND) == KIND_PROXY
            and obj.get("auto_sheet_target") == primary_name
            and obj.name.startswith(prefix)
        ):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                bpy.data.meshes.remove(data)
            removed += 1
    collection = bpy.data.collections.get(proxy_collection_name(prefix, primary_name))
    if collection is not None and not collection.objects and not collection.children:
        bpy.data.collections.remove(collection)
    return removed


def create_proxy(nodes, primary, settings, region, bind=True):
    collection = get_or_create_collection(proxy_collection_name(settings.prefix, primary.name))
    vertices, faces, vertex_uv = proxy_topology(nodes, triangulated=False)
    data = bpy.data.meshes.new(f"{settings.prefix}PROXY_P{region:02d}_{primary.name}_Mesh")
    data.from_pydata([tuple(vertex) for vertex in vertices], [], faces)
    data.update()
    obj = bpy.data.objects.new(f"{settings.prefix}PROXY_P{region:02d}_{primary.name}", data)
    collection.objects.link(obj)
    obj.display_type = 'WIRE'
    obj.show_wire = True
    obj.show_in_front = True
    obj.hide_render = True
    obj[PROP_KIND] = KIND_PROXY
    obj["auto_sheet_target"] = primary.name
    if not bind:
        return obj
    indices = list(range(len(data.vertices)))
    weights = {
        index: compute_weights(uv.x, uv.y, settings, region)
        for index, uv in enumerate(vertex_uv)
    }
    apply_weights(obj, indices, weights, settings)
    ensure_modifier(obj, settings.target_armature)
    return obj

class ContextState:
    def __init__(self, context):
        self.active = context.active_object.name if context.active_object else ""
        self.selected = [obj.name for obj in context.selected_objects]
        self.mode = context.active_object.mode if context.active_object else 'OBJECT'

    def restore(self, context):
        try:
            if context.active_object is not None and context.active_object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass
        for obj in context.view_layer.objects:
            try:
                obj.select_set(False)
            except RuntimeError:
                pass
        for name in self.selected:
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.name in context.view_layer.objects:
                obj.select_set(True)
        active = bpy.data.objects.get(self.active)
        if active is not None and active.name in context.view_layer.objects:
            context.view_layer.objects.active = active
            active.select_set(True)
            if self.mode != 'OBJECT':
                try:
                    bpy.ops.object.mode_set(mode=self.mode)
                except RuntimeError:
                    pass


def activate(context, obj, mode='OBJECT'):
    if context.active_object is not None and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for selected in context.selected_objects:
        selected.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    if mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=mode)


def read_regions(primary):
    try:
        regions = json.loads(primary.get(PROP_REGIONS, "[]"))
    except (TypeError, json.JSONDecodeError):
        regions = []
    return regions if isinstance(regions, list) else []


def vertex_world(mesh_obj, index):
    if mesh_obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        return mesh_obj.matrix_world @ bm.verts[index].co
    return mesh_obj.matrix_world @ mesh_obj.data.vertices[index].co


def validate_build(context, settings):
    validate_prefix(settings.prefix)
    primary = primary_from_settings(settings)
    if primary is None:
        raise ValueError("请先从编辑模式选区生成面片控制点")
    armature = settings.target_armature
    if armature is None or armature.type != 'ARMATURE' or armature.name not in context.view_layer.objects:
        raise ValueError("请选择有效的目标骨架")
    if not settings.parent_bone or armature.data.bones.get(settings.parent_bone) is None:
        raise ValueError("请选择有效的父骨骼")
    if settings.parent_bone.startswith(settings.prefix):
        raise ValueError("父骨骼不能使用面片生成前缀")
    if settings.bones_per_chain < 1 or settings.chains < 2:
        raise ValueError("每链骨骼数至少为 1，骨骼链数至少为 2")
    stored_mode = str(primary.get(PROP_MODE, settings.group_mode))
    if stored_mode != settings.group_mode:
        raise ValueError("处理方式已改变，请重新生成控制点")
    grids, _collection = find_controls(primary, settings.prefix, require=True)
    regions = read_regions(primary)
    if len(grids) != len(regions):
        raise ValueError("控制点区域数量与记录选区不一致，请重新生成控制点")
    return primary, armature, grids, regions


def build(context, settings):
    primary, armature, control_grids, regions = validate_build(context, settings)
    control_points = [
        [[obj.matrix_world.translation.copy() for obj in row] for row in grid]
        for grid in control_grids
    ]
    region_nodes = [
        build_proxy_nodes(grid, settings.bones_per_chain, settings.chains)
        for grid in control_points
    ]
    state = ContextState(context)
    created_bones = assignments = 0
    modifier_added = False
    try:
        activate(context, armature, mode='EDIT')
        cleanup_bones(armature, settings.prefix)
        created_bones = create_bones(armature, region_nodes, settings)
        bpy.ops.object.mode_set(mode='OBJECT')

        delete_proxies(str(primary.get(PROP_PREFIX, settings.prefix)), primary.name)
        if str(primary.get(PROP_PREFIX, settings.prefix)) != settings.prefix:
            delete_proxies(settings.prefix, primary.name)

        weights_by_mesh = {}
        indices_by_mesh = {}
        for region_index, (region, nodes) in enumerate(zip(regions, region_nodes)):
            projector = make_projector(nodes)
            for part in region.get("parts", []):
                mesh_obj = bpy.data.objects.get(str(part.get("mesh", "")))
                if mesh_obj is None or mesh_obj.type != 'MESH':
                    raise ValueError("记录的面片网格不存在，请重新生成控制点")
                mesh_weights = weights_by_mesh.setdefault(mesh_obj.name, {})
                mesh_indices = indices_by_mesh.setdefault(mesh_obj.name, set())
                for raw_index in part.get("indices", []):
                    index = int(raw_index)
                    if not 0 <= index < len(mesh_obj.data.vertices):
                        continue
                    u, v = project_uv(vertex_world(mesh_obj, index), projector)
                    mesh_weights[index] = compute_weights(u, v, settings, region_index)
                    mesh_indices.add(index)

        for mesh_name, mesh_weights in weights_by_mesh.items():
            mesh_obj = bpy.data.objects.get(mesh_name)
            activate(context, mesh_obj, mode='OBJECT')
            assignments += apply_weights(mesh_obj, indices_by_mesh[mesh_name], mesh_weights, settings)
            modifier_added |= ensure_modifier(mesh_obj, armature)

        if settings.create_proxy_mesh:
            for region_index, nodes in enumerate(region_nodes):
                create_proxy(nodes, primary, settings, region_index)
        primary[PROP_PREFIX] = settings.prefix
    finally:
        state.restore(context)
    return created_bones, sum(len(value) for value in indices_by_mesh.values()), assignments, modifier_added


def generate(context, settings):
    validate_prefix(settings.prefix)
    skirt = getattr(context.scene, "auto_skirt_settings", None)
    if skirt is not None and skirt.prefix == settings.prefix:
        raise ValueError("面片命名前缀不能与裙子生成器前缀相同")
    meshes, region_components = collect_selection(context, settings)
    primary = meshes[0]
    settings.target_mesh = primary
    region_grids = []
    display_sizes = []
    for components in region_components:
        grid, display_size = fit_region_grid(components)
        region_grids.append(grid)
        display_sizes.append(display_size)
    collection = create_controls(primary, settings, region_grids, max(display_sizes))

    regions = []
    for components in region_components:
        parts_by_mesh = {}
        for component in components:
            parts_by_mesh.setdefault(component["mesh"], set()).update(component["indices"])
        regions.append({
            "parts": [
                {"mesh": name, "indices": sorted(indices)}
                for name, indices in parts_by_mesh.items()
            ]
        })
    target_names = [mesh.name for mesh in meshes]
    primary[PROP_COLLECTION] = collection.name
    primary[PROP_TARGETS] = json.dumps(target_names, separators=(",", ":"))
    primary[PROP_REGIONS] = json.dumps(regions, separators=(",", ":"))
    primary[PROP_PREFIX] = settings.prefix
    primary[PROP_MODE] = settings.group_mode
    for mesh_obj in meshes:
        indices = sorted({
            int(index)
            for region in regions
            for part in region["parts"]
            if part["mesh"] == mesh_obj.name
            for index in part["indices"]
        })
        mesh_obj[PROP_INDICES] = json.dumps(indices, separators=(",", ":"))
    enter_edit(context, settings, collection)
    return sum(len(component["records"]) for components in region_components for component in components), len(region_components)


def delete_generated(context, settings, include_controls=False):
    primary = primary_from_settings(settings)
    if primary is None:
        raise ValueError("找不到面片生成记录")
    state = ContextState(context)
    bones = groups = 0
    try:
        armature = settings.target_armature
        if armature is not None and armature.type == 'ARMATURE':
            activate(context, armature, mode='EDIT')
            bones = cleanup_bones(armature, settings.prefix)
            bpy.ops.object.mode_set(mode='OBJECT')
        for mesh_obj in stored_targets(primary):
            activate(context, mesh_obj, mode='OBJECT')
            for group in list(mesh_obj.vertex_groups):
                if group.name.startswith(settings.prefix):
                    mesh_obj.vertex_groups.remove(group)
                    groups += 1
        delete_proxies(settings.prefix, primary.name)
        if include_controls:
            delete_controls(settings.prefix, primary.name)
            for key in (PROP_COLLECTION, PROP_REGIONS, PROP_TARGETS):
                if key in primary:
                    del primary[key]
    finally:
        state.restore(context)
    return bones, groups

# ---------------------------------------------------------------------------
# Operators and independent panel
# ---------------------------------------------------------------------------


class AUTOSHEET_PG_settings(bpy.types.PropertyGroup):
    target_mesh: PointerProperty(type=bpy.types.Object, poll=poll_mesh, options={'HIDDEN'})
    target_armature: PointerProperty(name="目标骨架", type=bpy.types.Object, poll=poll_armature)
    parent_bone: StringProperty(name="父骨骼", default="")
    bone_collection: StringProperty(
        name="骨骼集合",
        default="Sheet Bones",
        description="使用已有骨骼集合；输入不存在的名称时自动新建",
    )
    prefix: StringProperty(name="命名前缀", default="AutoSheet_")
    group_mode: EnumProperty(
        name="面片组织",
        items=(
            ('MERGED', "合并处理", "多个对象、连通块或皮肤层共用控制网格和骨骼"),
            ('COMPONENTS', "按连通块分别处理", "每个连通块生成独立控制网格和骨骼"),
        ),
        default='MERGED',
    )
    bones_per_chain: IntProperty(name="每链骨骼数", default=6, min=1, max=128)
    chains: IntProperty(name="骨骼链数", default=8, min=2, max=256)
    create_proxy_mesh: BoolProperty(name="生成代理网格", default=False)
    live_proxy_preview: BoolProperty(
        name="实时显示代理网格",
        default=True,
        description="编辑面片控制点时实时显示最终代理表面；仅用于预览",
    )
    flip_bone_axis: BoolProperty(
        name="翻转骨骼轴向",
        default=False,
        description="默认骨骼 +Z 与代理网格法向相反；开启后改为同向",
    )
    local_symmetry: BoolProperty(
        name="局部横向对称",
        default=False,
        description="按控制网格自身的局部中面镜像编辑横向控制点",
    )
    horizontal_reversed: BoolProperty(
        name="反转横向编号",
        default=False,
        description="交换 C00 到最后一列的编号方向，不移动控制点或翻转法向",
    )
    edit_active: BoolProperty(default=False, options={'HIDDEN'})
    edit_mesh_name: StringProperty(default="", options={'HIDDEN'})
    edit_snapshot: StringProperty(default="", options={'HIDDEN'})
    edit_last_positions: StringProperty(default="", options={'HIDDEN'})
    edit_hide_state: StringProperty(default="", options={'HIDDEN'})


class AUTOSHEET_OT_generate_controls(bpy.types.Operator):
    bl_idname = "auto_sheet.generate_controls"
    bl_label = "生成面片控制点"
    bl_description = "从当前选区拟合 4×4 控制网格；支持单层、闭合及实体化双层面片"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        try:
            count, regions = generate(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"已从 {count} 个顶点生成 {regions} 套面片控制网格")
        return {'FINISHED'}


class AUTOSHEET_OT_edit_controls(bpy.types.Operator):
    bl_idname = "auto_sheet.edit_controls"
    bl_label = "修改面片控制点"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        try:
            enter_edit(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "已进入面片控制点编辑")
        return {'FINISHED'}


class AUTOSHEET_OT_finish_edit(bpy.types.Operator):
    bl_idname = "auto_sheet.finish_edit"
    bl_label = "完成面片控制点编辑"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        if not settings.edit_active:
            return {'CANCELLED'}
        apply_local_mirror(context)
        leave_edit(context, settings, reset=False)
        self.report({'INFO'}, "面片控制点已固定")
        return {'FINISHED'}


class AUTOSHEET_OT_cancel_edit(bpy.types.Operator):
    bl_idname = "auto_sheet.cancel_edit"
    bl_label = "取消面片控制点编辑"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        if not settings.edit_active:
            return {'CANCELLED'}
        leave_edit(context, settings, reset=True)
        self.report({'INFO'}, "已取消并恢复控制点")
        return {'FINISHED'}


class AUTOSHEET_OT_reset_edit(bpy.types.Operator):
    bl_idname = "auto_sheet.reset_edit"
    bl_label = "复原面片控制点"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        primary = primary_from_settings(settings)
        if not settings.edit_active or primary is None:
            return {'CANCELLED'}
        try:
            grids, _collection = find_controls(primary, settings.prefix, require=True)
            restore_positions(grids, settings.edit_snapshot)
            settings.edit_last_positions = json.dumps(position_map(grids), separators=(",", ":"))
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "面片控制点已复原")
        return {'FINISHED'}


class AUTOSHEET_OT_reverse_horizontal(bpy.types.Operator):
    bl_idname = "auto_sheet.reverse_horizontal"
    bl_label = "反转横向编号"
    bl_description = "反转骨骼链编号，不移动控制点或翻转代理网格法向"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        settings.horizontal_reversed = not settings.horizontal_reversed
        state = "反向" if settings.horizontal_reversed else "正向"
        self.report({'INFO'}, f"横向编号已设为{state}")
        return {'FINISHED'}


class AUTOSHEET_OT_build(bpy.types.Operator):
    bl_idname = "auto_sheet.build"
    bl_label = "生成面片骨骼和权重"
    bl_description = "从面片控制网格生成代理曲面、相连骨骼链和归一化权重"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        try:
            if settings.edit_active:
                apply_local_mirror(context)
                leave_edit(context, settings, reset=False)
            bones, vertices, assignments, modifier_added = build(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        message = f"已生成 {bones} 根骨骼，为 {vertices} 个顶点写入 {assignments} 次权重"
        if modifier_added:
            message += "；已添加 Armature Modifier"
        self.report({'INFO'}, message)
        return {'FINISHED'}


class AUTOSHEET_OT_delete_controls(bpy.types.Operator):
    bl_idname = "auto_sheet.delete_controls"
    bl_label = "删除面片控制点"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        primary = primary_from_settings(settings)
        if primary is None:
            self.report({'ERROR'}, "找不到面片生成记录")
            return {'CANCELLED'}
        if settings.edit_active:
            leave_edit(context, settings, reset=False)
        prefixes = {settings.prefix, str(primary.get(PROP_PREFIX, settings.prefix))}
        removed = sum(delete_controls(prefix, primary.name) for prefix in prefixes)
        for key in (PROP_COLLECTION, PROP_REGIONS, PROP_TARGETS):
            if key in primary:
                del primary[key]
        self.report({'INFO'}, f"已删除 {removed} 个面片控制点")
        return {'FINISHED'}


class AUTOSHEET_OT_delete_generated(bpy.types.Operator):
    bl_idname = "auto_sheet.delete_generated"
    bl_label = "删除面片生成项"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_sheet_settings
        try:
            bones, groups = delete_generated(context, settings, include_controls=False)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"已删除 {bones} 根面片骨骼和 {groups} 个顶点组")
        return {'FINISHED'}


class AUTOSHEET_PT_main(bpy.types.Panel):
    bl_label = "面片骨骼生成器"
    bl_idname = "AUTOSHEET_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "自动裙骨"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.auto_sheet_settings
        layout.use_property_split = True
        layout.use_property_decorate = False

        target = layout.box()
        target.label(text="当前编辑模式面片")
        target.label(text="最高 Z 识别根部；支持单层、双层和实体化选区", icon='INFO')
        target.prop(settings, "target_armature")
        if settings.target_armature is not None:
            target.prop_search(settings, "parent_bone", settings.target_armature.data, "bones")
        else:
            target.prop(settings, "parent_bone")
        target.prop(settings, "bone_collection")

        options = layout.box()
        options.label(text="垂落面片选项")
        options.prop(settings, "prefix")
        options.prop(settings, "group_mode")
        options.prop(settings, "bones_per_chain")
        options.prop(settings, "chains")
        options.prop(settings, "create_proxy_mesh")
        options.prop(settings, "flip_bone_axis")

        if settings.edit_active:
            editing = layout.box()
            editing.label(text="正在编辑面片控制点", icon='EMPTY_AXIS')
            editing.prop(settings, "local_symmetry")
            editing.prop(settings, "live_proxy_preview")
            editing.prop(settings, "horizontal_reversed")
            editing.label(text="完成 / 取消 / 复原位于 3D 视图顶部")

        layout.operator("auto_sheet.generate_controls", text="生成控制点并编辑", icon='EMPTY_AXIS')
        layout.operator("auto_sheet.edit_controls", text="修改控制点", icon='GREASEPENCIL')
        layout.separator()
        layout.operator("auto_sheet.build", text="生成骨骼和权重", icon='ARMATURE_DATA')
        layout.separator()
        row = layout.row(align=True)
        row.operator("auto_sheet.delete_controls", text="删控制点", icon='TRASH')
        row.operator("auto_sheet.delete_generated", text="删生成项", icon='TRASH')


CLASSES = (
    AUTOSHEET_PG_settings,
    AUTOSHEET_OT_generate_controls,
    AUTOSHEET_OT_edit_controls,
    AUTOSHEET_OT_finish_edit,
    AUTOSHEET_OT_cancel_edit,
    AUTOSHEET_OT_reset_edit,
    AUTOSHEET_OT_reverse_horizontal,
    AUTOSHEET_OT_build,
    AUTOSHEET_OT_delete_controls,
    AUTOSHEET_OT_delete_generated,
    AUTOSHEET_PT_main,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.auto_sheet_settings = PointerProperty(type=AUTOSHEET_PG_settings)
    try:
        bpy.types.VIEW3D_HT_header.remove(draw_header)
    except Exception:
        pass
    bpy.types.VIEW3D_HT_header.append(draw_header)
    install_draw_handler()
    if depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update)


def unregister():
    remove_draw_handler()
    if depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update)
    try:
        bpy.types.VIEW3D_HT_header.remove(draw_header)
    except Exception:
        pass
    if hasattr(bpy.types.Scene, "auto_sheet_settings"):
        del bpy.types.Scene.auto_sheet_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)







