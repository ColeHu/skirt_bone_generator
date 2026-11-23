# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-

bl_info = {
    "name": "裙子骨骼生成器",
    "author": "Akazure",
    "version": (2, 0, 1),
    "blender": (4, 5, 0),
    "location": "3D 视图 > 右侧边栏 > 自动裙骨",
    "description": "根据控制点生成裙子骨骼",
    "category": "Rigging",
}

import importlib
import json
import math
import sys

# Blender keeps package submodules in memory when an add-on is disabled and
# enabled again. Reload both implementation modules so unified_generator's
# configuration guard and patched classes belong to this module generation.
# Otherwise the panel can fall back to the obsolete skirt-type UI after reload.
if "sheet_generator" in locals():
    sheet_generator = importlib.reload(sheet_generator)
else:
    from . import sheet_generator

if "unified_generator" in locals():
    unified_generator = importlib.reload(unified_generator)
else:
    from . import unified_generator

import bmesh
import bpy
import gpu
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from mathutils import Vector
from mathutils.bvhtree import BVHTree


CONTROL_DIRECTIONS = ("Front", "Right", "Back", "Left")
CONTROL_LEVELS = 4
ASYM_CONTROL_CHAIN_COLORS = (
    (0.78, 1.00, 0.10, 1.0),  # lemon
    (0.35, 0.95, 0.12, 1.0),  # chartreuse
    (0.08, 0.88, 0.32, 1.0),  # green
    (0.05, 0.84, 0.62, 1.0),  # teal
    (0.05, 0.72, 1.00, 1.0),  # cyan-blue
    (0.12, 0.42, 1.00, 1.0),  # blue
    (0.34, 0.24, 1.00, 1.0),  # indigo
    (0.60, 0.20, 1.00, 1.0),  # violet
    (0.88, 0.36, 1.00, 1.0),  # orchid
    (0.92, 0.90, 0.55, 1.0),  # cream
)
CONTROL_CHAIN_COLORS = {
    "Front": ASYM_CONTROL_CHAIN_COLORS[0],
    "Right": ASYM_CONTROL_CHAIN_COLORS[2],
    "Back": ASYM_CONTROL_CHAIN_COLORS[5],
    "Left": ASYM_CONTROL_CHAIN_COLORS[7],
}
EPSILON = 1.0e-8
MIN_WEIGHT = 0.001
PARENT_WEIGHT_KEY = "__AUTO_SKIRT_PARENT__"

PROP_COLLECTION = "auto_skirt_control_collection"
PROP_ARMATURE = "auto_skirt_target_armature"
PROP_PARENT = "auto_skirt_parent_bone"
PROP_PREFIX = "auto_skirt_prefix"
PROP_ROWS = "auto_skirt_rows"
PROP_COLUMNS = "auto_skirt_columns"
PROP_BONE_COLLECTION = "auto_skirt_bone_collection"
PROP_INDICES = "auto_skirt_selected_indices"
PROP_RECORDS = "auto_skirt_selection_records"
PROP_TARGET = "auto_skirt_target_mesh"
PROP_TARGETS = "auto_skirt_target_meshes"
PROP_KIND = "auto_skirt_generated_kind"
PROP_SHAPE_MODE = "auto_skirt_shape_mode"
PROP_ASYM_BASE_L3 = "auto_skirt_asym_base_l3"
PROP_ASYM_CONTROL_CHAINS = "auto_skirt_asym_control_chains"
PROP_ASYM_BONE_COUNTS = "auto_skirt_asym_bone_counts"

PROP_PROXY_KIND = "PROXY"
EDIT_KIND_CONTROL = "CONTROL"
EDIT_SNAPSHOT_EPSILON = 1.0e-7
BONE_CHAIN_SMOOTHING_PASSES = 3
_MIRROR_GUARD = False
_LR_SUFFIX_NAMING_ENABLED = False
_LR_SUFFIX_COLUMN_MAP = {}
_SELECTION_MARKER_DRAW_HANDLE = None
_ASYM_CONTROL_RESAMPLE_GUARD = False
_LIVE_PROXY_CACHE = {
    "signature": None,
    "triangles": [],
    "lines": [],
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def smoothstep(x):
    """Clamped cubic smoothstep."""
    x = clamp(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _safe_normalized(vector, fallback):
    vector = Vector(vector)
    if vector.length_squared <= EPSILON:
        vector = Vector(fallback)
    if vector.length_squared <= EPSILON:
        return Vector((0.0, 0.0, 1.0))
    return vector.normalized()


def _prefix_stem(prefix):
    return prefix.rstrip("_") or "AutoSkirt"


def _control_collection_name(prefix, mesh_name):
    return f"{_prefix_stem(prefix)}Controls_{mesh_name}"


def _debug_collection_name(prefix, mesh_name):
    return f"{_prefix_stem(prefix)}Debug_{mesh_name}"


def _proxy_collection_name(prefix, mesh_name):
    return f"{_prefix_stem(prefix)}Proxy_{mesh_name}"


def _control_name(prefix, direction, level):
    return f"{prefix}CTRL_{direction}_L{level}"


def _asym_control_name(prefix, chain, level):
    return f"{prefix}CTRL_C{int(chain):02d}_L{int(level)}"


def _asym_control_color(chain):
    return ASYM_CONTROL_CHAIN_COLORS[
        int(chain) % len(ASYM_CONTROL_CHAIN_COLORS)
    ]


def _plain_bone_name(prefix, column, row):
    """Bone name used by asymmetric skirts; .L/.R is deliberately disabled."""
    return f"{prefix}C{int(column):02d}_R{int(row):02d}"


def _column_average(samples, column):
    points = [Vector(row[column]) for row in samples if column < len(row)]
    return _average_points(points) if points else Vector((0.0, 0.0, 0.0))


def _lr_suffix_sort_key(centers, column):
    point = centers[column]
    return (math.atan2(point.y, point.x), abs(point.x), column)


def _configure_lr_suffix_naming(samples, enabled):
    """Configure temporary symmetric .L/.R bone naming from world-space samples.

    The center is the world origin: negative X columns become .L, positive X
    columns become .R.  Mirrored pairs share one column number so Blender-style
    left/right naming remains symmetric.
    """
    global _LR_SUFFIX_NAMING_ENABLED, _LR_SUFFIX_COLUMN_MAP
    _LR_SUFFIX_NAMING_ENABLED = bool(enabled)
    _LR_SUFFIX_COLUMN_MAP = {}
    if not enabled or not samples or not samples[0]:
        return

    columns = len(samples[0])
    centers = {column: _column_average(samples, column) for column in range(columns)}
    max_abs_x = max((abs(point.x) for point in centers.values()), default=0.0)
    side_epsilon = max(max_abs_x * 1.0e-5, 1.0e-6)

    left_columns = [
        column for column, point in centers.items()
        if point.x < -side_epsilon
    ]
    right_columns = [
        column for column, point in centers.items()
        if point.x > side_epsilon
    ]
    center_columns = [
        column for column, point in centers.items()
        if abs(point.x) <= side_epsilon
    ]

    unused_right = set(right_columns)
    pair_index = 0
    for left_column in sorted(left_columns, key=lambda col: _lr_suffix_sort_key(centers, col)):
        left_center = centers[left_column]
        mirrored = Vector((-left_center.x, left_center.y, left_center.z))
        right_column = None
        if unused_right:
            right_column = min(
                unused_right,
                key=lambda col: (centers[col] - mirrored).length_squared,
            )
            unused_right.remove(right_column)
        _LR_SUFFIX_COLUMN_MAP[left_column] = (pair_index, ".L")
        if right_column is not None:
            _LR_SUFFIX_COLUMN_MAP[right_column] = (pair_index, ".R")
        pair_index += 1

    for right_column in sorted(unused_right, key=lambda col: _lr_suffix_sort_key(centers, col)):
        _LR_SUFFIX_COLUMN_MAP[right_column] = (pair_index, ".R")
        pair_index += 1

    # Centerline columns are left unsuffixed and keep their original column
    # index, avoiding false .L/.R labels for exact X=0 samples.
    for center_column in center_columns:
        _LR_SUFFIX_COLUMN_MAP[center_column] = (center_column, "")


def _bone_name(prefix, column, row):
    if _LR_SUFFIX_NAMING_ENABLED:
        mapped = _LR_SUFFIX_COLUMN_MAP.get(column)
        if mapped is not None:
            mapped_column, suffix = mapped
            return f"{prefix}C{mapped_column:02d}_R{row:02d}{suffix}"
    return f"{prefix}C{column:02d}_R{row:02d}"


def _temporary_bone_name(prefix, column, row):
    """Physical build name used before optional .L/.R suffix assignment."""
    return f"{prefix}__BUILD_C{column:03d}_R{row:03d}"



def _poll_mesh(_self, obj):
    return obj is not None and obj.type == 'MESH'


def _poll_armature(_self, obj):
    return obj is not None and obj.type == 'ARMATURE'


def _resolve_target_mesh(context, settings):
    obj = settings.target_mesh
    if obj is None:
        active = context.active_object
        if active is not None and active.type == 'MESH':
            obj = active
    return obj


def _stored_target_meshes(primary_mesh):
    """Load the mesh set that shares one generated control cage."""
    if primary_mesh is None or primary_mesh.type != 'MESH':
        return []
    try:
        names = json.loads(primary_mesh.get(PROP_TARGETS, "[]"))
    except (TypeError, json.JSONDecodeError):
        names = []
    result = []
    for name in names:
        obj = bpy.data.objects.get(str(name))
        if obj is not None and obj.type == 'MESH' and obj not in result:
            result.append(obj)
    if primary_mesh not in result:
        result.insert(0, primary_mesh)
    return result


def _resolve_target_meshes(context, settings, prefer_edit_selection=False):
    """Resolve a shared cage target set, preferring multi-object Edit Mode."""
    selected_meshes = [
        obj for obj in context.selected_objects
        if obj.type == 'MESH' and obj.name in context.view_layer.objects
    ]
    edit_meshes = [obj for obj in selected_meshes if obj.mode == 'EDIT']
    primary = _resolve_target_mesh(context, settings)

    if edit_meshes:
        active = context.active_object
        if active is not None and active.type == 'MESH' and active in edit_meshes:
            primary = active
        elif primary not in edit_meshes:
            primary = edit_meshes[0]
        ordered = [primary] + [obj for obj in edit_meshes if obj != primary]
        return ordered

    if primary is not None and primary.type == 'MESH':
        stored = _stored_target_meshes(primary)
        if len(stored) > 1 or prefer_edit_selection:
            return stored

    if selected_meshes:
        if primary in selected_meshes:
            return [primary] + [obj for obj in selected_meshes if obj != primary]
        return selected_meshes
    return [primary] if primary is not None and primary.type == 'MESH' else []


def _validate_prefix(prefix):
    if not prefix or not prefix.strip("_ "):
        raise ValueError("Prefix 不能为空，也不能只包含下划线")


def _average_points(points):
    if not points:
        return Vector((0.0, 0.0, 0.0))
    total = Vector((0.0, 0.0, 0.0))
    for point in points:
        total += point
    return total / len(points)


def _scene_has_object(context, obj):
    return obj is not None and obj.name in context.view_layer.objects


class _ContextState:
    """Save and restore the user's active object, selection, and object mode."""

    def __init__(self, context):
        self.active_name = context.active_object.name if context.active_object else ""
        self.selected_names = [obj.name for obj in context.selected_objects]
        active = context.active_object
        self.mode = active.mode if active is not None else 'OBJECT'

    def restore(self, context):
        try:
            active = context.view_layer.objects.active
            if active is not None and active.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass

        for obj in context.view_layer.objects:
            try:
                obj.select_set(False)
            except RuntimeError:
                pass

        for name in self.selected_names:
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.name in context.view_layer.objects:
                try:
                    obj.select_set(True)
                except RuntimeError:
                    pass

        active = bpy.data.objects.get(self.active_name)
        if active is None or active.name not in context.view_layer.objects:
            return
        context.view_layer.objects.active = active
        try:
            active.select_set(True)
        except RuntimeError:
            pass

        if self.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode=self.mode)
            except RuntimeError:
                # Restoring selection is more important than failing the operator.
                pass


def _activate_object(context, obj, mode='OBJECT'):
    active = context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for selected in context.selected_objects:
        selected.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    if mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=mode)


# ---------------------------------------------------------------------------
# Selection collection and persistence
# ---------------------------------------------------------------------------


def get_selected_vertices(mesh_obj):
    """
    Return records containing vertex index, local position, and world position.

    Selection is intentionally read only from Mesh Edit Mode. An empty result is
    never interpreted as "use the whole object".
    """
    if mesh_obj is None or mesh_obj.type != 'MESH' or mesh_obj.mode != 'EDIT':
        return []

    bm = bmesh.from_edit_mesh(mesh_obj.data)
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    bm.normal_update()
    matrix_world = mesh_obj.matrix_world
    normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
    records = []
    for vert in bm.verts:
        if vert.select and not vert.hide:
            local = vert.co.copy()
            world = matrix_world @ local
            normal = _safe_normalized(
                normal_matrix @ vert.normal,
                Vector((0.0, 0.0, 1.0)),
            )
            records.append({
                "index": int(vert.index),
                "local": local,
                "world": world,
                "normal": normal,
            })
    records.sort(key=lambda item: item["index"])
    return records


def _store_selection_records(mesh_obj, records):
    compact_records = []
    indices = []
    for item in records:
        local = item["local"]
        world = item["world"]
        index = int(item["index"])
        indices.append(index)
        normal = item.get("normal", Vector((0.0, 0.0, 1.0)))
        compact_records.append([
            index,
            float(local.x), float(local.y), float(local.z),
            float(world.x), float(world.y), float(world.z),
            float(normal.x), float(normal.y), float(normal.z),
        ])
    mesh_obj[PROP_INDICES] = json.dumps(indices, separators=(",", ":"))
    mesh_obj[PROP_RECORDS] = json.dumps(compact_records, separators=(",", ":"))


def _load_selected_indices(mesh_obj):
    raw = mesh_obj.get(PROP_INDICES, "")
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    vertex_count = len(mesh_obj.data.vertices)
    return sorted({
        int(value) for value in values
        if isinstance(value, (int, float)) and 0 <= int(value) < vertex_count
    })


def _current_or_stored_indices(mesh_obj):
    current = get_selected_vertices(mesh_obj)
    if current:
        _store_selection_records(mesh_obj, current)
        return [item["index"] for item in current]
    return _load_selected_indices(mesh_obj)


def _store_asym_base_l3(mesh_obj, controls):
    data = {
        direction: [
            float(controls[direction][-1].x),
            float(controls[direction][-1].y),
            float(controls[direction][-1].z),
        ]
        for direction in CONTROL_DIRECTIONS
    }
    mesh_obj[PROP_ASYM_BASE_L3] = json.dumps(data, separators=(",", ":"))


def _load_asym_base_l3(mesh_obj):
    raw = mesh_obj.get(PROP_ASYM_BASE_L3, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    result = {}
    for direction in CONTROL_DIRECTIONS:
        value = data.get(direction)
        if isinstance(value, list) and len(value) == 3:
            try:
                result[direction] = Vector((
                    float(value[0]), float(value[1]), float(value[2]),
                ))
            except (TypeError, ValueError):
                pass
    return result

def _asym_count_property_name(chain):
    return f"asym_chain_bones_{int(chain):02d}"


def _get_asym_anchor_bone_counts(settings, chain_count=None):
    count = int(chain_count if chain_count is not None else settings.asym_control_chain_count)
    return [
        clamp(int(getattr(settings, _asym_count_property_name(chain))), 1, 64)
        for chain in range(count)
    ]


def _set_asym_anchor_bone_counts(settings, values, chain_count=None):
    count = int(chain_count if chain_count is not None else settings.asym_control_chain_count)
    values = list(values)
    fallback = clamp(int(settings.asym_default_bone_count), 1, 64)
    for chain in range(count):
        value = values[chain] if chain < len(values) else fallback
        setattr(
            settings,
            _asym_count_property_name(chain),
            int(clamp(round(float(value)), 1, 64)),
        )


def _load_stored_asym_bone_counts(mesh_obj, fallback=None):
    try:
        values = json.loads(mesh_obj.get(PROP_ASYM_BONE_COUNTS, "[]"))
    except (TypeError, json.JSONDecodeError):
        values = []
    result = [
        int(clamp(round(float(value)), 1, 64))
        for value in values
        if isinstance(value, (int, float))
    ]
    return result or list(fallback or [])

def _store_mesh_metadata(mesh_obj, settings, collection_name, target_meshes=None):
    mesh_obj[PROP_COLLECTION] = collection_name
    mesh_obj[PROP_ARMATURE] = (
        settings.target_armature.name if settings.target_armature else ""
    )
    mesh_obj[PROP_PARENT] = settings.parent_bone
    mesh_obj[PROP_PREFIX] = settings.prefix
    mesh_obj[PROP_ROWS] = settings.rows
    mesh_obj[PROP_COLUMNS] = settings.columns
    mesh_obj[PROP_BONE_COLLECTION] = settings.bone_collection
    mesh_obj[PROP_SHAPE_MODE] = settings.skirt_shape_mode
    if settings.skirt_shape_mode == 'ASYMMETRIC':
        mesh_obj[PROP_ASYM_CONTROL_CHAINS] = int(settings.asym_control_chain_count)
        mesh_obj[PROP_ASYM_BONE_COUNTS] = json.dumps(
            _get_asym_anchor_bone_counts(settings),
            separators=(",", ":"),
        )
    if target_meshes is not None:
        mesh_obj[PROP_TARGETS] = json.dumps(
            [obj.name for obj in target_meshes if obj is not None and obj.type == 'MESH'],
            separators=(",", ":"),
        )


# ---------------------------------------------------------------------------
# Collections and generated object cleanup
# ---------------------------------------------------------------------------


def get_or_create_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    scene_children = bpy.context.scene.collection.children
    if collection.name not in scene_children:
        scene_children.link(collection)
    return collection


def delete_generated_controls(prefix, mesh_name):
    """Delete only marked control objects whose names match the requested prefix."""
    object_prefix = f"{prefix}CTRL_"
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name.startswith(object_prefix)
            and obj.get(PROP_TARGET) == mesh_name
            and obj.get(PROP_KIND) == EDIT_KIND_CONTROL
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1

    for collection in list(bpy.data.collections):
        if (
            collection.name == _control_collection_name(prefix, mesh_name)
            and len(collection.objects) == 0
            and len(collection.children) == 0
        ):
            bpy.data.collections.remove(collection)
    return removed


def _delete_debug_objects(prefix, mesh_name):
    object_prefix = f"{prefix}DEBUG_"
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name.startswith(object_prefix)
            and obj.get(PROP_TARGET) == mesh_name
            and obj.get(PROP_KIND) == "DEBUG"
        ):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                bpy.data.curves.remove(data)
            removed += 1

    for collection in list(bpy.data.collections):
        if (
            collection.name == _debug_collection_name(prefix, mesh_name)
            and len(collection.objects) == 0
            and len(collection.children) == 0
        ):
            bpy.data.collections.remove(collection)
    return removed


def _delete_proxy_objects(prefix, mesh_name):
    object_prefix = f"{prefix}PROXY_"
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name.startswith(object_prefix)
            and obj.get(PROP_TARGET) == mesh_name
            and obj.get(PROP_KIND) == PROP_PROXY_KIND
        ):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                bpy.data.meshes.remove(data)
            removed += 1

    for collection in list(bpy.data.collections):
        if (
            collection.name == _proxy_collection_name(prefix, mesh_name)
            and len(collection.objects) == 0
            and len(collection.children) == 0
        ):
            bpy.data.collections.remove(collection)
    return removed


# ---------------------------------------------------------------------------
# Control point edit session
# ---------------------------------------------------------------------------


def _controls_are_asymmetric(control_objects):
    return isinstance(control_objects, (list, tuple))


def _flatten_control_objects(control_objects):
    if _controls_are_asymmetric(control_objects):
        return [
            obj
            for chain in control_objects
            for obj in chain
            if obj is not None
        ]
    return [
        control_objects[direction][level]
        for direction in CONTROL_DIRECTIONS
        for level in range(CONTROL_LEVELS)
        if control_objects[direction][level] is not None
    ]


def _apply_control_colors(control_objects):
    if _controls_are_asymmetric(control_objects):
        for chain, objects in enumerate(control_objects):
            for obj in objects:
                if obj is not None:
                    obj.color = _asym_control_color(chain)
        return
    for direction in CONTROL_DIRECTIONS:
        for obj in control_objects[direction]:
            if obj is not None:
                obj.color = CONTROL_CHAIN_COLORS[direction]


def _control_position_map(control_objects):
    return {
        obj.name: [float(obj.location.x), float(obj.location.y), float(obj.location.z)]
        for obj in _flatten_control_objects(control_objects)
    }


def _json_position_map(control_objects):
    return json.dumps(_control_position_map(control_objects), separators=(",", ":"))


def _parse_json_dict(raw):
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _vector_from_json(value):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return Vector((float(value[0]), float(value[1]), float(value[2])))
    except (TypeError, ValueError):
        return None


def _set_control_object_locked(obj, locked):
    obj.hide_select = bool(locked)
    obj.lock_location = (bool(locked), bool(locked), bool(locked))
    obj.lock_rotation = (bool(locked), bool(locked), bool(locked))
    obj.lock_scale = (bool(locked), bool(locked), bool(locked))


def _set_controls_locked(control_objects, locked):
    for obj in _flatten_control_objects(control_objects):
        _set_control_object_locked(obj, locked)
        obj.show_in_front = True


def _control_edit_collection(context, settings):
    mesh_obj = _resolve_target_mesh(context, settings)
    if mesh_obj is None and settings.control_edit_mesh_name:
        mesh_obj = bpy.data.objects.get(settings.control_edit_mesh_name)
    if mesh_obj is None or mesh_obj.type != 'MESH':
        raise ValueError("Target Mesh 不存在")
    collection_name = str(
        mesh_obj.get(
            PROP_COLLECTION,
            _control_collection_name(settings.prefix, mesh_obj.name),
        )
    )
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        raise ValueError("找不到控制点集合，请先生成控制点")
    controls = find_control_points(collection, settings.prefix, require_all=True)
    _apply_control_colors(controls)
    return mesh_obj, collection, controls


def _snapshot_control_centers(snapshot_raw):
    data = _parse_json_dict(snapshot_raw)
    centers = []
    for level in range(CONTROL_LEVELS):
        suffix = f"_L{level}"
        points = [
            point
            for key, value in data.items()
            if "CTRL_" in str(key) and str(key).endswith(suffix)
            if (point := _vector_from_json(value)) is not None
        ]
        centers.append(
            _average_points(points) if points else Vector((0.0, 0.0, 0.0))
        )
    return centers

def _restore_control_positions(control_objects, snapshot_raw):
    data = _parse_json_dict(snapshot_raw)
    for obj in _flatten_control_objects(control_objects):
        point = _vector_from_json(data.get(obj.name))
        if point is not None:
            obj.location = point


def _capture_hide_select_state(context, control_objects):
    control_names = {obj.name for obj in _flatten_control_objects(control_objects)}
    state = {
        obj.name: bool(obj.hide_select)
        for obj in context.view_layer.objects
        if obj.name not in control_names
    }
    return json.dumps(state, separators=(",", ":"))


def _restore_hide_select_state(context, settings):
    data = _parse_json_dict(settings.control_edit_hide_select_state)
    for name, hidden in data.items():
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.name in context.view_layer.objects:
            obj.hide_select = bool(hidden)


def _select_only_controls(context, control_objects):
    if context.active_object is not None and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    controls = _flatten_control_objects(control_objects)
    control_names = {obj.name for obj in controls}
    for obj in context.view_layer.objects:
        try:
            obj.select_set(bool(controls) and obj == controls[0])
        except RuntimeError:
            pass
    if controls:
        context.view_layer.objects.active = controls[0]


def _enter_control_edit_mode(context, settings, collection=None):
    mesh_obj = _resolve_target_mesh(context, settings)
    if mesh_obj is None or mesh_obj.type != 'MESH':
        raise ValueError("Target Mesh 不存在")
    if settings.target_mesh is None:
        settings.target_mesh = mesh_obj
    if collection is None:
        collection_name = str(
            mesh_obj.get(
                PROP_COLLECTION,
                _control_collection_name(settings.prefix, mesh_obj.name),
            )
        )
        collection = bpy.data.collections.get(collection_name)
    if collection is None:
        raise ValueError("找不到控制点集合，请先生成控制点")

    controls = find_control_points(collection, settings.prefix, require_all=True)
    settings.control_edit_mesh_name = mesh_obj.name
    settings.control_edit_snapshot = _json_position_map(controls)
    settings.control_edit_last_positions = settings.control_edit_snapshot
    settings.control_edit_hide_select_state = _capture_hide_select_state(context, controls)
    settings.control_edit_active = True
    _LIVE_PROXY_CACHE["signature"] = None

    if context.active_object is not None and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    for obj in _flatten_control_objects(controls):
        obj.hide_set(False, view_layer=context.view_layer)
    control_names = {obj.name for obj in _flatten_control_objects(controls)}
    for obj in context.view_layer.objects:
        if obj.name not in control_names:
            obj.hide_select = True
    _set_controls_locked(controls, False)
    _select_only_controls(context, controls)
    for area in context.screen.areas if context.screen else []:
        area.tag_redraw()
    return len(control_names)


def _leave_control_edit_mode(context, settings, reset=False):
    try:
        mesh_obj, _collection, controls = _control_edit_collection(context, settings)
        if reset:
            _restore_control_positions(controls, settings.control_edit_snapshot)
        _set_controls_locked(controls, True)
        for obj in _flatten_control_objects(controls):
            try:
                obj.select_set(False)
            except RuntimeError:
                pass
    except (ValueError, RuntimeError):
        mesh_obj = bpy.data.objects.get(settings.control_edit_mesh_name)
    _restore_hide_select_state(context, settings)

    if mesh_obj is not None and mesh_obj.name in context.view_layer.objects:
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass
        context.view_layer.objects.active = mesh_obj
        try:
            mesh_obj.select_set(True)
        except RuntimeError:
            pass

    settings.control_edit_active = False
    settings.control_edit_last_positions = ""
    settings.control_edit_snapshot = ""
    settings.control_edit_hide_select_state = ""
    _LIVE_PROXY_CACHE["signature"] = None
    _LIVE_PROXY_CACHE["triangles"] = []
    _LIVE_PROXY_CACHE["lines"] = []
    for area in context.screen.areas if context.screen else []:
        area.tag_redraw()


def _position_delta_squared(a, b):
    va = _vector_from_json(a)
    vb = _vector_from_json(b)
    if va is None or vb is None:
        return 0.0
    return (va - vb).length_squared


def _changed_control_names(current, previous):
    changed = set()
    for name, value in current.items():
        if _position_delta_squared(value, previous.get(name)) > EDIT_SNAPSHOT_EPSILON:
            changed.add(name)
    return changed


def _preferred_changed_control_name(context, controls, changed, previous):
    """Choose the user's edited control even when a handler has stale context.

    A depsgraph update can be delivered after Blender has changed the active
    object, or after the mirror target itself has caused a second update.  The
    old implementation then fell back to the hard-coded first item in a pair,
    which made the mirror appear to work intermittently.  Prefer the active
    changed control, then a sole selected changed control, and finally the
    control with the largest local-space displacement.
    """
    controls_by_name = {
        obj.name: obj for obj in _flatten_control_objects(controls)
    }
    active = context.active_object
    if active is not None and active.name in changed:
        return active.name

    selected = [
        name for name in changed
        if name in controls_by_name and controls_by_name[name].select_get()
    ]
    if len(selected) == 1:
        return selected[0]

    if changed:
        return max(
            changed,
            key=lambda name: _position_delta_squared(
                list(controls_by_name[name].location),
                previous.get(name),
            ) if name in controls_by_name else 0.0,
        )
    return ""


def _mirror_location(source, axis, center_value):
    point = Vector(source)
    if axis == 'X':
        point.x = 2.0 * center_value - point.x
    elif axis == 'Y':
        point.y = 2.0 * center_value - point.y
    return point


def _choose_mirror_source(name_a, name_b, changed, active_name):
    if active_name == name_a and name_a in changed:
        return name_a, name_b
    if active_name == name_b and name_b in changed:
        return name_b, name_a
    a_changed = name_a in changed
    b_changed = name_b in changed
    if a_changed and not b_changed:
        return name_a, name_b
    if b_changed and not a_changed:
        return name_b, name_a
    if a_changed and b_changed:
        return name_a, name_b
    return None, None


def _apply_mirror_pair(control_objects, current, changed, active_name, direction_a, direction_b, level, axis, center_value):
    obj_a = control_objects[direction_a][level]
    obj_b = control_objects[direction_b][level]
    source_name, target_name = _choose_mirror_source(
        obj_a.name,
        obj_b.name,
        changed,
        active_name,
    )
    if source_name is None:
        return False
    source_obj = bpy.data.objects.get(source_name)
    target_obj = bpy.data.objects.get(target_name)
    if source_obj is None or target_obj is None:
        return False
    target_obj.location = _mirror_location(source_obj.location, axis, center_value)
    current[target_obj.name] = [
        float(target_obj.location.x),
        float(target_obj.location.y),
        float(target_obj.location.z),
    ]
    return True


def _dynamic_mirror_pairs(control_objects, snapshot_raw, level, axis):
    snapshot = _parse_json_dict(snapshot_raw)
    points = {}
    for chain, objects in enumerate(control_objects):
        point = _vector_from_json(snapshot.get(objects[level].name))
        if point is not None:
            points[chain] = point
    if not points:
        return [], 0.0
    center = _average_points(points.values())
    center_value = center.x if axis == 'X' else center.y
    unpaired = set(points)
    pairs = []
    if len(unpaired) % 2:
        self_chain = min(
            unpaired,
            key=lambda chain: abs(
                (points[chain].x if axis == 'X' else points[chain].y)
                - center_value
            ),
        )
        pairs.append((self_chain, self_chain))
        unpaired.remove(self_chain)
    while unpaired:
        first = min(unpaired)
        unpaired.remove(first)
        reflected = _mirror_location(points[first], axis, center_value)
        second = min(
            unpaired,
            key=lambda chain: (points[chain] - reflected).length_squared,
        )
        unpaired.remove(second)
        pairs.append((first, second))
    return pairs, center_value


def _apply_dynamic_mirror_axis(
    control_objects,
    current,
    changed,
    active_name,
    level,
    axis,
    snapshot_raw,
):
    mirrored_names = set()
    pairs, center_value = _dynamic_mirror_pairs(
        control_objects, snapshot_raw, level, axis
    )
    for first, second in pairs:
        obj_a = control_objects[first][level]
        obj_b = control_objects[second][level]
        if first == second:
            if obj_a.name not in changed:
                continue
            obj_a.location = _mirror_location(
                obj_a.location,
                axis,
                center_value,
            ).lerp(obj_a.location, 0.5)
            if axis == 'X':
                obj_a.location.x = center_value
            else:
                obj_a.location.y = center_value
            current[obj_a.name] = list(obj_a.location)
            mirrored_names.add(obj_a.name)
            continue
        source_name, target_name = _choose_mirror_source(
            obj_a.name,
            obj_b.name,
            changed,
            active_name,
        )
        if source_name is None:
            continue
        source_obj = bpy.data.objects.get(source_name)
        target_obj = bpy.data.objects.get(target_name)
        if source_obj is None or target_obj is None:
            continue
        target_obj.location = _mirror_location(
            source_obj.location,
            axis,
            center_value,
        )
        current[target_obj.name] = list(target_obj.location)
        mirrored_names.add(target_obj.name)
    return mirrored_names

def _apply_control_edit_mirror(context, scene=None):
    global _MIRROR_GUARD
    if _MIRROR_GUARD:
        return
    scene = scene or context.scene
    if scene is None or not hasattr(scene, "auto_skirt_settings"):
        return
    settings = scene.auto_skirt_settings
    if not settings.control_edit_active:
        return
    if not settings.control_mirror_x and not settings.control_mirror_y:
        return

    try:
        _mesh_obj, _collection, controls = _control_edit_collection(context, settings)
    except (ValueError, RuntimeError):
        return

    current = _control_position_map(controls)
    previous = _parse_json_dict(settings.control_edit_last_positions)
    changed = _changed_control_names(current, previous)
    if not changed:
        return

    centers = _snapshot_control_centers(settings.control_edit_snapshot)
    active_name = _preferred_changed_control_name(
        context, controls, changed, previous
    )
    _MIRROR_GUARD = True
    try:
        if _controls_are_asymmetric(controls):
            for level in range(CONTROL_LEVELS):
                if settings.control_mirror_x:
                    changed.update(_apply_dynamic_mirror_axis(
                        controls,
                        current,
                        changed,
                        active_name,
                        level,
                        'X',
                        settings.control_edit_snapshot,
                    ))
                if settings.control_mirror_y:
                    changed.update(_apply_dynamic_mirror_axis(
                        controls,
                        current,
                        changed,
                        active_name,
                        level,
                        'Y',
                        settings.control_edit_snapshot,
                    ))
            settings.control_edit_last_positions = json.dumps(
                _control_position_map(controls),
                separators=(",", ":"),
            )
            return

        any_changed = False
        for level in range(CONTROL_LEVELS):
            center = centers[level]
            if settings.control_mirror_x:
                any_changed |= _apply_mirror_pair(
                    controls,
                    current,
                    changed,
                    active_name,
                    "Left",
                    "Right",
                    level,
                    'X',
                    center.x,
                )
            if settings.control_mirror_y:
                any_changed |= _apply_mirror_pair(
                    controls,
                    current,
                    changed,
                    active_name,
                    "Front",
                    "Back",
                    level,
                    'Y',
                    center.y,
                )
            # Front/Back lie on the X mirror plane and Left/Right lie on the
            # Y mirror plane.  Keep their on-plane coordinate fixed, so a
            # move cannot silently break the enabled symmetry.
            if settings.control_mirror_x:
                for direction in ("Front", "Back"):
                    obj = controls[direction][level]
                    if obj.name in changed:
                        obj.location.x = center.x
                        any_changed = True
            if settings.control_mirror_y:
                for direction in ("Left", "Right"):
                    obj = controls[direction][level]
                    if obj.name in changed:
                        obj.location.y = center.y
                        any_changed = True
        if any_changed:
            current = _control_position_map(controls)
        settings.control_edit_last_positions = json.dumps(current, separators=(",", ":"))
    finally:
        _MIRROR_GUARD = False


@persistent
def _auto_skirt_depsgraph_update(scene, depsgraph):
    del depsgraph
    if scene is None or not hasattr(scene, "auto_skirt_settings"):
        return
    if scene.auto_skirt_settings.control_edit_active:
        _apply_control_edit_mirror(bpy.context, scene)


def _draw_control_edit_header(self, context):
    scene = context.scene
    if scene is None or not hasattr(scene, "auto_skirt_settings"):
        return
    settings = scene.auto_skirt_settings
    if not settings.control_edit_active:
        return
    layout = self.layout
    layout.separator_spacer()
    row = layout.row(align=True)
    row.label(text="控制点编辑", icon='EMPTY_AXIS')
    row.prop(settings, "control_mirror_x", text="X 镜像")
    row.prop(settings, "control_mirror_y", text="Y 镜像")
    row.prop(settings, "control_live_proxy_preview", text="实时代理")
    row.operator("auto_skirt.reset_control_edit", text="复原", icon='LOOP_BACK')
    row.operator("auto_skirt.cancel_control_edit", text="取消", icon='CANCEL')
    row.operator("auto_skirt.finish_control_edit", text="完成", icon='CHECKMARK')


def _control_edit_selected_world_positions(scene):
    """Return the persisted target-vertex positions for the edit overlay."""
    if scene is None or not hasattr(scene, "auto_skirt_settings"):
        return []
    settings = scene.auto_skirt_settings
    if not settings.control_edit_active or not settings.show_mesh_points:
        return []

    primary_mesh = bpy.data.objects.get(settings.control_edit_mesh_name)
    if primary_mesh is None:
        primary_mesh = settings.target_mesh
    if primary_mesh is None or primary_mesh.type != 'MESH':
        return []

    positions = []
    for mesh_obj in _stored_target_meshes(primary_mesh):
        indices = _load_selected_indices(mesh_obj)
        matrix_world = mesh_obj.matrix_world
        positions.extend(
            matrix_world @ mesh_obj.data.vertices[index].co
            for index in indices
            if 0 <= index < len(mesh_obj.data.vertices)
        )
    return positions


def _control_edit_control_world_positions(scene):
    """Return editable control positions grouped by their colored chain."""
    if scene is None or not hasattr(scene, "auto_skirt_settings"):
        return {}
    settings = scene.auto_skirt_settings
    if not settings.control_edit_active:
        return {}
    primary_mesh = bpy.data.objects.get(settings.control_edit_mesh_name)
    if primary_mesh is None:
        primary_mesh = settings.target_mesh
    if primary_mesh is None or primary_mesh.type != 'MESH':
        return {}
    collection_name = str(primary_mesh.get(
        PROP_COLLECTION,
        _control_collection_name(settings.prefix, primary_mesh.name),
    ))
    collection = bpy.data.collections.get(collection_name)
    controls = find_control_points(collection, settings.prefix, require_all=False)
    if _controls_are_asymmetric(controls):
        return {
            f"C{chain:02d}": [
                obj.matrix_world.translation.copy()
                for obj in objects
                if obj is not None
            ]
            for chain, objects in enumerate(controls)
        }
    return {
        direction: [
            obj.matrix_world.translation.copy()
            for obj in controls.get(direction, [])
            if obj is not None
        ]
        for direction in CONTROL_DIRECTIONS
    }


def _proxy_draw_geometry(samples, closed_loop=True):
    """Convert a sampled grid to translucent triangles and visible grid lines."""
    rows = len(samples)
    columns = len(samples[0]) if rows else 0
    if rows < 2 or columns < 2:
        return [], []
    triangles = []
    lines = []
    column_limit = columns if closed_loop else columns - 1
    for row in range(rows):
        for column in range(column_limit):
            following = (column + 1) % columns
            lines.extend((samples[row][column], samples[row][following]))
    for row in range(rows - 1):
        for column in range(columns):
            lines.extend((samples[row][column], samples[row + 1][column]))
        for column in range(column_limit):
            following = (column + 1) % columns
            first = samples[row][column]
            second = samples[row][following]
            third = samples[row + 1][following]
            fourth = samples[row + 1][column]
            triangles.extend((first, second, third, first, third, fourth))
    return triangles, lines


def _live_proxy_signature(settings, primary_mesh, controls):
    values = [
        primary_mesh.name,
        settings.skirt_shape_mode,
        int(settings.rows),
        int(settings.columns),
        round(float(settings.bone_inset), 6),
        round(float(settings.slice_thickness), 6),
        settings.fit_mode,
        round(float(settings.ring_fit_strength), 6),
        bool(settings.longitudinal_smoothing),
        bool(settings.smooth_bone_chains),
    ]
    if _controls_are_asymmetric(controls):
        values.append(int(settings.asym_control_chain_count))
        values.extend(_get_asym_anchor_bone_counts(
            settings,
            len(controls),
        ))
        for chain in controls:
            for obj in chain:
                point = obj.matrix_world.translation
                values.extend(round(float(component), 6) for component in point)
    else:
        for direction in CONTROL_DIRECTIONS:
            for obj in controls[direction]:
                point = obj.matrix_world.translation
                values.extend(round(float(component), 6) for component in point)
    return tuple(values)


def _control_edit_live_proxy_geometry(scene):
    """Build and cache the same proxy surface used by the final skirt build."""
    settings = getattr(scene, "auto_skirt_settings", None)
    if (
        settings is None
        or not settings.control_edit_active
        or not settings.control_live_proxy_preview
    ):
        _LIVE_PROXY_CACHE["signature"] = None
        _LIVE_PROXY_CACHE["triangles"] = []
        _LIVE_PROXY_CACHE["lines"] = []
        return [], []

    primary_mesh = bpy.data.objects.get(settings.control_edit_mesh_name)
    if primary_mesh is None:
        primary_mesh = settings.target_mesh
    if primary_mesh is None or primary_mesh.type != 'MESH':
        return [], []
    collection_name = str(primary_mesh.get(
        PROP_COLLECTION,
        _control_collection_name(settings.prefix, primary_mesh.name),
    ))
    collection = bpy.data.collections.get(collection_name)
    controls = find_control_points(collection, settings.prefix, require_all=True)
    signature = _live_proxy_signature(settings, primary_mesh, controls)
    if signature == _LIVE_PROXY_CACHE["signature"]:
        return _LIVE_PROXY_CACHE["triangles"], _LIVE_PROXY_CACHE["lines"]

    surface_records = []
    for mesh_obj in _stored_target_meshes(primary_mesh):
        surface_records.extend(
            _records_from_indices(mesh_obj, _load_selected_indices(mesh_obj))
        )
    if len(surface_records) < 4:
        return [], []
    surface_points = [record["world"] for record in surface_records]
    surface_normals = [record["normal"] for record in surface_records]
    control_points = _control_world_locations(controls)

    if settings.skirt_shape_mode == 'ASYMMETRIC':
        if not _controls_are_asymmetric(control_points):
            raise ValueError("旧版不对称控制点不兼容，请重新生成控制点")
        preview_data = build_variable_asymmetric_data(
            control_points,
            surface_points,
            settings,
            _get_asym_anchor_bone_counts(settings, len(control_points)),
        )
        samples = preview_data["preview_grid"]
    else:
        samples = build_cage_samples(
            control_points,
            settings.rows,
            settings.columns,
            settings.bone_inset,
            True,
            surface_points=surface_points,
            surface_normals=surface_normals,
            slice_thickness=settings.slice_thickness,
            fit_mode=settings.fit_mode,
            ring_fit_strength=settings.ring_fit_strength,
            longitudinal_smoothing=settings.longitudinal_smoothing,
            smooth_bone_chains=settings.smooth_bone_chains,
        )
        samples = _uniform_proxy_row_samples(samples)

    triangles, lines = _proxy_draw_geometry(samples, True)
    _LIVE_PROXY_CACHE["signature"] = signature
    _LIVE_PROXY_CACHE["triangles"] = triangles
    _LIVE_PROXY_CACHE["lines"] = lines
    return triangles, lines


def _draw_control_edit_selection_markers():
    """Draw a live proxy, colored control chains and fitted target vertices."""
    try:
        scene = bpy.context.scene
        settings = scene.auto_skirt_settings
        reviewing = settings.workflow_stage == 3 and not (
            settings.control_edit_active or scene.auto_sheet_settings.edit_active
        )
        if reviewing:
            control_chains, selection_positions, proxy = unified_generator.generation_preview(scene)
            proxy_triangles, proxy_lines = proxy
        else:
            control_chains = _control_edit_control_world_positions(scene)
            selection_positions = _control_edit_selected_world_positions(scene)
            proxy_triangles, proxy_lines = _control_edit_live_proxy_geometry(scene)
        if not any(control_chains.values()) and not selection_positions and not proxy_lines:
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
        if selection_positions:
            gpu.state.point_size_set(4.0)
            shader.uniform_float("color", (1.0, 0.28, 0.04, 1.0))
            batch_for_shader(
                shader, 'POINTS', {"pos": selection_positions}
            ).draw(shader)
        gpu.state.point_size_set(15.0)
        reverses = unified_generator._main_reverses(settings) if reviewing else []
        for chain_key, positions in control_chains.items():
            if not positions:
                continue
            color = (
                unified_generator.display_color(int(chain_key[1:]))
                if chain_key.startswith("C")
                else CONTROL_CHAIN_COLORS[chain_key]
            )
            shader.uniform_float("color", color)
            batch_for_shader(shader, 'POINTS', {"pos": positions}).draw(shader)
            if len(positions) > 1:
                chain_lines = []
                for index in range(len(positions) - 1):
                    chain_lines.extend((positions[index], positions[index + 1]))
                batch_for_shader(shader, 'LINES', {"pos": chain_lines}).draw(shader)
                if reviewing:
                    arrows = unified_generator.generation_arrow_lines(
                        positions, settings.asym_default_bone_count, reverses[int(chain_key[1:])]
                    )
                    if arrows:
                        gpu.state.line_width_set(2.8)
                        batch_for_shader(shader, 'LINES', {"pos": arrows}).draw(shader)
    except (ReferenceError, RuntimeError, ValueError, KeyError):
        # A viewport may be closing or the edit session may be incomplete.
        pass
    finally:
        try:
            gpu.state.point_size_set(1.0)
            gpu.state.line_width_set(1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.blend_set('NONE')
        except (AttributeError, RuntimeError):
            pass

def _remove_selection_marker_draw_handler():
    global _SELECTION_MARKER_DRAW_HANDLE
    if _SELECTION_MARKER_DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(
                _SELECTION_MARKER_DRAW_HANDLE,
                'WINDOW',
            )
        except (ReferenceError, RuntimeError):
            pass
        _SELECTION_MARKER_DRAW_HANDLE = None
_ASYM_CONTROL_RESAMPLE_GUARD = False


def _install_selection_marker_draw_handler():
    global _SELECTION_MARKER_DRAW_HANDLE
    _remove_selection_marker_draw_handler()
    _SELECTION_MARKER_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_control_edit_selection_markers,
        (),
        'WINDOW',
        'POST_VIEW',
    )


def _force_fixed_weight_defaults(settings):
    # The cage is always a closed Front -> Right -> Back -> Left loop.
    # Keep the legacy RNA value for older saved scenes, but never expose it
    # as a user-facing choice.
    settings.closed_loop = True
    settings.top_blend_to_parent = True
    settings.normalize_weights = True


# ---------------------------------------------------------------------------
# Control point fitting and cage interpolation
# ---------------------------------------------------------------------------


def _world_axes(armature_obj):
    up = Vector((0.0, 0.0, 1.0))
    right = Vector((1.0, 0.0, 0.0))
    back = Vector((0.0, 1.0, 0.0))
    if armature_obj is not None:
        basis = armature_obj.matrix_world.to_3x3()
        up = _safe_normalized(basis @ Vector((0.0, 0.0, 1.0)), up)
        right = basis @ Vector((1.0, 0.0, 0.0))
        back = basis @ Vector((0.0, 1.0, 0.0))

    # Gram-Schmidt keeps all radial directions in the height-normal plane.
    right -= up * right.dot(up)
    right = _safe_normalized(right, Vector((1.0, 0.0, 0.0)))
    back -= up * back.dot(up)
    back -= right * back.dot(right)
    back = _safe_normalized(back, up.cross(right))
    if right.cross(back).dot(up) < 0.0:
        back.negate()
    return up, right, back


def _fit_control_positions(records, armature_obj):
    points = [item["world"].copy() for item in records]
    if len(points) < 4:
        raise ValueError("至少需要选择 4 个顶点来拟合控制笼")

    up, right, back = _world_axes(armature_obj)
    heights = [point.dot(up) for point in points]
    h_top = max(heights)
    h_bottom = min(heights)
    total_height = h_top - h_bottom
    if total_height <= EPSILON:
        raise ValueError("所选顶点沿高度轴的范围过小，无法生成控制笼")

    all_center = _average_points(points)
    slice_band = max(total_height * 0.12, total_height / 30.0)
    fallback_count = min(len(points), max(8, int(math.ceil(len(points) * 0.08))))

    controls = {direction: [] for direction in CONTROL_DIRECTIONS}
    direction_vectors = {
        "Front": -back,
        "Right": right,
        "Back": back,
        "Left": -right,
    }

    for level in range(CONTROL_LEVELS):
        fraction = level / (CONTROL_LEVELS - 1)
        target_h = h_top - total_height * fraction
        nearby = [
            point for point, height in zip(points, heights)
            if abs(height - target_h) <= slice_band
        ]
        if len(nearby) < 3:
            ordered = sorted(
                zip(points, heights),
                key=lambda pair: abs(pair[1] - target_h),
            )
            nearby = [pair[0] for pair in ordered[:fallback_count]]

        center = _average_points(nearby) if nearby else all_center.copy()
        center += up * (target_h - center.dot(up))

        for direction in CONTROL_DIRECTIONS:
            axis = direction_vectors[direction]
            radius = max((point - center).dot(axis) for point in nearby)
            if radius <= EPSILON:
                radius = max((point - center).dot(axis) for point in points)
            if radius <= EPSILON:
                radial_distances = [
                    ((point - center) - up * (point - center).dot(up)).length
                    for point in points
                ]
                radius = max(max(radial_distances), total_height * 0.02)
            controls[direction].append(center + axis * radius * 1.03)

    # Remove sideways center drift while preserving each direction's taper.
    # Front/Back share one Right-axis coordinate, so they form vertical lines
    # in front view. Left/Right likewise share one Back-axis coordinate, so
    # they form vertical lines in side view. This is evaluated in the
    # armature-aligned frame rather than assuming literal World X/Y.
    fixed_right = all_center.dot(right)
    fixed_back = all_center.dot(back)
    for level in range(CONTROL_LEVELS):
        for direction in ("Front", "Back"):
            point = controls[direction][level]
            controls[direction][level] = point + right * (fixed_right - point.dot(right))
        for direction in ("Right", "Left"):
            point = controls[direction][level]
            controls[direction][level] = point + back * (fixed_back - point.dot(back))

    diagonal = max((point - all_center).length for point in points) * 2.0
    display_size = max(diagonal * 0.025, total_height * 0.015, 0.001)
    return controls, display_size


def _circular_distance(first, second):
    delta = abs((float(first) - float(second)) % 1.0)
    return min(delta, 1.0 - delta)


def _raw_skirt_u(point, center, right, back):
    delta = Vector(point) - Vector(center)
    x = delta.dot(right)
    y = delta.dot(back)
    if x * x + y * y <= EPSILON:
        return 0.0
    return ((math.atan2(y, x) + math.pi * 0.5) % math.tau) / math.tau


def _radial_at_u(raw_u, right, back):
    theta = -math.pi * 0.5 + math.tau * (float(raw_u) % 1.0)
    return _safe_normalized(
        math.cos(theta) * right + math.sin(theta) * back,
        right,
    )


def _periodic_smooth_values(values, passes=1):
    result = [float(value) for value in values]
    if len(result) < 3:
        return result
    for _pass in range(max(int(passes), 0)):
        source = result
        result = [
            source[(index - 1) % len(source)] * 0.25
            + source[index] * 0.5
            + source[(index + 1) % len(source)] * 0.25
            for index in range(len(source))
        ]
    return result


def _fit_smooth_hem_profile(records, armature_obj, sample_count=128):
    points = [record["world"].copy() for record in records]
    if len(points) < 4:
        raise ValueError("至少需要 4 个顶点拟合不对称裙摆")

    up, right, back = _world_axes(armature_obj)
    heights = [point.dot(up) for point in points]
    top = max(heights)
    bottom = min(heights)
    height_span = top - bottom
    if height_span <= EPSILON:
        raise ValueError("选区沿腰口高度轴的范围过小")

    waist_band = max(height_span * 0.10, height_span / 40.0)
    waist_points = [
        point for point, height in zip(points, heights)
        if height >= top - waist_band
    ]
    center = _average_points(waist_points or points)
    center += up * (top - center.dot(up))
    point_u = [
        _raw_skirt_u(point, center, right, back)
        for point in points
    ]

    sample_count = max(int(sample_count), 32)
    fitted_heights = []
    fitted_radii = []
    base_window = max(2.5 / sample_count, 0.035)

    for sample in range(sample_count):
        raw_u = sample / sample_count
        window = base_window
        candidates = []
        for _attempt in range(6):
            candidates = [
                index for index, value in enumerate(point_u)
                if _circular_distance(value, raw_u) <= window
            ]
            if len(candidates) >= 8:
                break
            window = min(window * 1.75, 0.5)
        if not candidates:
            candidates = list(range(len(points)))

        candidates.sort(key=lambda index: heights[index])
        lower_count = max(3, int(math.ceil(len(candidates) * 0.05)))
        lower = candidates[:min(lower_count, len(candidates))]
        radial = _radial_at_u(raw_u, right, back)

        weighted_heights = []
        radial_values = []
        for index in lower:
            angular = _circular_distance(point_u[index], raw_u)
            weight = max(1.0 - angular / max(window, EPSILON), 0.05)
            axis_point = center + up * (heights[index] - center.dot(up))
            radius = max((points[index] - axis_point).dot(radial), 0.0)
            weighted_heights.append((heights[index], weight))
            radial_values.append((radius, weight))

        height_weight = sum(weight for _value, weight in weighted_heights)
        fitted_height = (
            sum(value * weight for value, weight in weighted_heights) / height_weight
            if height_weight > EPSILON else heights[lower[0]]
        )

        # Longest selected layer defines the shared envelope. The outer half
        # suppresses thickness noise while preserving the exterior silhouette.
        radial_values.sort(key=lambda item: item[0], reverse=True)
        outer = radial_values[:max(1, int(math.ceil(len(radial_values) * 0.5)))]
        radius_weight = sum(weight for _value, weight in outer)
        fitted_radius = (
            sum(value * weight for value, weight in outer) / radius_weight
            if radius_weight > EPSILON else max(value for value, _weight in radial_values)
        )
        fitted_heights.append(fitted_height)
        fitted_radii.append(max(fitted_radius, height_span * 0.005))

    fitted_heights = _periodic_smooth_values(fitted_heights, passes=2)
    fitted_radii = _periodic_smooth_values(fitted_radii, passes=2)
    profile_points = []
    for sample, (height, radius) in enumerate(zip(fitted_heights, fitted_radii)):
        raw_u = sample / sample_count
        radial = _radial_at_u(raw_u, right, back)
        axis_point = center + up * (height - center.dot(up))
        profile_points.append(axis_point + radial * radius * 1.03)

    return {
        "points": profile_points,
        "center": center,
        "up": up,
        "right": right,
        "back": back,
        "point_u": point_u,
        "source_points": points,
        "source_heights": heights,
    }


def _profile_point_at(profile, raw_u):
    points = profile["points"]
    if not points:
        return Vector(profile["center"])
    value = (float(raw_u) % 1.0) * len(points)
    first = int(math.floor(value)) % len(points)
    second = (first + 1) % len(points)
    return Vector(points[first]).lerp(
        Vector(points[second]),
        value - math.floor(value),
    )


def _profile_horizontal_sector_data(
    profile,
    chain_count,
    samples_per_sector=64,
):
    """Measure the selected hem envelope in top view for chain allocation."""
    up = _safe_normalized(profile["up"], Vector((0.0, 0.0, 1.0)))
    sectors = []
    for sector in range(chain_count):
        cumulative = [0.0]
        previous = _profile_point_at(profile, sector / chain_count)
        for step in range(1, samples_per_sector + 1):
            raw_u = (sector + step / samples_per_sector) / chain_count
            current = _profile_point_at(profile, raw_u)
            delta = current - previous
            horizontal = delta - up * delta.dot(up)
            cumulative.append(cumulative[-1] + max(horizontal.length, EPSILON))
            previous = current
        sectors.append({
            "cumulative": cumulative,
            "length": cumulative[-1],
            "samples": samples_per_sector,
        })
    return sectors


def _build_profile_arc_mapping(profile, resolution=256):
    """Map polar angle to horizontal outer-envelope arc length."""
    resolution = max(int(resolution), 64)
    up = _safe_normalized(profile["up"], Vector((0.0, 0.0, 1.0)))
    cumulative = [0.0]
    previous = _profile_point_at(profile, 0.0)
    for sample in range(1, resolution + 1):
        current = _profile_point_at(profile, sample / resolution)
        delta = current - previous
        horizontal = delta - up * delta.dot(up)
        cumulative.append(cumulative[-1] + max(horizontal.length, EPSILON))
        previous = current
    return {
        "resolution": resolution,
        "cumulative": cumulative,
        "total": cumulative[-1],
    }


def _build_point_cloud_average_arc_mapping(
    surface_points,
    control_points,
    profile,
    resolution=256,
    level_count=8,
):
    """Measure average horizontal perimeter from waist to hem.

    Hem-only arc sampling clusters chains around a large lobe and leaves the
    waist crowded. Averaging several fitted slices keeps spacing balanced over
    the complete proxy surface.
    """
    resolution = max(int(resolution), 64)
    level_count = max(int(level_count), 4)
    center = Vector(profile["center"])
    up = _safe_normalized(profile["up"], Vector((0.0, 0.0, 1.0)))
    right = profile["right"]
    back = profile["back"]
    raw_parameters = []
    rail_cache = {}
    rail_resolution = max(resolution, 128)
    for point in surface_points:
        point = Vector(point)
        raw_u = _raw_skirt_u(point, center, right, back)
        cache_index = int(round(raw_u * rail_resolution)) % rail_resolution
        if cache_index not in rail_cache:
            cached_u = cache_index / rail_resolution
            rail_cache[cache_index] = [
                _variable_asym_surface_point(
                    control_points,
                    cached_u,
                    step / 48.0,
                )
                for step in range(49)
            ]
        raw_parameters.append((
            raw_u,
            _closest_polyline_parameter(point, rail_cache[cache_index]),
        ))

    rings = []
    angular_band = max(2.5 / resolution, 0.035)
    for level in range(level_count):
        local_v = level / (level_count - 1)
        slice_center = _average_points([
            _variable_asym_surface_point(
                control_points,
                sample / max(resolution // 4, 32),
                local_v,
            )
            for sample in range(max(resolution // 4, 32))
        ])
        ring = []
        for sample in range(resolution):
            raw_u = sample / resolution
            radial = _radial_at_u(raw_u, right, back)
            v_band = 0.08
            candidates = []
            for _attempt in range(5):
                candidates = [
                    index
                    for index, (point_u, point_v) in enumerate(raw_parameters)
                    if _circular_distance(point_u, raw_u) <= angular_band
                    and abs(point_v - local_v) <= v_band
                ]
                if len(candidates) >= 4:
                    break
                v_band = min(v_band * 1.7, 0.5)
            if not candidates:
                candidates = sorted(
                    range(len(surface_points)),
                    key=lambda index: (
                        _circular_distance(raw_parameters[index][0], raw_u)
                        + abs(raw_parameters[index][1] - local_v)
                    ),
                )[:min(8, len(surface_points))]

            radii = []
            for index in candidates:
                delta = Vector(surface_points[index]) - slice_center
                horizontal = delta - up * delta.dot(up)
                if horizontal.length_squared <= EPSILON:
                    continue
                direction_score = horizontal.normalized().dot(radial)
                if direction_score <= 0.10:
                    continue
                radius = horizontal.dot(radial)
                if radius > 0.0:
                    radii.append(radius)
            if radii:
                radii.sort(reverse=True)
                outer = radii[:min(5, len(radii))]
                radius = max(sum(outer) / len(outer), outer[0] * 0.985)
            else:
                target = _variable_asym_surface_point(
                    control_points,
                    raw_u,
                    local_v,
                )
                delta = target - slice_center
                horizontal = delta - up * delta.dot(up)
                radius = horizontal.length
            axis_point = slice_center + up * (
                _variable_asym_surface_point(
                    control_points,
                    raw_u,
                    local_v,
                ) - slice_center
            ).dot(up)
            ring.append(axis_point + radial * max(radius, EPSILON))
        rings.append(ring)

    cumulative = [0.0]
    for sample in range(resolution):
        following = (sample + 1) % resolution
        length = 0.0
        for ring in rings:
            delta = ring[following] - ring[sample]
            horizontal = delta - up * delta.dot(up)
            length += horizontal.length
        cumulative.append(
            cumulative[-1] + max(length / len(rings), EPSILON)
        )
    return {
        "resolution": resolution,
        "cumulative": cumulative,
        "total": cumulative[-1],
    }


def _build_fitted_chain_arc_mapping(
    raw_us,
    dense_chains,
    up,
    resolution=256,
):
    """Build a correction map from measured final chain-to-chain spacing."""
    count = len(raw_us)
    if count < 3 or len(dense_chains) != count:
        return None
    up = _safe_normalized(up, Vector((0.0, 0.0, 1.0)))
    segment_lengths = []
    for column in range(count):
        following = (column + 1) % count
        values = []
        for first, second in zip(
            dense_chains[column],
            dense_chains[following],
        ):
            delta = Vector(second) - Vector(first)
            horizontal = delta - up * delta.dot(up)
            values.append(horizontal.length)
        segment_lengths.append(max(
            sum(values) / max(len(values), 1),
            EPSILON,
        ))

    starts = [float(value) % 1.0 for value in raw_us]
    accumulated = [0.0]
    for length in segment_lengths:
        accumulated.append(accumulated[-1] + length)
    total = accumulated[-1]
    resolution = max(int(resolution), 64)
    cumulative = []
    segment = 0
    for sample in range(resolution + 1):
        raw_u = sample / resolution
        while segment + 1 < count and raw_u >= starts[segment + 1]:
            segment += 1
        first_u = starts[segment]
        following_u = starts[segment + 1] if segment + 1 < count else 1.0
        span = max(following_u - first_u, EPSILON)
        factor = clamp((raw_u - first_u) / span, 0.0, 1.0)
        cumulative.append(
            accumulated[segment] + segment_lengths[segment] * factor
        )
    cumulative[-1] = total
    return {
        "resolution": resolution,
        "cumulative": cumulative,
        "total": total,
    }


def _fit_asymmetric_control_positions(records, armature_obj):
    standard, display_size = _fit_control_positions(records, armature_obj)
    profile = _fit_smooth_hem_profile(records, armature_obj)
    center = profile["center"]
    up = profile["up"]
    right = profile["right"]
    back = profile["back"]
    source_points = profile["source_points"]
    source_heights = profile["source_heights"]
    source_u = profile["point_u"]
    direction_u = {
        "Front": 0.0,
        "Right": 0.25,
        "Back": 0.5,
        "Left": 0.75,
    }

    controls = {direction: [] for direction in CONTROL_DIRECTIONS}
    for direction in CONTROL_DIRECTIONS:
        raw_u = direction_u[direction]
        radial = _radial_at_u(raw_u, right, back)
        top_point = Vector(standard[direction][0])
        bottom_point = _profile_point_at(profile, raw_u)
        top_height = top_point.dot(up)
        bottom_height = bottom_point.dot(up)
        local_span = max(top_height - bottom_height, EPSILON)

        candidates = [
            index for index, value in enumerate(source_u)
            if _circular_distance(value, raw_u) <= 0.14
        ]
        if len(candidates) < 8:
            candidates = sorted(
                range(len(source_points)),
                key=lambda index: _circular_distance(source_u[index], raw_u),
            )[:min(max(8, len(source_points) // 8), len(source_points))]

        direction_points = [top_point]
        for level in range(1, CONTROL_LEVELS - 1):
            local_v = level / (CONTROL_LEVELS - 1)
            target_height = top_height * (1.0 - local_v) + bottom_height * local_v
            ranked = sorted(
                candidates,
                key=lambda index: (
                    abs(source_heights[index] - target_height) / local_span
                    + _circular_distance(source_u[index], raw_u) * 1.5
                ),
            )
            chosen = ranked[:min(max(5, len(ranked) // 8), 24)]
            values = []
            for index in chosen:
                height_error = abs(source_heights[index] - target_height) / local_span
                angular = _circular_distance(source_u[index], raw_u) / 0.14
                weight = 1.0 / (0.03 + height_error * height_error + angular * angular)
                axis_point = center + up * (source_heights[index] - center.dot(up))
                radius = max((source_points[index] - axis_point).dot(radial), 0.0)
                values.append((source_heights[index], radius, weight))
            total = sum(weight for _height, _radius, weight in values)
            if total > EPSILON:
                height = sum(height * weight for height, _radius, weight in values) / total
                outer = sorted(values, key=lambda item: item[1], reverse=True)
                outer = outer[:max(1, int(math.ceil(len(outer) * 0.5)))]
                outer_total = sum(weight for _height, _radius, weight in outer)
                radius = sum(radius * weight for _height, radius, weight in outer) / outer_total
                axis_point = center + up * (height - center.dot(up))
                direction_points.append(axis_point + radial * radius * 1.03)
            else:
                direction_points.append(top_point.lerp(bottom_point, local_v))
        direction_points.append(bottom_point)
        controls[direction] = direction_points

    return controls, display_size, profile

def _fit_variable_asymmetric_control_positions(
    records,
    armature_obj,
    chain_count,
    bone_inset=0.05,
):
    """Create an easy-to-edit circumscribed circular asymmetric cage.

    All chains share one horizontal outer radius. Their L3 heights still
    follow the fitted hem, so an asymmetric skirt keeps useful initial chain
    lengths while the top view remains a regular circle.
    """
    chain_count = int(clamp(chain_count, 4, 24))
    profile = _fit_smooth_hem_profile(records, armature_obj)
    points = [item["world"].copy() for item in records]
    center = Vector(profile["center"])
    up = _safe_normalized(profile["up"], Vector((0.0, 0.0, 1.0)))
    right = profile["right"]
    back = profile["back"]
    heights = [point.dot(up) for point in points]
    top_height = max(heights)
    bottom_height = min(heights)
    height_span = max(top_height - bottom_height, EPSILON)

    horizontal_radii = []
    for point in points:
        delta = point - center
        horizontal = delta - up * delta.dot(up)
        horizontal_radii.append(horizontal.length)
    source_radius = max(max(horizontal_radii, default=0.0), height_span * 0.02)

    # Compensate the generated cage for Bone Inset. Its usable inner limit
    # remains slightly outside the selected point cloud instead of clipping it.
    inset_scale = max(1.0 - clamp(float(bone_inset), 0.0, 0.95), 0.05)
    cage_radius = source_radius * 1.08 / inset_scale

    controls = []
    for chain in range(chain_count):
        raw_u = chain / chain_count
        radial = _radial_at_u(raw_u, right, back)
        bottom_height_at_chain = _profile_point_at(profile, raw_u).dot(up)
        chain_points = []
        for level in range(CONTROL_LEVELS):
            local_v = level / (CONTROL_LEVELS - 1)
            height = _lerp_float(
                top_height,
                bottom_height_at_chain,
                local_v,
            )
            axis_point = center + up * (height - center.dot(up))
            chain_points.append(axis_point + radial * cage_radius)
        controls.append(chain_points)

    display_size = max(
        cage_radius * 0.04,
        height_span * 0.015,
        0.001,
    )
    return controls, display_size, profile

def _create_or_update_asymmetric_controls(
    mesh_obj,
    settings,
    controls,
    display_size,
    update_existing=False,
):
    collection_name = _control_collection_name(settings.prefix, mesh_obj.name)
    collection = get_or_create_collection(collection_name)
    collection["control_prefix"] = settings.prefix
    collection["control_levels"] = len(controls[0])
    existing = find_control_points(collection, settings.prefix, require_all=False)
    if update_existing and (
        not _controls_are_asymmetric(existing)
        or len(existing) != len(controls)
        or any(obj is None for chain in existing for obj in chain)
    ):
        raise ValueError(
            "当前不对称控制链数量或拓扑不一致，请重新生成控制点"
        )

    for chain, chain_points in enumerate(controls):
        for level in range(len(chain_points)):
            name = _asym_control_name(settings.prefix, chain, level)
            obj = bpy.data.objects.get(name)
            if obj is None or obj.get(PROP_TARGET) != mesh_obj.name:
                obj = bpy.data.objects.new(name, None)
                collection.objects.link(obj)
            elif collection.objects.get(obj.name) is None:
                collection.objects.link(obj)
            obj.empty_display_type = 'SPHERE'
            obj.empty_display_size = display_size
            obj.location = Vector(chain_points[level])
            obj.show_in_front = True
            obj.color = _asym_control_color(chain)
            obj[PROP_TARGET] = mesh_obj.name
            obj[PROP_KIND] = EDIT_KIND_CONTROL
            obj["auto_skirt_chain_index"] = chain
            obj["auto_skirt_level"] = level
            _set_control_object_locked(obj, True)

    _store_mesh_metadata(mesh_obj, settings, collection_name)
    return collection


def _asym_chain_arc_length(chain_points, samples=65):
    points = [
        interpolate_vertical_control(
            chain_points,
            sample / (samples - 1),
        )
        for sample in range(samples)
    ]
    return sum(
        (points[index + 1] - points[index]).length
        for index in range(len(points) - 1)
    )


def _auto_asym_anchor_bone_counts(settings, control_points):
    lengths = [_asym_chain_arc_length(chain) for chain in control_points]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    default_count = int(clamp(settings.asym_default_bone_count, 1, 64))
    if average_length <= EPSILON:
        counts = [default_count] * len(control_points)
    else:
        counts = [
            int(clamp(round(default_count * length / average_length), 1, 64))
            for length in lengths
        ]
    _set_asym_anchor_bone_counts(settings, counts, len(control_points))
    return counts

def _create_or_update_controls(
    mesh_obj,
    settings,
    controls,
    display_size,
    update_existing=False,
):
    collection_name = _control_collection_name(settings.prefix, mesh_obj.name)
    collection = get_or_create_collection(collection_name)
    existing = find_control_points(collection, settings.prefix, require_all=False)

    if update_existing:
        missing = [
            _control_name(settings.prefix, direction, level)
            for direction in CONTROL_DIRECTIONS
            for level in range(CONTROL_LEVELS)
            if level >= len(existing.get(direction, []))
            or existing[direction][level] is None
        ]
        if missing:
            raise ValueError(
                "Refit 需要完整的 16 个控制点；缺少：" + ", ".join(missing[:4])
            )

    for direction in CONTROL_DIRECTIONS:
        for level in range(CONTROL_LEVELS):
            name = _control_name(settings.prefix, direction, level)
            obj = bpy.data.objects.get(name)
            if obj is None or obj.get(PROP_TARGET) != mesh_obj.name:
                obj = bpy.data.objects.new(name, None)
                collection.objects.link(obj)
            elif collection.objects.get(obj.name) is None:
                collection.objects.link(obj)

            obj.empty_display_type = 'SPHERE'
            obj.empty_display_size = display_size
            obj.location = controls[direction][level]
            obj.show_in_front = True
            obj.color = CONTROL_CHAIN_COLORS[direction]
            obj[PROP_TARGET] = mesh_obj.name
            obj[PROP_KIND] = EDIT_KIND_CONTROL
            obj["auto_skirt_direction"] = direction
            obj["auto_skirt_level"] = level
            _set_control_object_locked(obj, True)

    _store_mesh_metadata(mesh_obj, settings, collection_name)
    return collection


def generate_control_points(context, settings):
    mesh_objects = _resolve_target_meshes(
        context, settings, prefer_edit_selection=True
    )
    if not mesh_objects:
        raise ValueError("Target Mesh does not exist.")
    if any(obj.mode != 'EDIT' for obj in mesh_objects):
        raise ValueError("Keep every target mesh in Edit Mode and select vertices.")

    records_by_mesh = []
    combined_records = []
    for mesh_obj in mesh_objects:
        records = get_selected_vertices(mesh_obj)
        if not records:
            raise ValueError(
                f"Mesh '{mesh_obj.name}' has no selected vertices."
            )
        records_by_mesh.append((mesh_obj, records))
        combined_records.extend(records)

    primary_mesh = mesh_objects[0]
    _validate_prefix(settings.prefix)
    asymmetric = settings.skirt_shape_mode == 'ASYMMETRIC'
    if asymmetric:
        controls, display_size, profile = _fit_variable_asymmetric_control_positions(
            combined_records, settings.target_armature, settings.asym_control_chain_count,
            settings.bone_inset,
        )
        if settings.workflow_enabled:
            controls = unified_generator.fit_closed_rails(
                combined_records, controls, profile, settings.control_levels
            )
        _auto_asym_anchor_bone_counts(settings, controls)
        settings.columns = max(settings.columns, len(controls))
        create_controls = _create_or_update_asymmetric_controls
    else:
        controls, display_size = _fit_control_positions(combined_records, settings.target_armature)
        create_controls = _create_or_update_controls

    settings.target_mesh = primary_mesh
    old_prefix = str(primary_mesh.get(PROP_PREFIX, settings.prefix))
    previous = bpy.data.collections.get(str(primary_mesh.get(PROP_COLLECTION, "")))
    if previous is not None:
        old_prefix = previous.get("control_prefix", old_prefix)
    for prefix in {old_prefix, settings.prefix}:
        delete_generated_controls(prefix, primary_mesh.name)
    collection = create_controls(primary_mesh, settings, controls, display_size, update_existing=False)
    if PROP_ASYM_BASE_L3 in primary_mesh:
        del primary_mesh[PROP_ASYM_BASE_L3]
    for mesh_obj, records in records_by_mesh:
        _store_selection_records(mesh_obj, records)
        _store_mesh_metadata(mesh_obj, settings, collection.name, mesh_objects)
    return collection, len(combined_records)


def _refit_control_points(context, settings):
    mesh_objects = _resolve_target_meshes(
        context, settings, prefer_edit_selection=True
    )
    if not mesh_objects:
        raise ValueError("Target Mesh does not exist.")
    if any(obj.mode != 'EDIT' for obj in mesh_objects):
        raise ValueError("Keep every target mesh in Edit Mode before refitting.")

    records_by_mesh = []
    combined_records = []
    for mesh_obj in mesh_objects:
        records = get_selected_vertices(mesh_obj)
        if not records:
            raise ValueError(
                f"Mesh '{mesh_obj.name}' has no selected vertices."
            )
        records_by_mesh.append((mesh_obj, records))
        combined_records.extend(records)

    primary_mesh = mesh_objects[0]
    settings.target_mesh = primary_mesh
    collection_name = str(
        primary_mesh.get(
            PROP_COLLECTION,
            _control_collection_name(settings.prefix, primary_mesh.name),
        )
    )
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        raise ValueError("Control Collection does not exist; generate controls first.")
    find_control_points(collection, settings.prefix, require_all=True)

    if settings.skirt_shape_mode == 'ASYMMETRIC':
        controls, display_size, _profile = (
            _fit_variable_asymmetric_control_positions(
                combined_records,
                settings.target_armature,
                settings.asym_control_chain_count,
                settings.bone_inset,
            )
        )
        _auto_asym_anchor_bone_counts(settings, controls)
        _create_or_update_asymmetric_controls(
            primary_mesh,
            settings,
            controls,
            display_size,
            update_existing=True,
        )
    else:
        controls, display_size = _fit_control_positions(
            combined_records, settings.target_armature
        )
        if PROP_ASYM_BASE_L3 in primary_mesh:
            del primary_mesh[PROP_ASYM_BASE_L3]
        _create_or_update_controls(
            primary_mesh,
            settings,
            controls,
            display_size,
            update_existing=True,
        )
    for mesh_obj, records in records_by_mesh:
        _store_selection_records(mesh_obj, records)
        _store_mesh_metadata(mesh_obj, settings, collection.name, mesh_objects)
    return len(combined_records)


def find_control_points(collection, prefix, require_all=True):
    if collection is not None:
        prefix = collection.get("control_prefix", prefix)
    dynamic = []
    if collection is not None:
        dynamic = [
            obj for obj in collection.objects
            if obj.type == 'EMPTY'
            and obj.get(PROP_KIND) == EDIT_KIND_CONTROL
            and obj.name.startswith(f"{prefix}CTRL_C")
            and int(obj.get("auto_skirt_chain_index", -1)) >= 0
        ]
    if dynamic:
        chain_count = max(
            int(obj.get("auto_skirt_chain_index", -1)) for obj in dynamic
        ) + 1
        level_count = int(collection.get("control_levels", 4))
        result = [[None] * level_count for _chain in range(chain_count)]
        for obj in dynamic:
            chain = int(obj.get("auto_skirt_chain_index", -1))
            level = int(obj.get("auto_skirt_level", -1))
            if 0 <= chain < chain_count and 0 <= level < level_count:
                result[chain][level] = obj
        missing = [
            _asym_control_name(prefix, chain, level)
            for chain in range(chain_count)
            for level in range(level_count)
            if result[chain][level] is None
        ]
        if require_all and missing:
            raise ValueError(
                f"不对称控制点不完整（需要 {chain_count * level_count} 个）；缺少："
                + ", ".join(missing[:6])
            )
        return result

    result = {direction: [None] * CONTROL_LEVELS for direction in CONTROL_DIRECTIONS}
    missing = []
    for direction in CONTROL_DIRECTIONS:
        for level in range(CONTROL_LEVELS):
            name = _control_name(prefix, direction, level)
            obj = collection.objects.get(name) if collection is not None else None
            if obj is None or obj.type != 'EMPTY':
                missing.append(name)
            else:
                result[direction][level] = obj
    if require_all and missing:
        raise ValueError(
            "控制点不完整（需要 16 个）；缺少：" + ", ".join(missing[:6])
        )
    return result


def _control_world_locations(control_objects):
    if _controls_are_asymmetric(control_objects):
        return [
            [obj.matrix_world.translation.copy() for obj in chain]
            for chain in control_objects
        ]
    return {
        direction: [
            obj.matrix_world.translation.copy()
            for obj in control_objects[direction]
        ]
        for direction in CONTROL_DIRECTIONS
    }

def interpolate_vertical_control(direction_points, v):
    """
    Stable four-level interpolation.

    Piecewise linear interpolation with smoothstep easing is deliberately used
    instead of an overshooting cubic spline, so results remain inside each pair
    of neighboring controls.
    """
    if len(direction_points) < 2:
        raise ValueError("每条控制链至少需要两个控制点")
    v = clamp(float(v), 0.0, 1.0)
    scaled = v * (len(direction_points) - 1)
    segment = min(int(math.floor(scaled)), len(direction_points) - 2)
    factor = smoothstep(scaled - segment)
    return Vector(direction_points[segment]).lerp(
        Vector(direction_points[segment + 1]), factor
    )


def _cage_section(control_points, v):
    section = {
        direction: interpolate_vertical_control(control_points[direction], v)
        for direction in CONTROL_DIRECTIONS
    }
    center = _average_points([section[name] for name in CONTROL_DIRECTIONS])
    return section, center


def _section_axes(section):
    """Return a stable polar frame for one cage section.

    The four control points define only the center, the local X/Y axes, and
    four cardinal radius limits.  Bone-ring samples should not be placed on
    the straight Front->Right->Back->Left boundary, otherwise the ring becomes
    a diamond.
    """
    center = _average_points([section[name] for name in CONTROL_DIRECTIONS])
    x_axis = _safe_normalized(
        section["Right"] - section["Left"],
        Vector((1.0, 0.0, 0.0)),
    )
    y_axis = section["Back"] - section["Front"]
    y_axis -= x_axis * y_axis.dot(x_axis)
    y_axis = _safe_normalized(y_axis, Vector((0.0, 1.0, 0.0)))
    return center, x_axis, y_axis


def _lerp_float(a, b, factor):
    return float(a) * (1.0 - factor) + float(b) * factor


def _smooth_periodic_radius(theta, r_front, r_right, r_back, r_left):
    """Smooth radius interpolation at polar angle theta.

    Cardinal angle convention in the section frame:
    Front = -pi/2, Right = 0, Back = pi/2, Left = pi.
    This returns a smooth r(theta) limit instead of a four-segment polygon
    boundary length.
    """
    radii = [
        max(float(r_front), EPSILON),
        max(float(r_right), EPSILON),
        max(float(r_back), EPSILON),
        max(float(r_left), EPSILON),
    ]
    # phase 0 is Front, then Right, Back, Left, and back to Front.
    phase = (float(theta) + math.pi * 0.5) % math.tau
    sector_float = phase / (math.pi * 0.5)
    sector = int(math.floor(sector_float)) % 4
    t = sector_float - math.floor(sector_float)
    # Cosine interpolation avoids the corner-like feel of linear cardinal radii.
    t = (1.0 - math.cos(t * math.pi)) * 0.5
    return _lerp_float(radii[sector], radii[(sector + 1) % 4], t)


def _cage_boundary_point(section, u, closed_loop):
    """Compatibility helper for UI/debug code.

    The fitted bone ring no longer uses this four-control polyline.  It is
    retained only for any external callers that still expect a boundary point.
    """
    center, x_axis, y_axis = _section_axes(section)
    if closed_loop:
        theta = -math.pi * 0.5 + math.tau * (u % 1.0)
    else:
        theta = -math.pi * 0.5 + math.pi * 1.5 * clamp(u, 0.0, 1.0)
    radial = _safe_normalized(
        math.cos(theta) * x_axis + math.sin(theta) * y_axis,
        x_axis,
    )
    r_front = (section["Front"] - center).length
    r_right = (section["Right"] - center).length
    r_back = (section["Back"] - center).length
    r_left = (section["Left"] - center).length
    radius = _smooth_periodic_radius(theta, r_front, r_right, r_back, r_left)
    return center + radial * radius

def _cage_longitudinal_axis(control_points):
    top_center = _average_points([
        control_points[direction][0]
        for direction in CONTROL_DIRECTIONS
    ])
    bottom_center = _average_points([
        control_points[direction][-1]
        for direction in CONTROL_DIRECTIONS
    ])
    return _safe_normalized(
        bottom_center - top_center,
        Vector((0.0, 0.0, -1.0)),
    )


def _surface_height_parameters(surface_points, control_points):
    top_center = _average_points([
        control_points[direction][0]
        for direction in CONTROL_DIRECTIONS
    ])
    bottom_center = _average_points([
        control_points[direction][-1]
        for direction in CONTROL_DIRECTIONS
    ])
    vertical = bottom_center - top_center
    if vertical.length_squared <= EPSILON:
        return [0.0] * len(surface_points)
    return [
        clamp((point - top_center).dot(vertical) / vertical.length_squared, 0.0, 1.0)
        for point in surface_points
    ]


def _cage_column_geometry(control_points, row_v, column, columns, closed_loop):
    """Return polar geometry for one fitted-ring column.

    This is intentionally angle/radius based.  The old implementation sampled
    Front->Right->Back->Left as a four-edge polyline, which made every ring a
    diamond no matter how many columns were requested.
    """
    section, _unused_center = _cage_section(control_points, row_v)
    center, x_axis, y_axis = _section_axes(section)

    if closed_loop:
        u = column / columns
        theta = -math.pi * 0.5 + math.tau * u
    else:
        u = column / max(columns - 1, 1)
        theta = -math.pi * 0.5 + math.pi * 1.5 * clamp(u, 0.0, 1.0)

    radial = _safe_normalized(
        math.cos(theta) * x_axis + math.sin(theta) * y_axis,
        x_axis,
    )

    r_front = (section["Front"] - center).length
    r_right = (section["Right"] - center).length
    r_back = (section["Back"] - center).length
    r_left = (section["Left"] - center).length
    cage_radius = _smooth_periodic_radius(
        theta, r_front, r_right, r_back, r_left
    )
    cage_point = center + radial * cage_radius
    return section, center, u, cage_point, radial, cage_radius


def _clamp_point_to_cage(
    point, control_points, row_v, column, columns, bone_inset, closed_loop
):
    _section, center, _u, _cage_point, radial, cage_radius = (
        _cage_column_geometry(
            control_points, row_v, column, columns, closed_loop
        )
    )
    maximum = cage_radius * (1.0 - clamp(bone_inset, 0.0, 0.95))
    radius = clamp((point - center).dot(radial), 0.0, maximum)
    return center + radial * radius

def _expanded_slice_indices(heights, row_h, slice_thickness, minimum_count=4):
    thickness = clamp(float(slice_thickness), 0.001, 0.5)
    selected = []
    for _attempt in range(6):
        selected = [
            index for index, height in enumerate(heights)
            if abs(height - row_h) <= thickness
        ]
        if len(selected) >= minimum_count:
            return selected, thickness
        if thickness >= 0.5:
            break
        thickness = min(thickness * 1.75, 0.5)

    if heights:
        nearest_count = min(len(heights), max(minimum_count, 8))
        selected = sorted(
            range(len(heights)),
            key=lambda index: abs(heights[index] - row_h),
        )[:nearest_count]
        if selected:
            thickness = max(
                thickness,
                max(abs(heights[index] - row_h) for index in selected) + 1.0e-6,
            )
    return selected, thickness


def _weighted_point(points, weighted_indices, fallback):
    total = sum(weight for _index, weight in weighted_indices)
    if total <= EPSILON:
        return Vector(fallback)
    result = Vector((0.0, 0.0, 0.0))
    for index, weight in weighted_indices:
        result += points[index] * weight
    return result / total


def _weighted_normal(normals, weighted_indices, fallback):
    if not normals or not weighted_indices:
        return _safe_normalized(fallback, Vector((1.0, 0.0, 0.0)))
    reference_index = max(weighted_indices, key=lambda item: item[1])[0]
    reference = _safe_normalized(normals[reference_index], fallback)
    result = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for index, weight in weighted_indices:
        normal = _safe_normalized(normals[index], reference)
        if normal.dot(reference) < 0.0:
            normal.negate()
        result += normal * weight
        total += weight
    if total > EPSILON:
        result /= total
    return _safe_normalized(result, reference)


def _fit_column_from_slice(
    center,
    radial,
    cage_radius,
    row_h,
    candidate_indices,
    effective_thickness,
    surface_points,
    surface_normals,
    surface_heights,
    fit_mode,
    bone_inset,
    ring_fit_strength,
):
    """Fit one column by radius, not by averaging world-space positions.

    Averaging coordinates first can pull the fitted point sideways and then
    project it back to the current ray.  Here the point cloud contributes only
    a radius along the current polar direction, which preserves a smooth
    theta -> r(theta) ring.
    """
    angular_threshold = 0.35
    candidates = []
    for index in candidate_indices:
        delta = surface_points[index] - center
        length = delta.length
        if length <= EPSILON:
            continue
        radius = delta.dot(radial)
        direction_score = delta.dot(radial) / length
        if radius <= 0.0 or direction_score <= angular_threshold:
            continue
        height_score = clamp(
            1.0 - abs(surface_heights[index] - row_h)
            / max(effective_thickness, EPSILON),
            0.001,
            1.0,
        )
        weight = max(direction_score * height_score, EPSILON)
        candidates.append((index, radius, weight))

    maximum_radius = cage_radius * (
        1.0 - clamp(float(bone_inset), 0.0, 0.95)
    )
    cage_inner = center + radial * maximum_radius
    if not candidates:
        return cage_inner, radial.copy()

    mode = str(fit_mode).upper()
    normal_candidates = []

    if mode == 'MEDIAN':
        radii = sorted(candidate[1] for candidate in candidates)
        middle = len(radii) // 2
        fit_radius = (
            radii[middle]
            if len(radii) % 2
            else (radii[middle - 1] + radii[middle]) * 0.5
        )
        normal_candidates = [
            (index, weight / (1.0 + abs(radius - fit_radius)))
            for index, radius, weight in candidates
        ]
    else:
        selected = candidates
        if mode == 'OUTER_ENVELOPE':
            top_k = min(5, len(candidates))
            selected = sorted(
                candidates, key=lambda item: item[1], reverse=True
            )[:top_k]

        total_weight = sum(weight for _index, _radius, weight in selected)
        if total_weight <= EPSILON:
            fit_radius = maximum_radius
        else:
            fit_radius = sum(
                radius * weight for _index, radius, weight in selected
            ) / total_weight
        normal_candidates = [
            (index, weight) for index, _radius, weight in selected
        ]

    fit_radius = clamp(fit_radius, 0.0, maximum_radius)
    final_radius = _lerp_float(
        maximum_radius,
        fit_radius,
        clamp(float(ring_fit_strength), 0.0, 1.0),
    )
    final_point = center + radial * final_radius
    final_normal = _weighted_normal(
        surface_normals, normal_candidates, radial
    )
    return final_point, final_normal

def _smooth_normal_triplet(previous, current, following):
    previous = Vector(previous)
    current = Vector(current)
    following = Vector(following)
    if previous.dot(current) < 0.0:
        previous.negate()
    if following.dot(current) < 0.0:
        following.negate()
    return _safe_normalized(
        previous * 0.25 + current * 0.5 + following * 0.25,
        current,
    )


def _smooth_bone_chain_grid(
    samples,
    normals,
    control_points,
    columns,
    bone_inset,
    closed_loop,
    passes=BONE_CHAIN_SMOOTHING_PASSES,
):
    """Filter each vertical bone chain, preserving its first and last joint."""
    rows = len(samples)
    if rows < 3:
        return samples, normals
    result = [[Vector(point).copy() for point in row] for row in samples]
    normal_result = [[Vector(normal).copy() for normal in row] for row in normals]
    for _pass in range(max(int(passes), 1)):
        previous_points = result
        previous_normals = normal_result
        result = [[point.copy() for point in row] for row in previous_points]
        normal_result = [[normal.copy() for normal in row] for row in previous_normals]
        for row in range(1, rows - 1):
            row_v = row / (rows - 1)
            for column in range(columns):
                point = (
                    previous_points[row - 1][column] * 0.25
                    + previous_points[row][column] * 0.50
                    + previous_points[row + 1][column] * 0.25
                )
                result[row][column] = _clamp_point_to_cage(
                    point,
                    control_points,
                    row_v,
                    column,
                    columns,
                    bone_inset,
                    closed_loop,
                )
                normal_result[row][column] = _smooth_normal_triplet(
                    previous_normals[row - 1][column],
                    previous_normals[row][column],
                    previous_normals[row + 1][column],
                )
    return result, normal_result


def sample_cage_row(
    control_points,
    row_v,
    columns,
    bone_inset,
    closed_loop=True,
    surface_points=None,
    reference_control_points=None,
    surface_heights=None,
    slice_band=0.08,
    surface_normals=None,
    fit_mode='OUTER_ENVELOPE',
    ring_fit_strength=1.0,
    return_normals=False,
):
    """Fit one polar bone ring to the selected point-cloud height slice.

    The ring is smoothed as radii along fixed theta directions.  Smoothing
    points directly can reintroduce polygonal bias, especially when the old
    four-control boundary is not meant to be the final shape.
    """
    del reference_control_points
    if columns < 4:
        raise ValueError("环向骨骼数必须至少为 4")
    surface_points = surface_points or []
    surface_normals = surface_normals or []
    if len(surface_normals) != len(surface_points):
        surface_normals = []
    if surface_heights is None and surface_points:
        surface_heights = _surface_height_parameters(
            surface_points, control_points
        )
    surface_heights = surface_heights or []
    candidate_indices, effective_thickness = _expanded_slice_indices(
        surface_heights, row_v, slice_band
    )

    samples = []
    normals = []
    centers = []
    radials = []
    max_radii = []
    fitted_radii = []

    for column in range(columns):
        _section, center, _u, _cage_point, radial, cage_radius = (
            _cage_column_geometry(
                control_points, row_v, column, columns, closed_loop
            )
        )
        point, normal = _fit_column_from_slice(
            center,
            radial,
            cage_radius,
            row_v,
            candidate_indices,
            effective_thickness,
            surface_points,
            surface_normals,
            surface_heights,
            fit_mode,
            bone_inset,
            ring_fit_strength,
        )
        maximum = cage_radius * (1.0 - clamp(float(bone_inset), 0.0, 0.95))
        radius = clamp((point - center).dot(radial), 0.0, maximum)
        centers.append(center)
        radials.append(radial)
        max_radii.append(maximum)
        fitted_radii.append(radius)
        samples.append(center + radial * radius)
        normals.append(normal)

    # One circumferential smoothing pass on r(theta), followed by clamping.
    smoothed_points = []
    smoothed_normals = []
    for column in range(columns):
        previous = (column - 1) % columns if closed_loop else max(column - 1, 0)
        following = (column + 1) % columns if closed_loop else min(column + 1, columns - 1)
        radius = (
            fitted_radii[previous] * 0.25
            + fitted_radii[column] * 0.5
            + fitted_radii[following] * 0.25
        )
        radius = clamp(radius, 0.0, max_radii[column])
        point = centers[column] + radials[column] * radius
        smoothed_points.append(_clamp_point_to_cage(
            point,
            control_points,
            row_v,
            column,
            columns,
            bone_inset,
            closed_loop,
        ))
        smoothed_normals.append(_smooth_normal_triplet(
            normals[previous], normals[column], normals[following]
        ))

    if return_normals:
        return smoothed_points, smoothed_normals
    return smoothed_points

def build_cage_samples(
    control_points,
    rows,
    columns,
    bone_inset,
    closed_loop=True,
    surface_points=None,
    reference_control_points=None,
    surface_normals=None,
    slice_thickness=0.08,
    fit_mode='OUTER_ENVELOPE',
    ring_fit_strength=1.0,
    longitudinal_smoothing=True,
    smooth_bone_chains=False,
    return_normals=False,
):
    if rows < 2:
        raise ValueError("纵向层数必须至少为 2")
    del reference_control_points
    surface_points = surface_points or []
    surface_normals = surface_normals or []
    if len(surface_normals) != len(surface_points):
        surface_normals = []
    surface_heights = (
        _surface_height_parameters(surface_points, control_points)
        if surface_points else []
    )
    samples = []
    normals = []
    for row in range(rows):
        row_points, row_normals = sample_cage_row(
            control_points,
            row / (rows - 1),
            columns,
            bone_inset,
            closed_loop,
            surface_points,
            None,
            surface_heights,
            slice_thickness,
            surface_normals,
            fit_mode,
            ring_fit_strength,
            True,
        )
        samples.append(row_points)
        normals.append(row_normals)

    if longitudinal_smoothing and rows > 2:
        smoothed_samples = [[point.copy() for point in row] for row in samples]
        smoothed_normals = [[normal.copy() for normal in row] for row in normals]
        for row in range(1, rows - 1):
            row_v = row / (rows - 1)
            for column in range(columns):
                point = (
                    samples[row - 1][column] * 0.2
                    + samples[row][column] * 0.6
                    + samples[row + 1][column] * 0.2
                )
                smoothed_samples[row][column] = _clamp_point_to_cage(
                    point,
                    control_points,
                    row_v,
                    column,
                    columns,
                    bone_inset,
                    closed_loop,
                )
                previous_normal = normals[row - 1][column].copy()
                current_normal = normals[row][column].copy()
                following_normal = normals[row + 1][column].copy()
                if previous_normal.dot(current_normal) < 0.0:
                    previous_normal.negate()
                if following_normal.dot(current_normal) < 0.0:
                    following_normal.negate()
                smoothed_normals[row][column] = _safe_normalized(
                    previous_normal * 0.2
                    + current_normal * 0.6
                    + following_normal * 0.2,
                    current_normal,
                )
        samples = smoothed_samples
        normals = smoothed_normals

    if smooth_bone_chains:
        samples, normals = _smooth_bone_chain_grid(
            samples,
            normals,
            control_points,
            columns,
            bone_inset,
            closed_loop,
        )

    if return_normals:
        return samples, normals
    return samples


# ---------------------------------------------------------------------------
# Bone generation
# ---------------------------------------------------------------------------


def cleanup_generated_bones(armature, prefix):
    """Remove prefix-matching edit bones. Armature must be in Edit Mode."""
    if armature.type != 'ARMATURE' or armature.mode != 'EDIT':
        raise ValueError("清理骨骼时 Armature 必须处于 Edit Mode")
    removed = 0
    for bone in list(armature.data.edit_bones):
        if bone.name.startswith(prefix):
            armature.data.edit_bones.remove(bone)
            removed += 1
    return removed


def _ring_tangent(samples, row, column, closed_loop):
    columns = len(samples[row])
    if closed_loop:
        previous = samples[row][(column - 1) % columns]
        following = samples[row][(column + 1) % columns]
    elif column == 0:
        previous = samples[row][0]
        following = samples[row][1]
    elif column == columns - 1:
        previous = samples[row][-2]
        following = samples[row][-1]
    else:
        previous = samples[row][column - 1]
        following = samples[row][column + 1]
    return following - previous


def _align_bone_roll(bone, tangent):
    longitudinal = bone.tail - bone.head
    if longitudinal.length_squared <= EPSILON:
        return
    tangent = Vector(tangent)
    tangent -= longitudinal.normalized() * tangent.dot(longitudinal.normalized())
    if tangent.length_squared > EPSILON:
        bone.align_roll(tangent.normalized())


def _point_at_polyline_arc(points, cumulative, distance):
    distance = clamp(distance, 0.0, cumulative[-1])
    for segment in range(len(points) - 1):
        if distance <= cumulative[segment + 1] + EPSILON:
            span = cumulative[segment + 1] - cumulative[segment]
            if span <= EPSILON:
                return points[segment].copy()
            factor = (distance - cumulative[segment]) / span
            return points[segment].lerp(points[segment + 1], factor)
    return points[-1].copy()


def _equal_chord_chain(polyline, bone_count):
    """Return bone_count + 1 connected curve points with equal chord lengths."""
    points = []
    for point in polyline:
        point = Vector(point)
        if not points or (point - points[-1]).length_squared > EPSILON:
            points.append(point.copy())
    if len(points) < 2:
        origin = points[0] if points else Vector((0.0, 0.0, 0.0))
        return [origin + Vector((0.0, 0.0, -0.001 * i)) for i in range(bone_count + 1)]

    cumulative_arc = [0.0]
    for index in range(len(points) - 1):
        cumulative_arc.append(
            cumulative_arc[-1] + (points[index + 1] - points[index]).length
        )
    total_arc = cumulative_arc[-1]
    if total_arc <= EPSILON:
        return [points[0].copy() for _index in range(bone_count + 1)]

    # Start with equal arc-length positions, then repeatedly redistribute the
    # arc parameters according to cumulative chord length. The fixed point of
    # this mapping has equal straight bone lengths while every joint remains
    # on the original sampled curve and both endpoints stay fixed.
    parameters = [
        total_arc * index / bone_count
        for index in range(bone_count + 1)
    ]
    for _iteration in range(200):
        chain = [
            _point_at_polyline_arc(points, cumulative_arc, parameter)
            for parameter in parameters
        ]
        lengths = [
            (chain[index + 1] - chain[index]).length
            for index in range(bone_count)
        ]
        if max(lengths) - min(lengths) <= max(total_arc * 1.0e-10, 1.0e-9):
            return chain

        cumulative_chord = [0.0]
        for length in lengths:
            cumulative_chord.append(cumulative_chord[-1] + length)
        total_chord = cumulative_chord[-1]
        if total_chord <= EPSILON:
            break

        redistributed = [0.0]
        for index in range(1, bone_count):
            target = total_chord * index / bone_count
            segment = 0
            while (
                segment < bone_count - 1
                and cumulative_chord[segment + 1] < target
            ):
                segment += 1
            span = cumulative_chord[segment + 1] - cumulative_chord[segment]
            factor = 0.0 if span <= EPSILON else (
                (target - cumulative_chord[segment]) / span
            )
            redistributed.append(
                parameters[segment]
                + (parameters[segment + 1] - parameters[segment]) * factor
            )
        redistributed.append(total_arc)

        # Damping prevents oscillation when a joint crosses an original
        # polyline corner. Endpoints remain exactly pinned.
        parameters = [
            0.0 if index == 0 else (
                total_arc if index == bone_count else
                parameters[index] * 0.2 + redistributed[index] * 0.8
            )
            for index in range(bone_count + 1)
        ]

    chain = [
        _point_at_polyline_arc(points, cumulative_arc, parameter)
        for parameter in parameters
    ]
    spread = max(
        (chain[index + 1] - chain[index]).length
        for index in range(bone_count)
    ) - min(
        (chain[index + 1] - chain[index]).length
        for index in range(bone_count)
    )
    if spread > max(total_arc * 1.0e-5, 1.0e-6):
        # Extremely pathological curves fall back to a straight in-cage chain
        # so connection, equal size, and the bottom boundary remain strict.
        return [
            points[0].lerp(points[-1], index / bone_count)
            for index in range(bone_count + 1)
        ]
    return chain


def _smooth_open_bone_chain(points, passes=BONE_CHAIN_SMOOTHING_PASSES):
    """Smooth generated joint positions without moving either chain endpoint."""
    result = [Vector(point).copy() for point in points]
    if len(result) < 3:
        return result
    for _pass in range(max(int(passes), 1)):
        previous = result
        result = [point.copy() for point in previous]
        for index in range(1, len(previous) - 1):
            result[index] = (
                previous[index - 1] * 0.25
                + previous[index] * 0.50
                + previous[index + 1] * 0.25
            )
    return result


def _interpolate_vector_on_polyline(points, vectors, target, fallback):
    if not vectors or len(vectors) != len(points):
        return _safe_normalized(fallback, Vector((1.0, 0.0, 0.0)))
    best_distance = float('inf')
    best_vector = Vector(vectors[0])
    for index in range(len(points) - 1):
        start = points[index]
        segment = points[index + 1] - start
        if segment.length_squared <= EPSILON:
            factor = 0.0
        else:
            factor = clamp(
                (target - start).dot(segment) / segment.length_squared,
                0.0,
                1.0,
            )
        closest = start + segment * factor
        distance = (target - closest).length_squared
        if distance < best_distance:
            best_distance = distance
            first = Vector(vectors[index])
            second = Vector(vectors[index + 1])
            if first.dot(second) < 0.0:
                second.negate()
            best_vector = first.lerp(second, factor)
    return _safe_normalized(best_vector, fallback)


def _get_or_create_bone_collection(armature_data, collection_name):
    """Return an existing Bone Collection, or create the requested top-level one."""
    name = str(collection_name or "").strip()
    if not name:
        return None
    collection = armature_data.collections.get(name)
    if collection is None:
        collection = armature_data.collections.new(name)
    return collection


def create_bones_from_samples(
    armature,
    samples,
    parent_bone_name,
    prefix,
    generate_end_bones,
    closed_loop=True,
    sample_normals=None,
    bone_collection_name="",
):
    # Kept in the public signature for file/API compatibility. End bones are
    # intentionally not generated anymore.
    del generate_end_bones, closed_loop
    if armature.type != 'ARMATURE' or armature.mode != 'EDIT':
        raise ValueError("创建骨骼时 Armature 必须处于编辑模式")
    rows = len(samples)
    columns = len(samples[0]) if samples else 0
    if rows < 2 or columns < 4:
        raise ValueError("骨骼采样数据无效")

    edit_bones = armature.data.edit_bones
    bone_collection = _get_or_create_bone_collection(
        armature.data,
        bone_collection_name,
    )
    parent_bone = edit_bones.get(parent_bone_name)
    if parent_bone is None:
        raise ValueError(f"找不到父骨骼：{parent_bone_name}")

    inverse = armature.matrix_world.inverted()
    normal_to_local = armature.matrix_world.to_3x3().transposed()
    local_samples = [
        [inverse @ point for point in row_points]
        for row_points in samples
    ]

    requested_names = {
        _bone_name(prefix, column, row)
        for column in range(columns)
        for row in range(rows)
    }
    temporary_names = {
        _temporary_bone_name(prefix, column, row)
        for column in range(columns)
        for row in range(rows)
    }
    conflicts = sorted(
        name for name in requested_names | temporary_names if edit_bones.get(name)
    )
    if conflicts:
        raise ValueError(
            "已存在同名生成骨骼；请启用“清除已有生成骨骼”："
            + ", ".join(conflicts[:4])
        )

    chains = [
        _equal_chord_chain(
            [local_samples[row][column] for row in range(rows)],
            rows,
        )
        for column in range(columns)
    ]
    local_sample_normals = None
    if sample_normals:
        local_sample_normals = [
            [
                _safe_normalized(
                    normal_to_local @ sample_normals[row][column],
                    Vector((1.0, 0.0, 0.0)),
                )
                for column in range(columns)
            ]
            for row in range(rows)
        ]

    created = []
    renamed_bones = []
    for column in range(columns):
        previous_bone = None
        chain = chains[column]
        source_polyline = [local_samples[row][column] for row in range(rows)]
        source_normals = (
            [local_sample_normals[row][column] for row in range(rows)]
            if local_sample_normals else []
        )
        for row in range(rows):
            final_name = _bone_name(prefix, column, row)
            # Geometry and roll are created under a physical column name.
            # Optional .L/.R suffixes are applied only after all axes are final.
            bone = edit_bones.new(_temporary_bone_name(prefix, column, row))
            bone.head = chain[row]
            bone.tail = chain[row + 1]
            bone.parent = parent_bone if row == 0 else previous_bone
            bone.use_connect = row > 0
            bone.use_deform = True
            if bone_collection is not None:
                bone_collection.assign(bone)

            # Bone local +Z follows the fitted selected-vertex normal. The
            # target is interpolated at the resampled chain head; align_roll
            # projects only the unavoidable component parallel to bone +Y.
            fitted_normal = _interpolate_vector_on_polyline(
                source_polyline,
                source_normals,
                chain[row],
                Vector((1.0, 0.0, 0.0)),
            )
            _align_bone_roll(bone, fitted_normal)
            created.append(final_name)
            renamed_bones.append((bone, final_name))
            previous_bone = bone

    for bone, final_name in renamed_bones:
        bone.name = final_name
        if bone.name != final_name:
            raise ValueError(f"Could not assign generated bone name: {final_name}")
    return created


# ---------------------------------------------------------------------------
# Cage parameterization and weights
# ---------------------------------------------------------------------------


def compute_vertex_cage_parameters(
    vertex_world_pos,
    control_points,
    closed_loop=True,
):
    top_center = _average_points([
        control_points[direction][0] for direction in CONTROL_DIRECTIONS
    ])
    bottom_center = _average_points([
        control_points[direction][-1] for direction in CONTROL_DIRECTIONS
    ])
    vertical = bottom_center - top_center
    if vertical.length_squared <= EPSILON:
        raise ValueError("控制笼顶部和底部中心重合，无法计算纵向参数")

    point = Vector(vertex_world_pos)
    h = clamp(
        (point - top_center).dot(vertical) / vertical.length_squared,
        0.0,
        1.0,
    )
    section, center = _cage_section(control_points, h)

    x_axis = section["Right"] - section["Left"]
    x_axis = _safe_normalized(x_axis, Vector((1.0, 0.0, 0.0)))
    y_axis = section["Back"] - section["Front"]
    y_axis -= x_axis * y_axis.dot(x_axis)
    y_axis = _safe_normalized(y_axis, Vector((0.0, 1.0, 0.0)))

    delta = point - center
    x = delta.dot(x_axis)
    y = delta.dot(y_axis)
    if x * x + y * y <= EPSILON:
        # The angle is undefined near the center. Use the closest cardinal point.
        distances = [
            (point - section[direction]).length_squared
            for direction in CONTROL_DIRECTIONS
        ]
        cardinal = min(range(4), key=lambda index: distances[index])
        u = cardinal / (4.0 if closed_loop else 3.0)
        return h, clamp(u, 0.0, 1.0)

    angle = math.atan2(y, x)
    phase = (angle + math.pi * 0.5) % math.tau
    if closed_loop:
        u = phase / math.tau
    else:
        open_span = math.pi * 1.5
        if phase <= open_span:
            u = phase / open_span
        else:
            # The omitted Left->Front quadrant maps to its nearest open endpoint.
            distance_to_left = phase - open_span
            distance_to_front = math.tau - phase
            u = 1.0 if distance_to_left <= distance_to_front else 0.0
    return h, clamp(u, 0.0, 1.0)


def _vertical_weights(t, curve):
    t = clamp(t, 0.0, 1.0)
    if curve == 'COSINE':
        first = (math.cos(math.pi * t) + 1.0) * 0.5
    elif curve == 'SHARP':
        first = clamp(
            (1.0 - math.tanh(2.0 * math.pi * abs(t) - math.pi)) * 0.5,
            0.0,
            1.0,
        )
    else:
        first = 1.0 - t
    return first, 1.0 - first


def compute_vertex_weights(
    vertex_world_pos,
    h,
    u,
    rows,
    columns,
    prefix,
    settings,
):
    del vertex_world_pos  # Kept in the public signature for future projection methods.
    row_value = clamp(h, 0.0, 1.0) * (rows - 1)
    row0 = min(int(math.floor(row_value)), rows - 1)
    row1 = min(row0 + 1, rows - 1)
    t = row_value - math.floor(row_value)
    row_weight0, row_weight1 = _vertical_weights(t, settings.weight_curve)

    if settings.closed_loop:
        column_value = (u % 1.0) * columns
        base_column = int(math.floor(column_value))
        column0 = base_column % columns
        column1 = (column0 + 1) % columns
        s = column_value - base_column
    else:
        column_value = clamp(u, 0.0, 1.0) * (columns - 1)
        column0 = min(int(math.floor(column_value)), columns - 1)
        column1 = min(column0 + 1, columns - 1)
        s = column_value - math.floor(column_value)

    column_weight1 = smoothstep(s)
    column_weight0 = 1.0 - column_weight1
    weights = {}
    contributions = (
        (row0, column0, row_weight0 * column_weight0),
        (row0, column1, row_weight0 * column_weight1),
        (row1, column0, row_weight1 * column_weight0),
        (row1, column1, row_weight1 * column_weight1),
    )
    for row, column, value in contributions:
        name = _bone_name(prefix, column, row)
        weights[name] = weights.get(name, 0.0) + value

    parent_weight = 0.0
    if settings.top_blend_to_parent and h < settings.top_blend_height:
        blend_position = h / max(settings.top_blend_height, EPSILON)
        parent_weight = 1.0 - smoothstep(blend_position)
        skirt_scale = 1.0 - parent_weight
        weights = {name: value * skirt_scale for name, value in weights.items()}

    weights = {
        name: value for name, value in weights.items()
        if value >= MIN_WEIGHT
    }

    # Parent is guaranteed a slot, as selected in the design interview.
    parent_is_active = parent_weight >= MIN_WEIGHT
    skirt_slots = settings.max_influences - (1 if parent_is_active else 0)
    sorted_skirt = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    weights = dict(sorted_skirt[:max(skirt_slots, 0)])
    if parent_is_active:
        weights[PARENT_WEIGHT_KEY] = parent_weight

    if settings.normalize_weights:
        total = sum(weights.values())
        if total > EPSILON:
            weights = {name: value / total for name, value in weights.items()}
    return weights


def apply_weights(
    mesh_obj,
    selected_indices,
    weights,
    prefix,
    parent_bone_name,
    clear_existing_generated_groups=True,
    top_blend_to_parent=True,
):
    if mesh_obj.type != 'MESH' or mesh_obj.mode != 'OBJECT':
        raise ValueError("写入权重时 Target Mesh 必须处于 Object Mode")
    selected_indices = sorted(set(selected_indices))
    if not selected_indices:
        raise ValueError("没有可用于赋权的已记录顶点")

    prefix_groups = [
        group for group in mesh_obj.vertex_groups
        if group.name.startswith(prefix)
    ]
    if clear_existing_generated_groups:
        for group in prefix_groups:
            mesh_obj.vertex_groups.remove(group)
    else:
        for group in prefix_groups:
            try:
                group.remove(selected_indices)
            except RuntimeError:
                pass

    needed_names = {
        name
        for vertex_weights in weights.values()
        for name in vertex_weights
        if name != PARENT_WEIGHT_KEY
    }
    groups = {}
    for name in needed_names:
        group = mesh_obj.vertex_groups.get(name)
        if group is None:
            group = mesh_obj.vertex_groups.new(name=name)
        groups[name] = group

    parent_group = None
    if top_blend_to_parent:
        parent_group = mesh_obj.vertex_groups.get(parent_bone_name)
        if parent_group is None:
            parent_group = mesh_obj.vertex_groups.new(name=parent_bone_name)
        try:
            parent_group.remove(selected_indices)
        except RuntimeError:
            pass

    assignment_count = 0
    for vertex_index in selected_indices:
        vertex_weights = weights.get(vertex_index, {})
        for name, value in vertex_weights.items():
            if value < MIN_WEIGHT:
                continue
            if name == PARENT_WEIGHT_KEY:
                if parent_group is not None:
                    parent_group.add([vertex_index], value, 'REPLACE')
                    assignment_count += 1
            else:
                groups[name].add([vertex_index], value, 'REPLACE')
                assignment_count += 1
    return assignment_count


def _ensure_armature_modifier(mesh_obj, armature_obj):
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object == armature_obj:
            return modifier, False
    modifier = mesh_obj.modifiers.new(
        name=f"{_prefix_stem(armature_obj.name)} Armature",
        type='ARMATURE',
    )
    modifier.object = armature_obj
    return modifier, True


# ---------------------------------------------------------------------------
# Debug curves
# ---------------------------------------------------------------------------


def _new_poly_curve(name, points, collection, cyclic, bevel_depth, mesh_name):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 1
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 1
    spline = curve_data.splines.new(type='POLY')
    spline.points.add(len(points) - 1)
    for index, point in enumerate(points):
        spline.points[index].co = (*point, 1.0)
    spline.use_cyclic_u = cyclic

    obj = bpy.data.objects.new(name, curve_data)
    collection.objects.link(obj)
    obj.show_in_front = True
    obj.hide_render = True
    obj[PROP_TARGET] = mesh_name
    obj[PROP_KIND] = "DEBUG"
    return obj


def create_debug_curves(
    samples,
    prefix="AutoSkirt_",
    mesh_name="Mesh",
    closed_loop=True,
):
    _delete_debug_objects(prefix, mesh_name)
    collection = get_or_create_collection(_debug_collection_name(prefix, mesh_name))
    rows = len(samples)
    columns = len(samples[0])
    segment_lengths = [
        (samples[row + 1][column] - samples[row][column]).length
        for row in range(rows - 1)
        for column in range(columns)
    ]
    typical_length = (
        sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.1
    )
    bevel_depth = max(typical_length * 0.015, 0.0001)

    objects = []
    for row, points in enumerate(samples):
        objects.append(_new_poly_curve(
            f"{prefix}DEBUG_Row_{row:02d}",
            points,
            collection,
            closed_loop,
            bevel_depth,
            mesh_name,
        ))
    for column in range(columns):
        points = [samples[row][column] for row in range(rows)]
        objects.append(_new_poly_curve(
            f"{prefix}DEBUG_Column_{column:02d}",
            points,
            collection,
            False,
            bevel_depth,
            mesh_name,
        ))
    return objects



def _uniform_proxy_row_samples(samples):
    """Resample proxy rows by average vertical arc length.

    The bone samples remain untouched.  Only the display proxy is reparameterized
    so its horizontal loop edges are evenly distributed down the fitted shape.
    """
    rows = len(samples)
    columns = len(samples[0]) if rows else 0
    if rows < 3 or columns < 1:
        return [[Vector(point) for point in row] for row in samples]

    cumulative = [0.0]
    for row in range(1, rows):
        lengths = [
            (Vector(samples[row][column]) - Vector(samples[row - 1][column])).length
            for column in range(columns)
        ]
        cumulative.append(cumulative[-1] + sum(lengths) / len(lengths))

    total_length = cumulative[-1]
    if total_length <= EPSILON:
        return [[Vector(point) for point in row] for row in samples]

    result = []
    source_row = 0
    for target_row in range(rows):
        target_distance = total_length * target_row / (rows - 1)
        while (
            source_row < rows - 2
            and cumulative[source_row + 1] < target_distance
        ):
            source_row += 1
        start = cumulative[source_row]
        end = cumulative[source_row + 1]
        factor = (
            (target_distance - start) / (end - start)
            if end - start > EPSILON else 0.0
        )
        result.append([
            Vector(samples[source_row][column]).lerp(
                Vector(samples[source_row + 1][column]),
                factor,
            )
            for column in range(columns)
        ])
    return result


def _proxy_grid_vertex_normals(samples, closed_loop=True):
    """Calculate the actual proxy-grid vertex normals from its face winding."""
    rows = len(samples)
    columns = len(samples[0]) if rows else 0
    if rows < 2 or columns < 2:
        return []

    normals = []
    for row in range(rows):
        previous_row = max(row - 1, 0)
        next_row = min(row + 1, rows - 1)
        row_center = _average_points(samples[row])
        row_normals = []
        for column in range(columns):
            if closed_loop:
                previous_column = (column - 1) % columns
                next_column = (column + 1) % columns
            else:
                previous_column = max(column - 1, 0)
                next_column = min(column + 1, columns - 1)

            horizontal = (
                Vector(samples[row][next_column])
                - Vector(samples[row][previous_column])
            )
            vertical = (
                Vector(samples[next_row][column])
                - Vector(samples[previous_row][column])
            )
            # This cross product follows create_proxy_mesh() face winding.
            # Its fallback also points toward the proxy interior.
            inward = horizontal.cross(vertical)
            inward = _safe_normalized(
                inward,
                row_center - Vector(samples[row][column]),
            )
            row_normals.append(inward)
        normals.append(row_normals)
    return normals


def create_proxy_mesh(
    samples,
    prefix="AutoSkirt_",
    mesh_name="Mesh",
    closed_loop=True,
):
    _delete_proxy_objects(prefix, mesh_name)
    collection = get_or_create_collection(_proxy_collection_name(prefix, mesh_name))
    proxy_samples = _uniform_proxy_row_samples(samples)
    rows = len(proxy_samples)
    columns = len(proxy_samples[0]) if rows else 0
    if rows < 2 or columns < 2:
        return None

    verts = [
        tuple(proxy_samples[row][column])
        for row in range(rows)
        for column in range(columns)
    ]
    faces = []
    column_limit = columns if closed_loop else columns - 1
    for row in range(rows - 1):
        for column in range(column_limit):
            next_column = (column + 1) % columns
            if not closed_loop and next_column <= column:
                continue
            faces.append((
                row * columns + column,
                row * columns + next_column,
                (row + 1) * columns + next_column,
                (row + 1) * columns + column,
            ))

    mesh_data = bpy.data.meshes.new(f"{prefix}PROXY_{mesh_name}_Mesh")
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    obj = bpy.data.objects.new(f"{prefix}PROXY_{mesh_name}", mesh_data)
    collection.objects.link(obj)
    obj.display_type = 'WIRE'
    obj.show_wire = True
    obj.show_in_front = True
    obj.hide_render = True
    obj[PROP_TARGET] = mesh_name
    obj[PROP_KIND] = PROP_PROXY_KIND
    return obj


def apply_proxy_mesh_weights(
    proxy_obj,
    armature_obj,
    control_points,
    rows,
    columns,
    prefix,
    parent_bone_name,
    settings,
):
    """Give the generated proxy the same cage-based bone weights as the target."""
    if proxy_obj is None or proxy_obj.type != 'MESH':
        return 0

    indices = list(range(len(proxy_obj.data.vertices)))
    weights = {}
    matrix_world = proxy_obj.matrix_world
    for index in indices:
        vertex_world = matrix_world @ proxy_obj.data.vertices[index].co
        h, u = compute_vertex_cage_parameters(
            vertex_world,
            control_points,
            True,
        )
        weights[index] = compute_vertex_weights(
            vertex_world,
            h,
            u,
            rows,
            columns,
            prefix,
            settings,
        )

    assignments = apply_weights(
        proxy_obj,
        indices,
        weights,
        prefix,
        parent_bone_name,
        True,
        settings.top_blend_to_parent,
    )
    _ensure_armature_modifier(proxy_obj, armature_obj)
    return assignments


def _periodic_cardinal_scalar(values, raw_u):
    phase = (float(raw_u) % 1.0) * 4.0
    first = int(math.floor(phase)) % 4
    factor = (1.0 - math.cos((phase - math.floor(phase)) * math.pi)) * 0.5
    return _lerp_float(values[first], values[(first + 1) % 4], factor)


def _periodic_cardinal_point(points, raw_u):
    """Clamped periodic Catmull-Rom through Front/Right/Back/Left."""
    phase = (float(raw_u) % 1.0) * 4.0
    segment = int(math.floor(phase)) % 4
    t = phase - math.floor(phase)
    p0 = Vector(points[(segment - 1) % 4])
    p1 = Vector(points[segment])
    p2 = Vector(points[(segment + 1) % 4])
    p3 = Vector(points[(segment + 2) % 4])
    t2 = t * t
    t3 = t2 * t
    value = (
        p1 * 2.0
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * t3
    ) * 0.5
    for axis in range(3):
        value[axis] = clamp(
            value[axis],
            min(p1[axis], p2[axis]),
            max(p1[axis], p2[axis]),
        )
    return value


def _apply_asymmetric_hem_offsets(profile, control_points, base_l3):
    up = profile["up"]
    right = profile["right"]
    back = profile["back"]
    direction_u = (0.0, 0.25, 0.5, 0.75)
    longitudinal = []
    radial_offsets = []
    for direction, raw_u in zip(CONTROL_DIRECTIONS, direction_u):
        current = Vector(control_points[direction][-1])
        baseline = Vector(base_l3.get(direction, current))
        offset = current - baseline
        radial = _radial_at_u(raw_u, right, back)
        longitudinal.append(offset.dot(up))
        radial_offsets.append(offset.dot(radial))

    adjusted = dict(profile)
    adjusted_points = []
    for index, point in enumerate(profile["points"]):
        raw_u = index / len(profile["points"])
        radial = _radial_at_u(raw_u, right, back)
        long_value = _periodic_cardinal_scalar(longitudinal, raw_u)
        radial_value = _periodic_cardinal_scalar(radial_offsets, raw_u)
        adjusted_points.append(
            Vector(point) + up * long_value + radial * radial_value
        )
    adjusted["points"] = [
        adjusted_points[(index - 1) % len(adjusted_points)] * 0.15
        + adjusted_points[index] * 0.70
        + adjusted_points[(index + 1) % len(adjusted_points)] * 0.15
        for index in range(len(adjusted_points))
    ]
    return adjusted


def _asym_ring_point(control_points, hem_profile, level, raw_u):
    if level == CONTROL_LEVELS - 1:
        return _profile_point_at(hem_profile, raw_u)
    points = [
        control_points[direction][level]
        for direction in CONTROL_DIRECTIONS
    ]
    return _periodic_cardinal_point(points, raw_u)


def _asym_control_surface_point(control_points, hem_profile, raw_u, local_v):
    rail_controls = [
        _asym_ring_point(control_points, hem_profile, level, raw_u)
        for level in range(CONTROL_LEVELS)
    ]
    return interpolate_vertical_control(rail_controls, local_v)


def _build_average_arc_mapping(control_points, hem_profile, resolution=128):
    resolution = max(int(resolution), 32)
    rings = [
        [
            _asym_ring_point(
                control_points,
                hem_profile,
                level,
                sample / resolution,
            )
            for sample in range(resolution)
        ]
        for level in range(CONTROL_LEVELS)
    ]
    cumulative = [0.0]
    for sample in range(resolution):
        following = (sample + 1) % resolution
        length = sum(
            (rings[level][following] - rings[level][sample]).length
            for level in range(CONTROL_LEVELS)
        ) / CONTROL_LEVELS
        cumulative.append(cumulative[-1] + max(length, EPSILON))
    return {
        "resolution": resolution,
        "cumulative": cumulative,
        "total": cumulative[-1],
    }


def _raw_u_from_arc_s(mapping, arc_s):
    target = (float(arc_s) % 1.0) * mapping["total"]
    cumulative = mapping["cumulative"]
    low = 0
    high = mapping["resolution"]
    while low + 1 < high:
        middle = (low + high) // 2
        if cumulative[middle] <= target:
            low = middle
        else:
            high = middle
    span = cumulative[low + 1] - cumulative[low]
    factor = 0.0 if span <= EPSILON else (
        (target - cumulative[low]) / span
    )
    return (low + factor) / mapping["resolution"]


def _arc_s_from_raw_u(mapping, raw_u):
    value = (float(raw_u) % 1.0) * mapping["resolution"]
    index = int(math.floor(value)) % mapping["resolution"]
    factor = value - math.floor(value)
    cumulative = mapping["cumulative"]
    distance = _lerp_float(
        cumulative[index],
        cumulative[index + 1],
        factor,
    )
    return distance / max(mapping["total"], EPSILON)


def _closest_polyline_parameter(point, polyline):
    points = [Vector(value) for value in polyline]
    if len(points) < 2:
        return 0.0
    cumulative = [0.0]
    for index in range(len(points) - 1):
        cumulative.append(
            cumulative[-1] + (points[index + 1] - points[index]).length
        )
    total = cumulative[-1]
    if total <= EPSILON:
        return 0.0

    best_distance = float('inf')
    best_arc = 0.0
    point = Vector(point)
    for index in range(len(points) - 1):
        segment = points[index + 1] - points[index]
        if segment.length_squared <= EPSILON:
            factor = 0.0
        else:
            factor = clamp(
                (point - points[index]).dot(segment) / segment.length_squared,
                0.0,
                1.0,
            )
        closest = points[index] + segment * factor
        distance = (point - closest).length_squared
        if distance < best_distance:
            best_distance = distance
            best_arc = cumulative[index] + segment.length * factor
    return clamp(best_arc / total, 0.0, 1.0)


def _initial_asym_parameters(
    surface_points,
    control_points,
    hem_profile,
    arc_mapping,
):
    center = hem_profile["center"]
    right = hem_profile["right"]
    back = hem_profile["back"]
    rail_cache = {}
    parameters = []
    for point in surface_points:
        raw_u = _raw_skirt_u(point, center, right, back)
        cache_index = int(round(raw_u * arc_mapping["resolution"])) % arc_mapping["resolution"]
        if cache_index not in rail_cache:
            cached_u = cache_index / arc_mapping["resolution"]
            rail_cache[cache_index] = [
                _asym_control_surface_point(
                    control_points,
                    hem_profile,
                    cached_u,
                    step / 32.0,
                )
                for step in range(33)
            ]
        local_v = _closest_polyline_parameter(
            point,
            rail_cache[cache_index],
        )
        parameters.append((
            _arc_s_from_raw_u(arc_mapping, raw_u),
            local_v,
        ))
    return parameters

def _fit_asym_dense_point(
    target,
    center,
    target_s,
    target_v,
    candidates,
    surface_points,
    surface_parameters,
    bone_inset,
    ring_fit_strength,
    up=None,
):
    """Fit one outer-envelope point in the horizontal cage plane.

    Separating height from radius is essential for strongly asymmetric hems:
    a chain being higher or lower must not shorten its top-view radius.
    """
    target = Vector(target)
    center = Vector(center)
    up = _safe_normalized(
        Vector(up) if up is not None else Vector((0.0, 0.0, 1.0)),
        Vector((0.0, 0.0, 1.0)),
    )
    slice_center = center + up * (target - center).dot(up)
    radial_vector = target - slice_center
    radial_vector -= up * radial_vector.dot(up)
    cage_radius = radial_vector.length
    if cage_radius <= EPSILON:
        return target
    radial = radial_vector / cage_radius
    maximum = cage_radius * (1.0 - clamp(float(bone_inset), 0.0, 0.95))

    band = 0.06
    nearby = []
    for _attempt in range(6):
        nearby = [
            index for index in candidates
            if abs(surface_parameters[index][1] - target_v) <= band
        ]
        if len(nearby) >= 4:
            break
        band = min(band * 1.7, 0.5)
    if not nearby:
        nearby = candidates

    scored = []
    for index in nearby:
        delta = Vector(surface_points[index]) - slice_center
        horizontal = delta - up * delta.dot(up)
        if horizontal.length_squared <= EPSILON:
            continue
        radius = horizontal.dot(radial)
        if radius <= 0.0:
            continue
        direction_score = horizontal.normalized().dot(radial)
        if direction_score <= 0.10:
            continue
        ds = _circular_distance(surface_parameters[index][0], target_s)
        dv = abs(surface_parameters[index][1] - target_v)
        angular_score = max(1.0 - ds / 0.25, 0.05)
        vertical_score = max(1.0 - dv / max(band, EPSILON), 0.05)
        weight = direction_score * direction_score * angular_score * vertical_score
        scored.append((radius, weight))
    if not scored:
        return slice_center + radial * maximum

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:min(8, len(scored))]
    total = sum(weight for _radius, weight in selected)
    weighted = (
        sum(radius * weight for radius, weight in selected) / total
        if total > EPSILON else selected[0][0]
    )
    # Keep a small exterior safety margin when inner layers share the slice.
    fitted = max(weighted, selected[0][0] * 1.02)
    fitted = clamp(fitted, 0.0, maximum)
    radius = _lerp_float(
        maximum,
        fitted,
        clamp(float(ring_fit_strength), 0.0, 1.0),
    )
    return slice_center + radial * radius

def _cyclic_layout_bracket(values, value):
    value = float(value) % 1.0
    count = len(values)
    if count < 2:
        return 0, 0, 0.0
    for following in range(1, count):
        if value < values[following]:
            first = following - 1
            span = max(values[following] - values[first], EPSILON)
            return first, following, clamp(
                (value - values[first]) / span,
                0.0,
                1.0,
            )
    first = count - 1
    following = 0
    span = max(1.0 - values[first] + values[0], EPSILON)
    return first, following, clamp(
        (value - values[first]) / span,
        0.0,
        1.0,
    )


def _ray_segment_radius_in_plane(first, second, ray, up):
    """Return the radius where a 2D ray intersects a polygon edge."""
    first = Vector(first)
    second = Vector(second)
    ray = _safe_normalized(ray, first)
    up = _safe_normalized(up, Vector((0.0, 0.0, 1.0)))
    segment = second - first
    denominator = up.dot(segment.cross(ray))
    if abs(denominator) <= EPSILON:
        return max(first.dot(ray), second.dot(ray), 0.0)
    factor = -up.dot(first.cross(ray)) / denominator
    if factor < -1.0e-5 or factor > 1.0 + 1.0e-5:
        return max(first.dot(ray), second.dot(ray), 0.0)
    intersection = first + segment * clamp(factor, 0.0, 1.0)
    return max(intersection.dot(ray), 0.0)


def _expand_variable_outer_envelope(
    dense_chains,
    centers,
    layout,
    control_points,
    surface_points,
    surface_parameters,
    up,
    bone_inset,
):
    """Expand sparse polygon rings so straight edges cover the outer point cloud."""
    columns = len(dense_chains)
    if columns < 3 or not dense_chains:
        return dense_chains
    dense_count = len(dense_chains[0]) - 1
    if dense_count < 1:
        return dense_chains
    up = _safe_normalized(up, Vector((0.0, 0.0, 1.0)))
    buckets = [[] for _step in range(dense_count + 1)]
    for index, (_point_s, point_v) in enumerate(surface_parameters):
        step = int(clamp(round(point_v * dense_count), 0, dense_count))
        buckets[step].append(index)

    layout_s = layout["physical_s"]
    raw_us = layout["raw_us"]
    inset_scale = 1.0 - clamp(float(bone_inset), 0.0, 0.95)
    for step, indices in enumerate(buckets):
        if not indices:
            continue
        local_v = step / dense_count
        center = Vector(centers[step])
        horizontal_points = []
        for column in range(columns):
            point = Vector(dense_chains[column][step])
            delta = point - center
            horizontal_points.append(delta - up * delta.dot(up))
        scales = [1.0] * columns
        for index in indices:
            delta = Vector(surface_points[index]) - center
            horizontal = delta - up * delta.dot(up)
            point_radius = horizontal.length
            if point_radius <= EPSILON:
                continue
            ray = horizontal / point_radius
            first, following, _factor = _cyclic_layout_bracket(
                layout_s,
                surface_parameters[index][0],
            )
            available = _ray_segment_radius_in_plane(
                horizontal_points[first],
                horizontal_points[following],
                ray,
                up,
            )
            if available <= EPSILON or point_radius <= available:
                continue
            expansion = point_radius / available * 1.01
            scales[first] = max(scales[first], expansion)
            scales[following] = max(scales[following], expansion)

        for column, scale in enumerate(scales):
            if scale <= 1.0 + 1.0e-6:
                continue
            cage_target = _variable_asym_surface_point(
                control_points,
                raw_us[column],
                local_v,
            )
            cage_axis = center + up * (cage_target - center).dot(up)
            cage_delta = cage_target - cage_axis
            cage_delta -= up * cage_delta.dot(up)
            cage_radius = cage_delta.length * inset_scale
            if cage_radius <= EPSILON:
                continue
            point = Vector(dense_chains[column][step])
            point_axis = center + up * (point - center).dot(up)
            radial = cage_delta.normalized()
            current_radius = max((point - point_axis).dot(radial), 0.0)
            expanded_radius = min(current_radius * scale, cage_radius)
            dense_chains[column][step] = point_axis + radial * expanded_radius
    return dense_chains


def build_asymmetric_joint_grid(
    control_points,
    surface_points,
    settings,
    base_l3,
):
    profile = _fit_smooth_hem_profile(
        [{"world": Vector(point)} for point in surface_points],
        settings.target_armature,
        sample_count=max(128, settings.columns * 8),
    )
    hem_profile = _apply_asymmetric_hem_offsets(
        profile,
        control_points,
        base_l3,
    )
    arc_mapping = _build_average_arc_mapping(
        control_points,
        hem_profile,
        resolution=max(128, settings.columns * 8),
    )
    surface_parameters = _initial_asym_parameters(
        surface_points,
        control_points,
        hem_profile,
        arc_mapping,
    )

    dense_count = min(max(settings.rows * 4, 48), 256)
    center_sample_count = max(settings.columns * 2, 32)
    centers = []
    for step in range(dense_count + 1):
        local_v = step / dense_count
        centers.append(_average_points([
            _asym_control_surface_point(
                control_points,
                hem_profile,
                _raw_u_from_arc_s(
                    arc_mapping,
                    sample / center_sample_count,
                ),
                local_v,
            )
            for sample in range(center_sample_count)
        ]))

    chains = []
    for column in range(settings.columns):
        target_s = column / settings.columns
        raw_u = _raw_u_from_arc_s(arc_mapping, target_s)
        window = max(1.75 / settings.columns, 0.04)
        candidates = []
        for _attempt in range(6):
            candidates = [
                index for index, (point_s, _point_v) in enumerate(surface_parameters)
                if _circular_distance(point_s, target_s) <= window
            ]
            if len(candidates) >= 8:
                break
            window = min(window * 1.6, 0.5)
        if not candidates:
            candidates = list(range(len(surface_points)))

        dense_chain = []
        for step in range(dense_count + 1):
            local_v = step / dense_count
            target = _asym_control_surface_point(
                control_points,
                hem_profile,
                raw_u,
                local_v,
            )
            dense_chain.append(_fit_asym_dense_point(
                target,
                centers[step],
                target_s,
                local_v,
                candidates,
                surface_points,
                surface_parameters,
                settings.bone_inset,
                settings.ring_fit_strength,
                up=hem_profile["up"],
            ))
        chains.append(_equal_chord_chain(dense_chain, settings.rows))

    joint_grid = [
        [chains[column][row] for column in range(settings.columns)]
        for row in range(settings.rows + 1)
    ]
    chain_lengths = [
        sum(
            (chains[column][row + 1] - chains[column][row]).length
            for row in range(settings.rows)
        )
        for column in range(settings.columns)
    ]
    return joint_grid, chain_lengths, hem_profile, arc_mapping


def _joint_surface_point(joint_grid, arc_s, local_v):
    row_count = len(joint_grid)
    columns = len(joint_grid[0]) if row_count else 0
    if row_count < 2 or columns < 2:
        return Vector((0.0, 0.0, 0.0))

    row_value = clamp(float(local_v), 0.0, 1.0) * (row_count - 1)
    row0 = min(int(math.floor(row_value)), row_count - 1)
    row1 = min(row0 + 1, row_count - 1)
    row_factor = row_value - math.floor(row_value)

    column_value = (float(arc_s) % 1.0) * columns
    column0 = int(math.floor(column_value)) % columns
    column1 = (column0 + 1) % columns
    column_factor = column_value - math.floor(column_value)

    first = Vector(joint_grid[row0][column0]).lerp(
        Vector(joint_grid[row0][column1]),
        column_factor,
    )
    second = Vector(joint_grid[row1][column0]).lerp(
        Vector(joint_grid[row1][column1]),
        column_factor,
    )
    return first.lerp(second, row_factor)


def _build_asymmetric_parameter_projector(joint_grid):
    bone_count = len(joint_grid) - 1
    columns = len(joint_grid[0]) if joint_grid else 0
    if bone_count < 1 or columns < 4:
        raise ValueError("不对称参数曲面关节点无效")

    s_resolution = min(max(columns * 4, 64), 512)
    v_resolution = min(max(bone_count * 4, 32), 512)
    vertices = [
        _joint_surface_point(
            joint_grid,
            column / s_resolution,
            row / v_resolution,
        )
        for row in range(v_resolution + 1)
        for column in range(s_resolution)
    ]
    triangles = []
    triangle_uv = []
    for row in range(v_resolution):
        v0 = row / v_resolution
        v1 = (row + 1) / v_resolution
        for column in range(s_resolution):
            following = (column + 1) % s_resolution
            s0 = column / s_resolution
            s1 = (column + 1) / s_resolution
            a = row * s_resolution + column
            b = row * s_resolution + following
            c = (row + 1) * s_resolution + column
            d = (row + 1) * s_resolution + following
            triangles.append((a, b, d))
            triangle_uv.append((
                Vector((s0, v0)),
                Vector((s1, v0)),
                Vector((s1, v1)),
            ))
            triangles.append((a, d, c))
            triangle_uv.append((
                Vector((s0, v0)),
                Vector((s1, v1)),
                Vector((s0, v1)),
            ))
    tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    return tree, vertices, triangles, triangle_uv


def _barycentric_coordinates(point, first, second, third):
    first = Vector(first)
    second = Vector(second)
    third = Vector(third)
    point = Vector(point)
    edge0 = second - first
    edge1 = third - first
    delta = point - first
    d00 = edge0.dot(edge0)
    d01 = edge0.dot(edge1)
    d11 = edge1.dot(edge1)
    d20 = delta.dot(edge0)
    d21 = delta.dot(edge1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= EPSILON:
        return 1.0, 0.0, 0.0
    second_weight = (d11 * d20 - d01 * d21) / denominator
    third_weight = (d00 * d21 - d01 * d20) / denominator
    first_weight = 1.0 - second_weight - third_weight
    return first_weight, second_weight, third_weight


def _project_asymmetric_parameters(point, projector):
    tree, vertices, triangles, triangle_uv = projector
    nearest = tree.find_nearest(Vector(point))
    if nearest is None or nearest[2] is None:
        return 0.0, 0.0
    location, _normal, triangle_index, _distance = nearest
    triangle = triangles[triangle_index]
    weights = _barycentric_coordinates(
        location,
        vertices[triangle[0]],
        vertices[triangle[1]],
        vertices[triangle[2]],
    )
    uv_values = triangle_uv[triangle_index]
    uv = (
        uv_values[0] * weights[0]
        + uv_values[1] * weights[1]
        + uv_values[2] * weights[2]
    )
    return uv.x % 1.0, clamp(uv.y, 0.0, 1.0)

def _interpolate_cyclic_values(values, arc_s):
    if not values:
        return 0.0
    value = (float(arc_s) % 1.0) * len(values)
    first = int(math.floor(value)) % len(values)
    second = (first + 1) % len(values)
    return _lerp_float(
        values[first],
        values[second],
        value - math.floor(value),
    )


def _median_value(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _asymmetric_parent_weight(arc_s, local_v, chain_lengths):
    reference_length = _median_value(chain_lengths)
    blend_distance = reference_length * 0.15
    if blend_distance <= EPSILON:
        return 0.0
    local_length = _interpolate_cyclic_values(chain_lengths, arc_s)
    distance = clamp(float(local_v), 0.0, 1.0) * local_length
    if distance >= blend_distance:
        return 0.0
    return 1.0 - smoothstep(distance / blend_distance)


def compute_asymmetric_vertex_weights(
    arc_s,
    local_v,
    rows,
    columns,
    prefix,
    settings,
    parent_weight,
):
    # Influence nodes are centered in each equal-length chain segment.
    row_value = clamp(float(local_v) * rows - 0.5, 0.0, rows - 1.0)
    row0 = min(int(math.floor(row_value)), rows - 1)
    row1 = min(row0 + 1, rows - 1)
    row_factor = row_value - math.floor(row_value)
    row_weight0, row_weight1 = _vertical_weights(
        row_factor,
        settings.weight_curve,
    )

    column_value = (float(arc_s) % 1.0) * columns
    base_column = int(math.floor(column_value))
    column0 = base_column % columns
    column1 = (column0 + 1) % columns
    column_factor = column_value - base_column
    column_weight1 = smoothstep(column_factor)
    column_weight0 = 1.0 - column_weight1

    skirt_scale = 1.0 - clamp(float(parent_weight), 0.0, 1.0)
    weights = {}
    for row, column, value in (
        (row0, column0, row_weight0 * column_weight0),
        (row0, column1, row_weight0 * column_weight1),
        (row1, column0, row_weight1 * column_weight0),
        (row1, column1, row_weight1 * column_weight1),
    ):
        name = _bone_name(prefix, column, row)
        weights[name] = weights.get(name, 0.0) + value * skirt_scale

    weights = {
        name: value for name, value in weights.items()
        if value >= MIN_WEIGHT
    }
    parent_is_active = parent_weight >= MIN_WEIGHT
    skirt_slots = settings.max_influences - (1 if parent_is_active else 0)
    weights = dict(sorted(
        weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max(skirt_slots, 0)])
    if parent_is_active:
        weights[PARENT_WEIGHT_KEY] = parent_weight

    if settings.normalize_weights:
        total = sum(weights.values())
        if total > EPSILON:
            weights = {
                name: value / total
                for name, value in weights.items()
            }
    return weights


def create_bones_from_joint_grid(
    armature,
    joint_grid,
    parent_bone_name,
    prefix,
    sample_normals=None,
    bone_collection_name="",
):
    if armature.type != 'ARMATURE' or armature.mode != 'EDIT':
        raise ValueError("创建骨骼时 Armature 必须处于编辑模式")
    bone_count = len(joint_grid) - 1
    columns = len(joint_grid[0]) if joint_grid else 0
    if bone_count < 1 or columns < 4:
        raise ValueError("不对称裙摆关节点网格无效")
    if any(len(row) != columns for row in joint_grid):
        raise ValueError("不对称裙摆关节点列数不一致")

    edit_bones = armature.data.edit_bones
    parent_bone = edit_bones.get(parent_bone_name)
    if parent_bone is None:
        raise ValueError(f"找不到父骨骼：{parent_bone_name}")
    bone_collection = _get_or_create_bone_collection(
        armature.data,
        bone_collection_name,
    )

    requested_names = {
        _bone_name(prefix, column, row)
        for column in range(columns)
        for row in range(bone_count)
    }
    temporary_names = {
        _temporary_bone_name(prefix, column, row)
        for column in range(columns)
        for row in range(bone_count)
    }
    conflicts = sorted(
        name for name in requested_names | temporary_names
        if edit_bones.get(name)
    )
    if conflicts:
        raise ValueError(
            "已存在同名生成骨骼；请启用“清除已有生成骨骼”："
            + ", ".join(conflicts[:4])
        )

    inverse = armature.matrix_world.inverted()
    normal_to_local = armature.matrix_world.to_3x3().transposed()
    local_joints = [
        [inverse @ Vector(point) for point in row]
        for row in joint_grid
    ]
    local_normals = None
    if sample_normals:
        local_normals = [
            [
                _safe_normalized(
                    normal_to_local @ Vector(sample_normals[row][column]),
                    Vector((1.0, 0.0, 0.0)),
                )
                for column in range(columns)
            ]
            for row in range(min(len(sample_normals), bone_count + 1))
        ]

    created = []
    renamed = []
    for column in range(columns):
        previous = None
        for row in range(bone_count):
            final_name = _bone_name(prefix, column, row)
            bone = edit_bones.new(
                _temporary_bone_name(prefix, column, row)
            )
            bone.head = local_joints[row][column]
            bone.tail = local_joints[row + 1][column]
            if (bone.tail - bone.head).length_squared <= EPSILON:
                raise ValueError(
                    f"不对称裙摆骨骼长度为零：C{column:02d} R{row:02d}"
                )
            bone.parent = parent_bone if row == 0 else previous
            bone.use_connect = row > 0
            bone.use_deform = True
            if bone_collection is not None:
                bone_collection.assign(bone)
            target_normal = (
                local_normals[row][column]
                if local_normals and row < len(local_normals)
                else Vector((1.0, 0.0, 0.0))
            )
            _align_bone_roll(bone, target_normal)
            created.append(final_name)
            renamed.append((bone, final_name))
            previous = bone

    for bone, final_name in renamed:
        bone.name = final_name
        if bone.name != final_name:
            raise ValueError(f"Could not assign generated bone name: {final_name}")
    return created


def create_asymmetric_proxy_mesh(
    joint_grid,
    prefix="AutoSkirt_",
    mesh_name="Mesh",
):
    rows = len(joint_grid)
    columns = len(joint_grid[0]) if rows else 0
    if rows < 2 or columns < 4:
        return None
    collection = get_or_create_collection(
        _proxy_collection_name(prefix, mesh_name)
    )
    vertices = [
        tuple(joint_grid[row][column])
        for row in range(rows)
        for column in range(columns)
    ]
    faces = []
    for row in range(rows - 1):
        for column in range(columns):
            following = (column + 1) % columns
            faces.append((
                row * columns + column,
                row * columns + following,
                (row + 1) * columns + following,
                (row + 1) * columns + column,
            ))

    mesh_data = bpy.data.meshes.new(f"{prefix}PROXY_{mesh_name}_Mesh")
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()
    obj = bpy.data.objects.new(f"{prefix}PROXY_{mesh_name}", mesh_data)
    collection.objects.link(obj)
    obj.display_type = 'WIRE'
    obj.show_wire = True
    obj.show_in_front = True
    obj.hide_render = True
    obj[PROP_TARGET] = mesh_name
    obj[PROP_KIND] = PROP_PROXY_KIND
    return obj


def apply_asymmetric_proxy_weights(
    proxy_obj,
    armature_obj,
    joint_grid,
    chain_lengths,
    rows,
    columns,
    prefix,
    parent_bone_name,
    settings,
):
    if proxy_obj is None or proxy_obj.type != 'MESH':
        return 0
    indices = list(range(len(proxy_obj.data.vertices)))
    weights = {}
    for joint_row in range(rows + 1):
        local_v = joint_row / rows
        for column in range(columns):
            index = joint_row * columns + column
            arc_s = column / columns
            parent_weight = _asymmetric_parent_weight(
                arc_s,
                local_v,
                chain_lengths,
            )
            weights[index] = compute_asymmetric_vertex_weights(
                arc_s,
                local_v,
                rows,
                columns,
                prefix,
                settings,
                parent_weight,
            )
    assignments = apply_weights(
        proxy_obj,
        indices,
        weights,
        prefix,
        parent_bone_name,
        True,
        settings.top_blend_to_parent,
    )
    _ensure_armature_modifier(proxy_obj, armature_obj)
    return assignments


def _variable_asym_ring_point_linear(control_points, level, raw_u):
    count = len(control_points)
    if count < 2:
        raise ValueError("不对称裙摆至少需要 4 条控制链")
    value = (float(raw_u) % 1.0) * count
    first = int(math.floor(value)) % count
    second = (first + 1) % count
    factor = smoothstep(value - math.floor(value))
    return Vector(control_points[first][level]).lerp(
        Vector(control_points[second][level]),
        factor,
    )


_VARIABLE_ASYM_FRAME_CACHE = {
    "source": None,
    "frame": None,
}


def _variable_asym_control_frame(control_points):
    """Return stable centers and a height axis for the closed control cage."""
    if _VARIABLE_ASYM_FRAME_CACHE["source"] is control_points:
        return _VARIABLE_ASYM_FRAME_CACHE["frame"]

    centers = [
        _average_points([Vector(chain[level]) for chain in control_points])
        for level in range(CONTROL_LEVELS)
    ]
    up = centers[0] - centers[-1]
    if up.length_squared <= EPSILON:
        up = _average_points([
            Vector(chain[0]) - Vector(chain[-1])
            for chain in control_points
        ])
    up = _safe_normalized(up, Vector((0.0, 0.0, 1.0)))
    frame = (centers, up)
    _VARIABLE_ASYM_FRAME_CACHE["source"] = control_points
    _VARIABLE_ASYM_FRAME_CACHE["frame"] = frame
    return frame


def _variable_asym_ring_point(control_points, level, raw_u):
    count = len(control_points)
    value = (float(raw_u) % 1.0) * count
    first = int(math.floor(value)) % count
    second = (first + 1) % count
    factor = smoothstep(value - math.floor(value))
    centers, up = _variable_asym_control_frame(control_points)
    center = centers[level]
    first_delta = Vector(control_points[first][level]) - center
    second_delta = Vector(control_points[second][level]) - center
    first_height = first_delta.dot(up)
    second_height = second_delta.dot(up)
    first_radial = first_delta - up * first_height
    second_radial = second_delta - up * second_height
    first_radius = first_radial.length
    second_radius = second_radial.length
    if first_radius <= EPSILON or second_radius <= EPSILON:
        return _variable_asym_ring_point_linear(control_points, level, raw_u)

    first_direction = first_radial / first_radius
    second_direction = second_radial / second_radius
    angle = math.atan2(
        up.dot(first_direction.cross(second_direction)),
        clamp(first_direction.dot(second_direction), -1.0, 1.0),
    )
    if angle <= EPSILON:
        angle += math.tau
    angle *= factor
    direction = _safe_normalized(
        first_direction * math.cos(angle)
        + up.cross(first_direction) * math.sin(angle),
        first_direction,
    )
    radius = _lerp_float(first_radius, second_radius, factor)
    height = _lerp_float(first_height, second_height, factor)
    return center + direction * radius + up * height

def _variable_asym_surface_point(control_points, raw_u, local_v):
    rail = [
        _variable_asym_ring_point(control_points, level, raw_u)
        for level in range(CONTROL_LEVELS)
    ]
    return interpolate_vertical_control(rail, local_v)


def _variable_sector_arc_data(control_points, samples_per_sector=64):
    chain_count = len(control_points)
    sectors = []
    for sector in range(chain_count):
        cumulative = [0.0]
        previous = [
            _variable_asym_ring_point(
                control_points,
                level,
                sector / chain_count,
            )
            for level in range(CONTROL_LEVELS)
        ]
        for step in range(1, samples_per_sector + 1):
            raw_u = (sector + step / samples_per_sector) / chain_count
            current = [
                _variable_asym_ring_point(control_points, level, raw_u)
                for level in range(CONTROL_LEVELS)
            ]
            length = sum(
                (current[level] - previous[level]).length
                for level in range(CONTROL_LEVELS)
            ) / CONTROL_LEVELS
            cumulative.append(cumulative[-1] + max(length, EPSILON))
            previous = current
        sectors.append({
            "cumulative": cumulative,
            "length": cumulative[-1],
            "samples": samples_per_sector,
        })
    return sectors


def _allocate_sector_intervals(sector_lengths, total_columns):
    sector_count = len(sector_lengths)
    if total_columns < sector_count:
        raise ValueError("骨骼链数不能少于控制点链数")
    intervals = [1] * sector_count
    remaining = total_columns - sector_count
    total_length = sum(sector_lengths)
    if remaining <= 0:
        return intervals
    if total_length <= EPSILON:
        for index in range(remaining):
            intervals[index % sector_count] += 1
        return intervals
    exact = [remaining * length / total_length for length in sector_lengths]
    floors = [int(math.floor(value)) for value in exact]
    for index, value in enumerate(floors):
        intervals[index] += value
    leftover = remaining - sum(floors)
    order = sorted(
        range(sector_count),
        key=lambda index: (exact[index] - floors[index], sector_lengths[index]),
        reverse=True,
    )
    for index in order[:leftover]:
        intervals[index] += 1
    return intervals


def _sector_fraction_from_arc(sector_data, fraction):
    fraction = clamp(float(fraction), 0.0, 1.0)
    cumulative = sector_data["cumulative"]
    target = fraction * sector_data["length"]
    low = 0
    high = sector_data["samples"]
    while low + 1 < high:
        middle = (low + high) // 2
        if cumulative[middle] <= target:
            low = middle
        else:
            high = middle
    span = cumulative[low + 1] - cumulative[low]
    factor = 0.0 if span <= EPSILON else (
        (target - cumulative[low]) / span
    )
    return (low + factor) / sector_data["samples"]


def _build_variable_chain_layout(
    control_points,
    total_columns,
    anchor_bone_counts,
    sector_data=None,
    arc_mapping=None,
):
    chain_count = len(control_points)
    if arc_mapping is not None:
        # Sample the complete fitted envelope at globally uniform arc-length
        # intervals. Control anchors still drive the interpolated bone counts,
        # but no sector is allowed to create a visibly short or long edge.
        physical_s = [column / total_columns for column in range(total_columns)]
        raw_us = [
            _raw_u_from_arc_s(arc_mapping, value)
            for value in physical_s
        ]
        bone_counts = []
        sector_intervals = [0] * chain_count
        for raw_u in raw_us:
            value = (raw_u % 1.0) * chain_count
            first = int(math.floor(value)) % chain_count
            following = (first + 1) % chain_count
            factor = value - math.floor(value)
            sector_intervals[first] += 1
            bone_counts.append(int(clamp(round(_lerp_float(
                anchor_bone_counts[first],
                anchor_bone_counts[following],
                factor,
            )), 1, 64)))

        # Keep every user count represented by the nearest unique generated
        # chain without moving that chain away from its uniform position.
        anchor_columns = []
        used_columns = set()
        for anchor in range(chain_count):
            anchor_s = _arc_s_from_raw_u(
                arc_mapping,
                anchor / chain_count,
            )
            candidates = sorted(
                range(total_columns),
                key=lambda column: _circular_distance(
                    column / total_columns,
                    anchor_s,
                ),
            )
            column = next(
                candidate for candidate in candidates
                if candidate not in used_columns
            )
            used_columns.add(column)
            anchor_columns.append(column)
            bone_counts[column] = int(clamp(
                anchor_bone_counts[anchor],
                1,
                64,
            ))
        return {
            "raw_us": raw_us,
            "physical_s": physical_s,
            "bone_counts": bone_counts,
            "sector_intervals": sector_intervals,
            "anchor_columns": anchor_columns,
        }

    sectors = (
        sector_data
        if sector_data is not None and len(sector_data) == chain_count
        else _variable_sector_arc_data(control_points)
    )
    lengths = [sector["length"] for sector in sectors]
    intervals = _allocate_sector_intervals(lengths, total_columns)
    total_length = max(sum(lengths), EPSILON)
    accumulated_length = 0.0
    raw_us = []
    physical_s = []
    bone_counts = []
    anchor_columns = []
    for sector, interval_count in enumerate(intervals):
        anchor_columns.append(len(raw_us))
        following = (sector + 1) % chain_count
        for step in range(interval_count):
            arc_fraction = step / interval_count
            local_fraction = _sector_fraction_from_arc(
                sectors[sector],
                arc_fraction,
            )
            raw_us.append((sector + local_fraction) / chain_count)
            physical_s.append((
                accumulated_length + arc_fraction * lengths[sector]
            ) / total_length)
            bone_counts.append(int(clamp(round(_lerp_float(
                anchor_bone_counts[sector],
                anchor_bone_counts[following],
                arc_fraction,
            )), 1, 64)))
        accumulated_length += lengths[sector]
    return {
        "raw_us": raw_us,
        "physical_s": physical_s,
        "bone_counts": bone_counts,
        "sector_intervals": intervals,
        "anchor_columns": anchor_columns,
    }


def _build_variable_arc_mapping(control_points, resolution=256):
    resolution = max(int(resolution), 64)
    cumulative = [0.0]
    previous = [
        _variable_asym_ring_point(control_points, level, 0.0)
        for level in range(CONTROL_LEVELS)
    ]
    for sample in range(1, resolution + 1):
        raw_u = sample / resolution
        current = [
            _variable_asym_ring_point(control_points, level, raw_u)
            for level in range(CONTROL_LEVELS)
        ]
        length = sum(
            (current[level] - previous[level]).length
            for level in range(CONTROL_LEVELS)
        ) / CONTROL_LEVELS
        cumulative.append(cumulative[-1] + max(length, EPSILON))
        previous = current
    return {
        "resolution": resolution,
        "cumulative": cumulative,
        "total": cumulative[-1],
    }


def _initial_variable_asym_parameters(
    surface_points,
    control_points,
    profile,
    arc_mapping,
):
    center = profile["center"]
    right = profile["right"]
    back = profile["back"]
    rail_cache = {}
    parameters = []
    for point in surface_points:
        raw_u = _raw_skirt_u(point, center, right, back)
        cache_index = int(round(
            raw_u * arc_mapping["resolution"]
        )) % arc_mapping["resolution"]
        if cache_index not in rail_cache:
            cached_u = cache_index / arc_mapping["resolution"]
            rail_cache[cache_index] = [
                _variable_asym_surface_point(
                    control_points,
                    cached_u,
                    step / 48.0,
                )
                for step in range(49)
            ]
        local_v = _closest_polyline_parameter(
            point,
            rail_cache[cache_index],
        )
        parameters.append((
            _arc_s_from_raw_u(arc_mapping, raw_u),
            local_v,
        ))
    return parameters


def _interpolate_vector_samples(values, factor):
    if not values:
        return Vector((1.0, 0.0, 0.0))
    if len(values) == 1:
        return Vector(values[0])
    value = clamp(float(factor), 0.0, 1.0) * (len(values) - 1)
    first = min(int(math.floor(value)), len(values) - 1)
    second = min(first + 1, len(values) - 1)
    return _safe_normalized(
        Vector(values[first]).lerp(
            Vector(values[second]),
            value - math.floor(value),
        ),
        values[first],
    )


def build_variable_asymmetric_data(
    control_points,
    surface_points,
    settings,
    anchor_bone_counts,
):
    if not 4 <= len(control_points) <= 24:
        raise ValueError("不对称控制点链数必须在 4 到 24 之间")
    if int(settings.columns) < len(control_points):
        raise ValueError("骨骼链数不能少于控制点链数")
    profile = _fit_smooth_hem_profile(
        [{"world": Vector(point)} for point in surface_points],
        settings.target_armature,
        sample_count=max(128, settings.columns * 8),
    )
    arc_mapping = _build_point_cloud_average_arc_mapping(
        surface_points,
        control_points,
        profile,
        resolution=max(128, settings.columns * 8),
    )
    max_bones = max(int(value) for value in anchor_bone_counts)
    dense_count = min(max(max_bones * 10, 64), 640)
    center_sample_count = max(settings.columns * 2, 32)
    centers = []
    for step in range(dense_count + 1):
        local_v = step / dense_count
        centers.append(_average_points([
            _variable_asym_surface_point(
                control_points,
                sample / center_sample_count,
                local_v,
            )
            for sample in range(center_sample_count)
        ]))

    layout = None
    surface_parameters = None
    dense_chains = None
    # Two correction passes use the actually fitted surface to remove residual
    # spacing differences introduced by waist/hem shape changes.
    for spacing_pass in range(3):
        layout = _build_variable_chain_layout(
            control_points,
            int(settings.columns),
            anchor_bone_counts,
            arc_mapping=arc_mapping,
        )
        surface_parameters = _initial_variable_asym_parameters(
            surface_points,
            control_points,
            profile,
            arc_mapping,
        )
        dense_chains = []
        for column, (raw_u, target_s, _bone_count) in enumerate(zip(
            layout["raw_us"],
            layout["physical_s"],
            layout["bone_counts"],
        )):
            window = max(1.75 / settings.columns, 0.04)
            candidates = []
            for _attempt in range(6):
                candidates = [
                    index
                    for index, (point_s, _point_v) in enumerate(surface_parameters)
                    if _circular_distance(point_s, target_s) <= window
                ]
                if len(candidates) >= 8:
                    break
                window = min(window * 1.6, 0.5)
            if not candidates:
                candidates = list(range(len(surface_points)))

            dense_chain = []
            for step in range(dense_count + 1):
                local_v = step / dense_count
                target = _variable_asym_surface_point(
                    control_points,
                    raw_u,
                    local_v,
                )
                dense_chain.append(_fit_asym_dense_point(
                    target,
                    centers[step],
                    target_s,
                    local_v,
                    candidates,
                    surface_points,
                    surface_parameters,
                    settings.bone_inset,
                    settings.ring_fit_strength,
                    up=profile["up"],
                ))
            dense_chains.append(dense_chain)

        dense_chains = _expand_variable_outer_envelope(
            dense_chains,
            centers,
            layout,
            control_points,
            surface_points,
            surface_parameters,
            profile["up"],
            settings.bone_inset,
        )
        if spacing_pass < 2:
            corrected_mapping = _build_fitted_chain_arc_mapping(
                layout["raw_us"],
                dense_chains,
                profile["up"],
                resolution=max(128, settings.columns * 8),
            )
            if corrected_mapping is not None:
                arc_mapping = corrected_mapping

    bone_chains = []
    preview_chains = []
    chain_lengths = []
    for column, (dense_chain, bone_count) in enumerate(zip(
        dense_chains,
        layout["bone_counts"],
    )):
        bone_chain = _equal_chord_chain(dense_chain, bone_count)
        preview_chain = _equal_chord_chain(dense_chain, max_bones)
        if settings.smooth_bone_chains:
            bone_chain = _smooth_open_bone_chain(bone_chain)
            preview_chain = _smooth_open_bone_chain(preview_chain)
        bone_chains.append(bone_chain)
        preview_chains.append(preview_chain)
        chain_lengths.append(sum(
            (bone_chain[index + 1] - bone_chain[index]).length
            for index in range(len(bone_chain) - 1)
        ))

    preview_grid = [
        [preview_chains[column][row] for column in range(settings.columns)]
        for row in range(max_bones + 1)
    ]
    proxy_normals = _proxy_grid_vertex_normals(preview_grid, True)
    bone_z_axes = []
    for column, bone_count in enumerate(layout["bone_counts"]):
        column_normals = [row[column] for row in proxy_normals]
        bone_z_axes.append([
            -_interpolate_vector_samples(
                column_normals,
                joint / bone_count,
            )
            for joint in range(bone_count + 1)
        ])
    return {
        "bone_chains": bone_chains,
        "preview_grid": preview_grid,
        "bone_counts": layout["bone_counts"],
        "anchor_columns": layout["anchor_columns"],
        "sector_intervals": layout["sector_intervals"],
        "chain_lengths": chain_lengths,
        "bone_z_axes": bone_z_axes,
        "projector": _build_asymmetric_parameter_projector(preview_grid),
        "raw_us": layout["raw_us"],
    }


def create_variable_asymmetric_bones(
    armature,
    bone_chains,
    bone_z_axes,
    parent_bone_name,
    prefix,
    bone_collection_name="",
):
    if armature.type != 'ARMATURE' or armature.mode != 'EDIT':
        raise ValueError("创建骨骼时 Armature 必须处于编辑模式")
    if len(bone_chains) < 4:
        raise ValueError("不对称裙摆骨骼链数量无效")
    edit_bones = armature.data.edit_bones
    parent_bone = edit_bones.get(parent_bone_name)
    if parent_bone is None:
        raise ValueError(f"找不到父骨骼：{parent_bone_name}")
    bone_collection = _get_or_create_bone_collection(
        armature.data,
        bone_collection_name,
    )
    requested = {
        _plain_bone_name(prefix, column, row)
        for column, chain in enumerate(bone_chains)
        for row in range(len(chain) - 1)
    }
    conflicts = sorted(name for name in requested if edit_bones.get(name))
    if conflicts:
        raise ValueError(
            "已存在同名生成骨骼；请启用“清除已有生成骨骼”："
            + ", ".join(conflicts[:4])
        )
    inverse = armature.matrix_world.inverted()
    normal_to_local = armature.matrix_world.to_3x3().transposed()
    created = []
    for column, chain in enumerate(bone_chains):
        previous = None
        for row in range(len(chain) - 1):
            name = _plain_bone_name(prefix, column, row)
            bone = edit_bones.new(name)
            bone.head = inverse @ Vector(chain[row])
            bone.tail = inverse @ Vector(chain[row + 1])
            if (bone.tail - bone.head).length_squared <= EPSILON:
                raise ValueError(f"不对称裙摆骨骼长度为零：{name}")
            bone.parent = parent_bone if row == 0 else previous
            bone.use_connect = row > 0
            bone.use_deform = True
            if bone_collection is not None:
                bone_collection.assign(bone)
            normal = (
                bone_z_axes[column][row]
                if column < len(bone_z_axes)
                and row < len(bone_z_axes[column])
                else Vector((1.0, 0.0, 0.0))
            )
            _align_bone_roll(
                bone,
                _safe_normalized(
                    normal_to_local @ Vector(normal),
                    Vector((1.0, 0.0, 0.0)),
                ),
            )
            previous = bone
            created.append(name)
    return created


def compute_variable_asymmetric_vertex_weights(
    arc_s,
    local_v,
    bone_counts,
    prefix,
    settings,
    parent_weight,
):
    columns = len(bone_counts)
    column_value = (float(arc_s) % 1.0) * columns
    base_column = int(math.floor(column_value))
    column0 = base_column % columns
    column1 = (column0 + 1) % columns
    column_factor = column_value - base_column
    column_weight1 = smoothstep(column_factor)
    column_weight0 = 1.0 - column_weight1
    skirt_scale = 1.0 - clamp(float(parent_weight), 0.0, 1.0)
    weights = {}
    for column, column_weight in (
        (column0, column_weight0),
        (column1, column_weight1),
    ):
        count = int(bone_counts[column])
        row_value = clamp(
            float(local_v) * count - 0.5,
            0.0,
            count - 1.0,
        )
        row0 = min(int(math.floor(row_value)), count - 1)
        row1 = min(row0 + 1, count - 1)
        row_factor = row_value - math.floor(row_value)
        row_weight0, row_weight1 = _vertical_weights(
            row_factor,
            settings.weight_curve,
        )
        for row, row_weight in (
            (row0, row_weight0),
            (row1, row_weight1),
        ):
            name = _plain_bone_name(prefix, column, row)
            weights[name] = weights.get(name, 0.0) + (
                row_weight * column_weight * skirt_scale
            )
    weights = {
        name: value for name, value in weights.items()
        if value >= MIN_WEIGHT
    }
    parent_is_active = parent_weight >= MIN_WEIGHT
    skirt_slots = settings.max_influences - (1 if parent_is_active else 0)
    weights = dict(sorted(
        weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max(skirt_slots, 0)])
    if parent_is_active:
        weights[PARENT_WEIGHT_KEY] = parent_weight
    if settings.normalize_weights:
        total = sum(weights.values())
        if total > EPSILON:
            weights = {name: value / total for name, value in weights.items()}
    return weights


def apply_variable_asymmetric_proxy_weights(
    proxy_obj,
    armature_obj,
    preview_grid,
    bone_counts,
    chain_lengths,
    prefix,
    parent_bone_name,
    settings,
):
    if proxy_obj is None or proxy_obj.type != 'MESH':
        return 0
    rows = len(preview_grid)
    columns = len(bone_counts)
    indices = list(range(len(proxy_obj.data.vertices)))
    weights = {}
    for row in range(rows):
        local_v = row / max(rows - 1, 1)
        for column in range(columns):
            index = row * columns + column
            arc_s = column / columns
            parent_weight = _asymmetric_parent_weight(
                arc_s,
                local_v,
                chain_lengths,
            )
            weights[index] = compute_variable_asymmetric_vertex_weights(
                arc_s,
                local_v,
                bone_counts,
                prefix,
                settings,
                parent_weight,
            )
    assignments = apply_weights(
        proxy_obj,
        indices,
        weights,
        prefix,
        parent_bone_name,
        True,
        settings.top_blend_to_parent,
    )
    _ensure_armature_modifier(proxy_obj, armature_obj)
    return assignments

def _prepare_asymmetric_build_data(
    primary_mesh,
    armature,
    control_points,
    surface_points,
    settings,
):
    del armature
    if not _controls_are_asymmetric(control_points):
        raise ValueError("旧版不对称控制点不兼容，请重新生成控制点")
    if len(control_points) != int(settings.asym_control_chain_count):
        raise ValueError("不对称控制点链数已改变，请重新应用控制链数量")
    anchor_counts = _get_asym_anchor_bone_counts(
        settings,
        len(control_points),
    )
    data = build_variable_asymmetric_data(
        control_points,
        surface_points,
        settings,
        anchor_counts,
    )
    data["anchor_bone_counts"] = anchor_counts
    primary_mesh[PROP_ASYM_CONTROL_CHAINS] = len(control_points)
    primary_mesh[PROP_ASYM_BONE_COUNTS] = json.dumps(
        anchor_counts,
        separators=(",", ":"),
    )
    return data

# ---------------------------------------------------------------------------
# Build validation and orchestration
# ---------------------------------------------------------------------------


def _records_from_indices(mesh_obj, indices):
    """Read current point-cloud positions and world normals without mode changes."""
    records = []
    matrix_world = mesh_obj.matrix_world
    normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
    if mesh_obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        for index in indices:
            if 0 <= index < len(bm.verts):
                vertex = bm.verts[index]
                local = vertex.co.copy()
                records.append({
                    "index": index,
                    "local": local,
                    "world": matrix_world @ local,
                    "normal": _safe_normalized(
                        normal_matrix @ vertex.normal,
                        Vector((0.0, 0.0, 1.0)),
                    ),
                })
    else:
        for index in indices:
            if 0 <= index < len(mesh_obj.data.vertices):
                vertex = mesh_obj.data.vertices[index]
                local = vertex.co.copy()
                records.append({
                    "index": index,
                    "local": local,
                    "world": matrix_world @ local,
                    "normal": _safe_normalized(
                        normal_matrix @ vertex.normal,
                        Vector((0.0, 0.0, 1.0)),
                    ),
                })
    return records


def _validate_build(context, settings, require_selection):
    _validate_prefix(settings.prefix)
    mesh_objects = _resolve_target_meshes(context, settings)
    armature = settings.target_armature
    if not mesh_objects:
        raise ValueError("Target Mesh does not exist.")
    if armature is None or armature.type != 'ARMATURE':
        raise ValueError("Target Armature does not exist.")
    for mesh_obj in mesh_objects:
        if not _scene_has_object(context, mesh_obj):
            raise ValueError(f"Target Mesh '{mesh_obj.name}' is not in this View Layer.")
    if not _scene_has_object(context, armature):
        raise ValueError("Target Armature is not in this View Layer.")
    if settings.skirt_shape_mode == 'STANDARD' and settings.rows < 2:
        raise ValueError("纵向层数必须至少为 2")
    if settings.columns < 4:
        raise ValueError("骨骼链数必须至少为 4")
    if (
        settings.skirt_shape_mode == 'ASYMMETRIC'
        and settings.columns < settings.asym_control_chain_count
    ):
        raise ValueError("骨骼链数不能少于控制点链数")
    if not settings.parent_bone:
        raise ValueError("A Parent Bone is required.")
    if settings.parent_bone.startswith(settings.prefix):
        raise ValueError("Parent Bone cannot use the generated prefix.")
    if armature.data.bones.get(settings.parent_bone) is None:
        raise ValueError("Parent Bone does not exist in the Target Armature.")

    primary_mesh = mesh_objects[0]
    stored_mode = str(primary_mesh.get(PROP_SHAPE_MODE, 'STANDARD'))
    if stored_mode != settings.skirt_shape_mode:
        raise ValueError("裙摆类型已改变，请重新生成控制点")
    collection_name = str(
        primary_mesh.get(
            PROP_COLLECTION,
            _control_collection_name(settings.prefix, primary_mesh.name),
        )
    )
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        raise ValueError("Control Collection does not exist; generate controls first.")
    controls = find_control_points(collection, settings.prefix, require_all=True)
    if settings.skirt_shape_mode == 'ASYMMETRIC':
        if not _controls_are_asymmetric(controls):
            raise ValueError("旧版不对称控制点不兼容，请重新生成控制点")
        if len(controls) != int(settings.asym_control_chain_count):
            raise ValueError("控制点链数与设置不一致")

    selection_items = []
    if require_selection:
        for mesh_obj in mesh_objects:
            indices = _current_or_stored_indices(mesh_obj)
            if not indices:
                raise ValueError(
                    f"Target Mesh '{mesh_obj.name}' has no recorded selected vertices."
                )
            selection_items.append((mesh_obj, indices))
    return primary_mesh, mesh_objects, armature, controls, selection_items


def _required_variable_bones_exist(armature, prefix, bone_counts):
    missing = [
        _plain_bone_name(prefix, column, row)
        for column, count in enumerate(bone_counts)
        for row in range(int(count))
        if armature.data.bones.get(
            _plain_bone_name(prefix, column, row)
        ) is None
    ]
    if missing:
        raise ValueError(
            "缺少用于赋权的不对称裙骨，请先生成骨骼："
            + ", ".join(missing[:4])
        )

def _asym_bone_count_jump_warning(anchor_counts):
    """Allow sharp count changes, but make the interpolation visible to users."""
    counts = [int(value) for value in anchor_counts]
    if len(counts) < 2:
        return ""
    jumps = [
        (
            index,
            (index + 1) % len(counts),
            abs(value - counts[(index + 1) % len(counts)]),
        )
        for index, value in enumerate(counts)
    ]
    largest = max(jumps, key=lambda item: item[2])
    if largest[2] <= 2:
        return ""
    return (
        f"相邻控制链 C{largest[0]:02d} 与 C{largest[1]:02d} "
        f"的骨骼数相差 {largest[2]}；已按设置生成，中间骨骼链使用线性插值。"
    )


def _required_bones_exist(armature, prefix, rows, columns):
    missing = [
        _bone_name(prefix, column, row)
        for column in range(columns)
        for row in range(rows)
        if armature.data.bones.get(_bone_name(prefix, column, row)) is None
    ]
    if missing:
        raise ValueError(
            "缺少用于赋权的生成骨骼，请先 Only Build Bones："
            + ", ".join(missing[:4])
        )


def _build(context, settings, build_bones, build_weights):
    primary_mesh, mesh_objects, armature, control_objects, selection_items = (
        _validate_build(context, settings, require_selection=True)
    )
    control_points = _control_world_locations(control_objects)
    surface_records = []
    for mesh_obj, selected_indices in selection_items:
        surface_records.extend(_records_from_indices(mesh_obj, selected_indices))
    if len(surface_records) < 4:
        raise ValueError("At least four valid selected vertices are required.")
    surface_points = [record["world"] for record in surface_records]
    surface_normals = [record["normal"] for record in surface_records]
    asymmetric = settings.skirt_shape_mode == 'ASYMMETRIC'

    asymmetric_data = None
    if asymmetric:
        asymmetric_data = _prepare_asymmetric_build_data(
            primary_mesh,
            armature,
            control_points,
            surface_points,
            settings,
        )
        samples = asymmetric_data["preview_grid"]
        bone_z_axes = asymmetric_data["bone_z_axes"]
    else:
        samples, _fitted_sample_normals = build_cage_samples(
            control_points,
            settings.rows,
            settings.columns,
            settings.bone_inset,
            settings.closed_loop,
            surface_points=surface_points,
            surface_normals=surface_normals,
            slice_thickness=settings.slice_thickness,
            fit_mode=settings.fit_mode,
            ring_fit_strength=settings.ring_fit_strength,
            longitudinal_smoothing=settings.longitudinal_smoothing,
            smooth_bone_chains=settings.smooth_bone_chains,
            return_normals=True,
        )
        # Standard proxy behavior and roll reference are unchanged.
        proxy_samples = _uniform_proxy_row_samples(samples)
        proxy_vertex_normals = _proxy_grid_vertex_normals(
            proxy_samples,
            settings.closed_loop,
        )
        bone_z_axes = [
            [-normal for normal in row_normals]
            for row_normals in proxy_vertex_normals
        ]

    _configure_lr_suffix_naming(samples, settings.use_lr_suffix_names if not asymmetric else False)
    state = _ContextState(context)
    created_bones = 0
    assignments = 0
    modifier_added = False
    try:
        if build_bones:
            _activate_object(context, armature, mode='EDIT')
            if settings.clear_existing_generated_bones:
                cleanup_generated_bones(armature, settings.prefix)
            if asymmetric:
                created = create_variable_asymmetric_bones(
                    armature,
                    asymmetric_data["bone_chains"],
                    bone_z_axes,
                    settings.parent_bone,
                    settings.prefix,
                    bone_collection_name=settings.bone_collection,
                )
            else:
                created = create_bones_from_samples(
                    armature,
                    samples,
                    settings.parent_bone,
                    settings.prefix,
                    False,
                    settings.closed_loop,
                    sample_normals=bone_z_axes,
                    bone_collection_name=settings.bone_collection,
                )
            created_bones = len(created)
            bpy.ops.object.mode_set(mode='OBJECT')

            _delete_debug_objects(settings.prefix, primary_mesh.name)
            _delete_proxy_objects(settings.prefix, primary_mesh.name)
            if settings.create_proxy_mesh:
                if asymmetric:
                    proxy_obj = create_asymmetric_proxy_mesh(
                        asymmetric_data["preview_grid"],
                        settings.prefix,
                        primary_mesh.name,
                    )
                    if not settings.workflow_enabled or settings.bind_weights:
                        apply_variable_asymmetric_proxy_weights(
                            proxy_obj,
                            armature,
                            asymmetric_data["preview_grid"],
                            asymmetric_data["bone_counts"],
                            asymmetric_data["chain_lengths"],
                            settings.prefix,
                            settings.parent_bone,
                            settings,
                        )
                else:
                    proxy_obj = create_proxy_mesh(
                        samples,
                        settings.prefix,
                        primary_mesh.name,
                        settings.closed_loop,
                    )
                    apply_proxy_mesh_weights(
                        proxy_obj,
                        armature,
                        control_points,
                        settings.rows,
                        settings.columns,
                        settings.prefix,
                        settings.parent_bone,
                        settings,
                    )
            if settings.create_debug_curves:
                create_debug_curves(
                    samples,
                    settings.prefix,
                    primary_mesh.name,
                    settings.closed_loop,
                )

        if build_weights:
            if asymmetric:
                _required_variable_bones_exist(
                    armature,
                    settings.prefix,
                    asymmetric_data["bone_counts"],
                )
            else:
                _required_bones_exist(
                    armature,
                    settings.prefix,
                    settings.rows,
                    settings.columns,
                )
            for mesh_obj, selected_indices in selection_items:
                _activate_object(context, mesh_obj, mode='OBJECT')
                all_weights = {}
                matrix_world = mesh_obj.matrix_world
                for index in selected_indices:
                    vertex_world = matrix_world @ mesh_obj.data.vertices[index].co
                    if asymmetric:
                        arc_s, local_v = _project_asymmetric_parameters(
                            vertex_world,
                            asymmetric_data["projector"],
                        )
                        parent_weight = _asymmetric_parent_weight(
                            arc_s,
                            local_v,
                            asymmetric_data["chain_lengths"],
                        )
                        all_weights[index] = (
                            compute_variable_asymmetric_vertex_weights(
                                arc_s,
                                local_v,
                                asymmetric_data["bone_counts"],
                                settings.prefix,
                                settings,
                                parent_weight,
                            )
                        )
                    else:
                        h, u = compute_vertex_cage_parameters(
                            vertex_world,
                            control_points,
                            settings.closed_loop,
                        )
                        all_weights[index] = compute_vertex_weights(
                            vertex_world,
                            h,
                            u,
                            settings.rows,
                            settings.columns,
                            settings.prefix,
                            settings,
                        )
                assignments += apply_weights(
                    mesh_obj,
                    selected_indices,
                    all_weights,
                    settings.prefix,
                    settings.parent_bone,
                    settings.clear_existing_generated_vertex_groups,
                    settings.top_blend_to_parent,
                )
                _modifier, added = _ensure_armature_modifier(mesh_obj, armature)
                modifier_added |= added

        collection_name = str(
            primary_mesh.get(
                PROP_COLLECTION,
                _control_collection_name(settings.prefix, primary_mesh.name),
            )
        )
        for mesh_obj in mesh_objects:
            _store_mesh_metadata(
                mesh_obj,
                settings,
                collection_name,
                mesh_objects,
            )
    finally:
        state.restore(context)

    return {
        "bones": created_bones,
        "vertices": sum(len(indices) for _mesh, indices in selection_items)
        if build_weights else 0,
        "assignments": assignments,
        "modifier_added": modifier_added,
        "warning": (
            _asym_bone_count_jump_warning(
                _get_asym_anchor_bone_counts(
                    settings,
                    settings.asym_control_chain_count,
                )
            )
            if asymmetric
            else ""
        ),
    }

def _delete_generated_bones_and_groups(context, settings):
    mesh_objects = _resolve_target_meshes(context, settings)
    armature = settings.target_armature
    _validate_prefix(settings.prefix)
    if not mesh_objects:
        raise ValueError("Target Mesh does not exist.")
    if armature is None or armature.type != 'ARMATURE':
        raise ValueError("Target Armature does not exist.")

    state = _ContextState(context)
    removed_bones = 0
    removed_groups = 0
    try:
        _activate_object(context, armature, mode='EDIT')
        removed_bones = cleanup_generated_bones(armature, settings.prefix)
        bpy.ops.object.mode_set(mode='OBJECT')

        for mesh_obj in mesh_objects:
            _activate_object(context, mesh_obj, mode='OBJECT')
            for group in list(mesh_obj.vertex_groups):
                if group.name.startswith(settings.prefix):
                    mesh_obj.vertex_groups.remove(group)
                    removed_groups += 1
            _delete_debug_objects(settings.prefix, mesh_obj.name)
            _delete_proxy_objects(settings.prefix, mesh_obj.name)
    finally:
        state.restore(context)
    return removed_bones, removed_groups


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def _resample_asym_control_points(control_points, new_count):
    old_count = len(control_points)
    if old_count < 1:
        return []
    result = []
    for chain in range(new_count):
        value = chain * old_count / new_count
        first = int(math.floor(value)) % old_count
        second = (first + 1) % old_count
        factor = smoothstep(value - math.floor(value))
        result.append([
            Vector(control_points[first][level]).lerp(
                Vector(control_points[second][level]),
                factor,
            )
            for level in range(CONTROL_LEVELS)
        ])
    return result


def _resample_cyclic_counts(values, new_count, fallback):
    values = list(values)
    if not values:
        return [int(fallback)] * new_count
    old_count = len(values)
    result = []
    for chain in range(new_count):
        value = chain * old_count / new_count
        first = int(math.floor(value)) % old_count
        second = (first + 1) % old_count
        factor = value - math.floor(value)
        result.append(int(clamp(round(_lerp_float(
            values[first], values[second], factor
        )), 1, 64)))
    return result


def _update_asym_control_chain_count(settings, context):
    global _ASYM_CONTROL_RESAMPLE_GUARD
    if _ASYM_CONTROL_RESAMPLE_GUARD or settings.skirt_shape_mode != 'ASYMMETRIC':
        return
    if context is None or context.scene is None:
        return
    primary_mesh = _resolve_target_mesh(context, settings)
    if primary_mesh is None or primary_mesh.type != 'MESH':
        return
    collection_name = str(primary_mesh.get(
        PROP_COLLECTION,
        _control_collection_name(settings.prefix, primary_mesh.name),
    ))
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return
    controls = find_control_points(collection, settings.prefix, require_all=False)
    if not _controls_are_asymmetric(controls) or not controls:
        return
    old_count = len(controls)
    new_count = int(clamp(settings.asym_control_chain_count, 4, 24))
    if old_count == new_count:
        return

    _ASYM_CONTROL_RESAMPLE_GUARD = True
    try:
        world_points = _control_world_locations(controls)
        new_points = _resample_asym_control_points(world_points, new_count)
        old_counts = _load_stored_asym_bone_counts(
            primary_mesh,
            _get_asym_anchor_bone_counts(settings, old_count),
        )
        new_counts = _resample_cyclic_counts(
            old_counts,
            new_count,
            settings.asym_default_bone_count,
        )
        display_size = max(
            (obj.empty_display_size for obj in _flatten_control_objects(controls)),
            default=0.05,
        )
        for obj in list(_flatten_control_objects(controls)):
            bpy.data.objects.remove(obj, do_unlink=True)
        _set_asym_anchor_bone_counts(settings, new_counts, new_count)
        settings.columns = max(int(settings.columns), new_count)
        new_collection = _create_or_update_asymmetric_controls(
            primary_mesh,
            settings,
            new_points,
            display_size,
            update_existing=False,
        )
        new_controls = find_control_points(
            new_collection,
            settings.prefix,
            require_all=True,
        )
        target_meshes = _stored_target_meshes(primary_mesh)
        for mesh_obj in target_meshes:
            _store_mesh_metadata(
                mesh_obj,
                settings,
                new_collection.name,
                target_meshes,
            )
        if settings.control_edit_active:
            _set_controls_locked(new_controls, False)
            _select_only_controls(context, new_controls)
            settings.control_edit_snapshot = _json_position_map(new_controls)
            settings.control_edit_last_positions = settings.control_edit_snapshot
        _LIVE_PROXY_CACHE["signature"] = None
        for area in context.screen.areas if context.screen else []:
            area.tag_redraw()
    finally:
        _ASYM_CONTROL_RESAMPLE_GUARD = False

class AUTOSKIRT_OT_generate_control_points(bpy.types.Operator):
    bl_idname = "auto_skirt.generate_control_points"
    bl_label = "生成控制点"
    bl_description = "根据编辑模式选区生成标准四向控制点或 4–24 条不对称控制链"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        try:
            collection, count = generate_control_points(context, settings)
            _enter_control_edit_mode(context, settings, collection)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        control_count = sum(
            1 for obj in collection.objects
            if obj.get(PROP_KIND) == EDIT_KIND_CONTROL
        )
        self.report(
            {'INFO'},
            f"已从 {count} 个顶点生成 {control_count} 个控制点，并进入控制点编辑：{collection.name}",
        )
        return {'FINISHED'}


class AUTOSKIRT_OT_refit_control_points(bpy.types.Operator):
    bl_idname = "auto_skirt.refit_control_points"
    bl_label = "按选区重新拟合控制点"
    bl_description = "将现有控制点重新拟合到当前编辑模式选区"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        try:
            count = _refit_control_points(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"已按 {count} 个选中顶点重新拟合控制点")
        return {'FINISHED'}


class AUTOSKIRT_OT_edit_control_points(bpy.types.Operator):
    bl_idname = "auto_skirt.edit_control_points"
    bl_label = "修改控制点"
    bl_description = "解锁已有控制点并进入控制点编辑界面"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        try:
            _enter_control_edit_mode(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "已进入控制点编辑；完成、取消和复原在 3D 视图顶部")
        return {'FINISHED'}


class AUTOSKIRT_OT_finish_control_edit(bpy.types.Operator):
    bl_idname = "auto_skirt.finish_control_edit"
    bl_label = "完成控制点编辑"
    bl_description = "锁定当前控制点位置，只能再次通过面板修改"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        if not settings.control_edit_active:
            self.report({'INFO'}, "当前不在控制点编辑状态")
            return {'CANCELLED'}
        _apply_control_edit_mirror(context)
        _leave_control_edit_mode(context, settings, reset=False)
        self.report({'INFO'}, "控制点已固定")
        return {'FINISHED'}


class AUTOSKIRT_OT_cancel_control_edit(bpy.types.Operator):
    bl_idname = "auto_skirt.cancel_control_edit"
    bl_label = "取消控制点编辑"
    bl_description = "放弃本次控制点移动并锁定控制点"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        if not settings.control_edit_active:
            self.report({'INFO'}, "当前不在控制点编辑状态")
            return {'CANCELLED'}
        _leave_control_edit_mode(context, settings, reset=True)
        self.report({'INFO'}, "已取消并恢复到进入编辑前的位置")
        return {'FINISHED'}


class AUTOSKIRT_OT_reset_control_edit(bpy.types.Operator):
    bl_idname = "auto_skirt.reset_control_edit"
    bl_label = "复原控制点"
    bl_description = "复原到进入本次控制点编辑时的位置，但继续编辑"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        if not settings.control_edit_active:
            self.report({'INFO'}, "当前不在控制点编辑状态")
            return {'CANCELLED'}
        try:
            _mesh_obj, _collection, controls = _control_edit_collection(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        _restore_control_positions(controls, settings.control_edit_snapshot)
        settings.control_edit_last_positions = _json_position_map(controls)
        _select_only_controls(context, controls)
        self.report({'INFO'}, "控制点已复原")
        return {'FINISHED'}


def _persist_asym_count_settings(primary_mesh, settings):
    values = _get_asym_anchor_bone_counts(settings)
    targets = _stored_target_meshes(primary_mesh)
    for mesh_obj in targets:
        mesh_obj[PROP_ASYM_CONTROL_CHAINS] = int(
            settings.asym_control_chain_count
        )
        mesh_obj[PROP_ASYM_BONE_COUNTS] = json.dumps(
            values,
            separators=(",", ":"),
        )
    _LIVE_PROXY_CACHE["signature"] = None


class AUTOSKIRT_OT_auto_asym_bone_counts(bpy.types.Operator):
    bl_idname = "auto_skirt.auto_asym_bone_counts"
    bl_label = "自动计算每链骨骼数"
    bl_description = "按各控制链实际弧长和默认骨骼数自动计算"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        if settings.skirt_shape_mode != 'ASYMMETRIC':
            self.report({'ERROR'}, "仅不对称裙摆使用此功能")
            return {'CANCELLED'}
        try:
            primary_mesh, _collection, controls = _control_edit_collection(
                context, settings
            )
            if not _controls_are_asymmetric(controls):
                raise ValueError("请先重新生成不对称裙摆控制点")
            counts = _auto_asym_anchor_bone_counts(
                settings,
                _control_world_locations(controls),
            )
            _persist_asym_count_settings(primary_mesh, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "已自动计算：" + ", ".join(map(str, counts)))
        return {'FINISHED'}


class AUTOSKIRT_OT_uniform_asym_bone_counts(bpy.types.Operator):
    bl_idname = "auto_skirt.uniform_asym_bone_counts"
    bl_label = "全部使用默认数量"
    bl_description = "把所有控制链骨骼数设为默认每链骨骼数"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        value = int(clamp(settings.asym_default_bone_count, 1, 64))
        _set_asym_anchor_bone_counts(
            settings,
            [value] * settings.asym_control_chain_count,
        )
        primary_mesh = _resolve_target_mesh(context, settings)
        if primary_mesh is not None:
            _persist_asym_count_settings(primary_mesh, settings)
        self.report({'INFO'}, f"所有控制链已设为 {value} 根骨骼")
        return {'FINISHED'}


class AUTOSKIRT_OT_reset_asym_bone_counts(bpy.types.Operator):
    bl_idname = "auto_skirt.reset_asym_bone_counts"
    bl_label = "恢复默认"
    bl_description = "恢复默认每链 6 根骨骼"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        settings.asym_default_bone_count = 6
        _set_asym_anchor_bone_counts(
            settings,
            [6] * settings.asym_control_chain_count,
        )
        primary_mesh = _resolve_target_mesh(context, settings)
        if primary_mesh is not None:
            _persist_asym_count_settings(primary_mesh, settings)
        self.report({'INFO'}, "不对称裙摆骨骼数已恢复为每链 6 根")
        return {'FINISHED'}

class AUTOSKIRT_OT_build(bpy.types.Operator):
    bl_idname = "auto_skirt.build"
    bl_label = "生成自动裙骨"
    bl_description = "生成骨骼、投影权重，或同时生成两者"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(
        name="操作",
        items=(
            ('BOTH', "骨骼和权重", "生成骨骼并应用权重"),
            ('BONES', "仅生成骨骼", "只生成骨骼"),
            ('WEIGHTS', "仅应用权重", "只应用权重"),
        ),
        default='BOTH',
        options={'HIDDEN'},
    )

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        build_bones = self.action in {'BOTH', 'BONES'}
        build_weights = self.action in {'BOTH', 'WEIGHTS'}
        try:
            if settings.control_edit_active:
                _apply_control_edit_mirror(context)
                _leave_control_edit_mode(context, settings, reset=False)
            _force_fixed_weight_defaults(settings)
            result = _build(context, settings, build_bones, build_weights)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        parts = []
        if build_bones:
            parts.append(f"骨骼 {result['bones']} 根")
        if build_weights:
            parts.append(
                f"顶点 {result['vertices']} 个 / 权重写入 {result['assignments']} 次"
            )
            if result["modifier_added"]:
                parts.append("已添加 Armature Modifier")
        self.report({'INFO'}, "；".join(parts))
        if result.get("warning"):
            self.report({'WARNING'}, result["warning"])
        return {'FINISHED'}


class AUTOSKIRT_OT_delete_controls(bpy.types.Operator):
    bl_idname = "auto_skirt.delete_controls"
    bl_label = "删除生成的控制点"
    bl_description = "仅删除本插件标记且前缀匹配的控制点"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        if settings.control_edit_active:
            _leave_control_edit_mode(context, settings, reset=False)

        mesh_objects = _resolve_target_meshes(context, settings)
        if not mesh_objects:
            self.report({'ERROR'}, "Target Mesh does not exist.")
            return {'CANCELLED'}
        primary_mesh = mesh_objects[0]
        try:
            _validate_prefix(settings.prefix)
            prefixes = {
                settings.prefix,
                str(primary_mesh.get(PROP_PREFIX, settings.prefix)),
            }
            removed = sum(
                delete_generated_controls(prefix, primary_mesh.name)
                for prefix in prefixes
            )
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        for mesh_obj in mesh_objects:
            if PROP_COLLECTION in mesh_obj:
                del mesh_obj[PROP_COLLECTION]
            if PROP_TARGETS in mesh_obj:
                del mesh_obj[PROP_TARGETS]
            if PROP_ASYM_BASE_L3 in mesh_obj:
                del mesh_obj[PROP_ASYM_BASE_L3]
            if PROP_ASYM_CONTROL_CHAINS in mesh_obj:
                del mesh_obj[PROP_ASYM_CONTROL_CHAINS]
            if PROP_ASYM_BONE_COUNTS in mesh_obj:
                del mesh_obj[PROP_ASYM_BONE_COUNTS]
            if PROP_SHAPE_MODE in mesh_obj:
                del mesh_obj[PROP_SHAPE_MODE]

        self.report({'INFO'}, f"已删除 {removed} 个生成控制点")
        return {'FINISHED'}


class AUTOSKIRT_OT_delete_bones_groups(bpy.types.Operator):
    bl_idname = "auto_skirt.delete_bones_groups"
    bl_label = "删除生成的骨骼和顶点组"
    bl_description = "删除前缀匹配的生成骨骼、顶点组和调试曲线"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.auto_skirt_settings
        try:
            bones, groups = _delete_generated_bones_and_groups(context, settings)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"已删除 {bones} 根骨骼和 {groups} 个顶点组")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Properties and UI
# ---------------------------------------------------------------------------


class AUTOSKIRT_PG_settings(bpy.types.PropertyGroup):
    # Internal primary mesh of the current multi-mesh edit session.  It is
    # recorded automatically and intentionally has no user-facing selector.
    target_mesh: PointerProperty(
        type=bpy.types.Object,
        poll=_poll_mesh,
        options={'HIDDEN'},
    )
    target_armature: PointerProperty(
        name="目标骨架",
        type=bpy.types.Object,
        poll=_poll_armature,
    )
    parent_bone: StringProperty(
        name="父骨骼",
        default="",
        description="第一层裙骨的父骨骼，同时用于顶部混合顶点组",
    )
    bone_collection: StringProperty(
        name="骨骼集合",
        default="Skirt Bones",
        description="输入已有骨骼集合名称以使用它；不存在时会自动新建",
    )
    prefix: StringProperty(
        name="命名前缀",
        default="AutoSkirt_",
    )
    skirt_shape_mode: EnumProperty(
        name="裙摆类型",
        items=(
            ('STANDARD', "标准裙摆", "使用原有稳定的闭合裙摆算法"),
            ('ASYMMETRIC', "不对称裙摆", "4–24 条可编辑控制链，每条链可设置独立骨骼数"),
        ),
        default='STANDARD',
    )
    use_lr_suffix_names: BoolProperty(
        name="左右后缀 (.L/.R)",
        default=False,
        description="仅标准裙摆使用；不对称裙摆固定按 C/R 编号",
    )
    asym_control_chain_count: IntProperty(
        name="控制点链数",
        default=6,
        min=4,
        max=24,
        description="不对称裙摆的环向控制链数量；修改后从当前控制笼重采样",
        update=_update_asym_control_chain_count,
    )
    asym_default_bone_count: IntProperty(
        name="默认每链骨骼数",
        default=6,
        min=1,
        max=64,
        description="自动计算时与平均裙长对应的骨骼数量",
    )
    asym_chain_bones_00: IntProperty(name="C00", default=6, min=1, max=64)
    asym_chain_bones_01: IntProperty(name="C01", default=6, min=1, max=64)
    asym_chain_bones_02: IntProperty(name="C02", default=6, min=1, max=64)
    asym_chain_bones_03: IntProperty(name="C03", default=6, min=1, max=64)
    asym_chain_bones_04: IntProperty(name="C04", default=6, min=1, max=64)
    asym_chain_bones_05: IntProperty(name="C05", default=6, min=1, max=64)
    asym_chain_bones_06: IntProperty(name="C06", default=6, min=1, max=64)
    asym_chain_bones_07: IntProperty(name="C07", default=6, min=1, max=64)
    asym_chain_bones_08: IntProperty(name="C08", default=6, min=1, max=64)
    asym_chain_bones_09: IntProperty(name="C09", default=6, min=1, max=64)
    rows: IntProperty(
        name="纵向层数",
        default=6,
        min=2,
        max=248,
    )
    columns: IntProperty(
        name="骨骼链数",
        default=12,
        min=4,
        max=256,
    )
    control_levels: IntProperty(
        name="每链控制点数",
        default=4,
        min=2,
        max=32,
    )
    closed_loop: BoolProperty(
        name="闭合环",
        default=True,
        description="闭合 Left 到 Front；关闭时生成 Front→Right→Back→Left 开放链",
        options={'HIDDEN'},
    )
    bone_inset: FloatProperty(
        name="骨骼内缩",
        default=0.05,
        min=0.0,
        max=0.95,
        subtype='FACTOR',
    )
    slice_thickness: FloatProperty(
        name="切片厚度",
        default=0.08,
        min=0.001,
        max=0.5,
        subtype='FACTOR',
        description="按归一化高度选取每层点云的初始半厚度",
    )
    fit_mode: EnumProperty(
        name="骨骼环拟合模式",
        items=(
            ('OUTER_ENVELOPE', "外轮廓", "取当前方向最外层候选点，适合多层裙和大衣"),
            ('WEIGHTED_AVERAGE', "加权平均", "对方向和高度匹配的候选点加权平均"),
            ('MEDIAN', "中位数", "使用有效候选点的中位半径"),
        ),
        default='OUTER_ENVELOPE',
    )
    ring_fit_strength: FloatProperty(
        name="骨骼环拟合强度",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        description="0 保持在控制笼内边界，1 完全贴合点云计算结果",
    )
    longitudinal_smoothing: BoolProperty(
        name="纵向轻度平滑",
        default=True,
        description="对中间层做 0.2/0.6/0.2 平滑，顶底层不变",
    )
    smooth_bone_chains: BoolProperty(
        name="平滑骨骼链",
        default=False,
        description="生成前对每条骨骼链做 3 次纵向平滑，保留首尾关节，抑制面褶皱造成的起伏",
    )
    top_blend_to_parent: BoolProperty(
        name="顶部混合到父骨骼",
        default=True,
    )
    top_blend_height: FloatProperty(
        name="顶部混合高度",
        default=0.15,
        min=0.001,
        max=1.0,
        subtype='FACTOR',
    )
    weight_curve: EnumProperty(
        name="权重曲线",
        items=(
            ('LINEAR', "线性", "层间使用线性插值"),
            ('COSINE', "余弦", "层间使用余弦平滑插值"),
            ('SHARP', "锐利", "层间使用更集中的双曲正切插值"),
        ),
        default='COSINE',
    )
    max_influences: IntProperty(
        name="最大影响数",
        default=4,
        min=1,
        max=16,
        description="启用顶部混合时，父骨骼始终保留一个影响名额",
    )
    normalize_weights: BoolProperty(
        name="权重归一化",
        default=True,
    )
    clear_existing_generated_bones: BoolProperty(
        name="清除已有生成骨骼",
        default=True,
    )
    clear_existing_generated_vertex_groups: BoolProperty(
        name="清除已有生成顶点组",
        default=True,
    )
    create_debug_curves: BoolProperty(
        name="创建调试曲线",
        default=False,
    )
    create_proxy_mesh: BoolProperty(
        name="生成代理网格",
        default=False,
        description="按当前骨骼环采样生成一个只用于检查形状的代理网格",
    )
    control_edit_active: BoolProperty(
        name="正在编辑控制点",
        default=False,
        options={'HIDDEN'},
    )
    control_mirror_x: BoolProperty(
        name="X 镜像",
        default=False,
        description="左右控制点沿全局 X 对称编辑",
    )
    control_mirror_y: BoolProperty(
        name="Y 镜像",
        default=False,
        description="前后控制点沿全局 Y 对称编辑",
    )
    control_live_proxy_preview: BoolProperty(
        name="实时显示代理网格",
        default=True,
        description="编辑控制点时实时显示最终骨骼环对应的代理表面；仅用于预览",
    )
    control_edit_snapshot: StringProperty(
        name="控制点编辑快照",
        default="",
        options={'HIDDEN'},
    )
    control_edit_last_positions: StringProperty(
        name="控制点编辑上一帧位置",
        default="",
        options={'HIDDEN'},
    )
    control_edit_hide_select_state: StringProperty(
        name="控制点编辑选择锁定状态",
        default="",
        options={'HIDDEN'},
    )
    control_edit_mesh_name: StringProperty(
        name="控制点编辑目标网格",
        default="",
        options={'HIDDEN'},
    )


class AUTOSKIRT_PT_main(bpy.types.Panel):
    bl_label = "裙子骨骼生成器"
    bl_idname = "AUTOSKIRT_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "自动裙骨"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.auto_skirt_settings

        layout.use_property_split = True
        layout.use_property_decorate = False

        target_box = layout.box()
        target_box.label(text="当前编辑网格")
        target_box.label(
            text="在 Edit Mode 中选中一个或多个网格及其顶点",
            icon='INFO',
        )
        target_box.prop(settings, "target_armature")
        if settings.target_armature is not None:
            target_box.prop_search(
                settings,
                "parent_bone",
                settings.target_armature.data,
                "bones",
            )
        else:
            target_box.prop(settings, "parent_bone")
        # A plain text field is intentional: prop_search rejects names that
        # do not already exist, while this value may name a new collection.
        target_box.prop(settings, "bone_collection")

        option_box = layout.box()
        option_box.label(text="生成选项")
        option_box.prop(settings, "prefix")
        option_box.prop(settings, "skirt_shape_mode")
        if settings.skirt_shape_mode == 'ASYMMETRIC':
            option_box.prop(settings, "asym_control_chain_count")
            chain_row = option_box.row()
            chain_row.alert = settings.columns < settings.asym_control_chain_count
            chain_row.prop(settings, "columns")
            option_box.prop(settings, "asym_default_bone_count")

            count_box = option_box.box()
            count_box.label(text="每条控制点链骨骼数")
            cardinal = ("Front", "Right", "Back", "Left")
            chain_count = int(settings.asym_control_chain_count)
            for chain in range(chain_count):
                value = chain * 4.0 / chain_count
                segment = int(math.floor(value)) % 4
                fraction = value - math.floor(value)
                if fraction <= 1.0e-5:
                    region = cardinal[segment]
                else:
                    region = f"{cardinal[segment]}→{cardinal[(segment + 1) % 4]}"
                row = count_box.row(align=True)
                row.label(text=f"C{chain:02d}  {region}")
                row.prop(
                    settings,
                    _asym_count_property_name(chain),
                    text="",
                )
            batch = count_box.row(align=True)
            batch.operator(
                AUTOSKIRT_OT_auto_asym_bone_counts.bl_idname,
                text="自动计算",
            )
            batch.operator(
                AUTOSKIRT_OT_uniform_asym_bone_counts.bl_idname,
                text="全部相同",
            )
            batch.operator(
                AUTOSKIRT_OT_reset_asym_bone_counts.bl_idname,
                text="恢复默认",
            )
        else:
            option_box.prop(settings, "use_lr_suffix_names")
            option_box.prop(settings, "rows")
            option_box.prop(settings, "columns")
        option_box.prop(settings, "smooth_bone_chains")
        option_box.prop(settings, "create_proxy_mesh")

        if settings.control_edit_active:
            edit_box = layout.box()
            edit_box.label(text="正在编辑控制点", icon='EMPTY_AXIS')
            edit_box.label(text="完成 / 取消 / 复原在 3D 视图顶部")
            edit_box.prop(settings, "control_mirror_x", text="X 镜像")
            edit_box.prop(settings, "control_mirror_y", text="Y 镜像")
            edit_box.prop(settings, "control_live_proxy_preview")

        layout.separator()
        layout.operator(
            AUTOSKIRT_OT_generate_control_points.bl_idname,
            text="生成控制点并编辑",
            icon='EMPTY_AXIS',
        )
        layout.operator(
            AUTOSKIRT_OT_edit_control_points.bl_idname,
            text="修改控制点",
            icon='GREASEPENCIL',
        )

        layout.separator()
        operator = layout.operator(
            AUTOSKIRT_OT_build.bl_idname,
            text="生成骨骼和权重",
            icon='ARMATURE_DATA',
        )
        operator.action = 'BOTH'

        layout.separator()
        maintenance = layout.row(align=True)
        maintenance.operator(
            AUTOSKIRT_OT_delete_controls.bl_idname,
            text="删控制点",
            icon='TRASH',
        )
        maintenance.operator(
            AUTOSKIRT_OT_delete_bones_groups.bl_idname,
            text="删生成项",
            icon='TRASH',
        )

classes = (
    AUTOSKIRT_PG_settings,
    AUTOSKIRT_OT_generate_control_points,
    AUTOSKIRT_OT_refit_control_points,
    AUTOSKIRT_OT_edit_control_points,
    AUTOSKIRT_OT_finish_control_edit,
    AUTOSKIRT_OT_cancel_control_edit,
    AUTOSKIRT_OT_reset_control_edit,
    AUTOSKIRT_OT_auto_asym_bone_counts,
    AUTOSKIRT_OT_uniform_asym_bone_counts,
    AUTOSKIRT_OT_reset_asym_bone_counts,
    AUTOSKIRT_OT_build,
    AUTOSKIRT_OT_delete_controls,
    AUTOSKIRT_OT_delete_bones_groups,
    AUTOSKIRT_PT_main,
)


unified_generator.configure(sys.modules[__name__], sheet_generator)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.auto_skirt_settings = PointerProperty(
        type=AUTOSKIRT_PG_settings
    )
    try:
        bpy.types.VIEW3D_HT_header.remove(_draw_control_edit_header)
    except Exception:
        pass
    bpy.types.VIEW3D_HT_header.append(_draw_control_edit_header)
    _install_selection_marker_draw_handler()
    if _auto_skirt_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_auto_skirt_depsgraph_update)
    sheet_generator.register()


def unregister():
    sheet_generator.unregister()
    _remove_selection_marker_draw_handler()
    if _auto_skirt_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_auto_skirt_depsgraph_update)
    try:
        bpy.types.VIEW3D_HT_header.remove(_draw_control_edit_header)
    except Exception:
        pass
    if hasattr(bpy.types.Scene, "auto_skirt_settings"):
        del bpy.types.Scene.auto_skirt_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
