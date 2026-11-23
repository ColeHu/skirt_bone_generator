# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""Unified closed-skirt/open-surface feature layer.

The mature fitting and weighting implementations remain in ``__init__`` and
``sheet_generator``. This module adds their shared UI and editing features.
"""

import json
import math
import time

import bpy
import gpu
from bpy.props import BoolProperty, IntProperty, EnumProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

SCHEMA_VERSION = 2
PROP_SCHEMA = "auto_skirt_unified_schema"
PROP_GENERATOR_TYPE = "auto_skirt_unified_type"
PROP_REVERSES = "auto_skirt_chain_reverses"
PROP_NORMAL_VERSION = "auto_skirt_control_normal_version"
PROP_AUTO_QUATERNION = "auto_skirt_auto_normal_quaternion"

CHAIN_COLORS = (
    (0.18, 0.82, 1.00, 1.0),
    (0.10, 0.95, 0.68, 1.0),
    (0.46, 0.96, 0.28, 1.0),
    (0.86, 0.96, 0.24, 1.0),
    (1.00, 0.82, 0.24, 1.0),
    (0.56, 0.78, 1.00, 1.0),
    (0.34, 0.54, 1.00, 1.0),
    (0.54, 0.38, 1.00, 1.0),
    (0.76, 0.42, 1.00, 1.0),
    (0.94, 0.54, 0.96, 1.0),
    (0.44, 0.96, 0.88, 1.0),
    (0.96, 0.94, 0.62, 1.0),
    (1.00, 0.40, 0.32, 1.0), (0.98, 0.68, 0.30, 1.0),
    (0.73, 0.94, 0.16, 1.0), (0.28, 0.90, 0.40, 1.0),
    (0.12, 0.84, 0.78, 1.0), (0.35, 0.68, 0.98, 1.0),
    (0.42, 0.38, 0.95, 1.0), (0.69, 0.32, 0.96, 1.0),
    (0.90, 0.32, 0.78, 1.0), (1.00, 0.47, 0.62, 1.0),
    (0.95, 0.65, 0.52, 1.0), (0.65, 0.84, 0.48, 1.0),
)

_M = None
_S = None
_CONFIGURED = False
_ACTIVE_SKIRT_DATA = None
_ACTIVE_SHEET_DATA = None
_PARAMETER_CACHE = {}
_PREVIEW_MODE = False
_PREVIEW_CACHE = {
    kind: {
        "input_signature": None,
        "quick": ([], []),
        "full": ([], []),
        "changed_at": 0.0,
        "full_at": 0.0,
        "timer": False,
        "normal_lines": [],
    }
    for kind in ("SKIRT", "SURFACE")
}


def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def _smoothstep(value):
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _safe_normalized(value, fallback=(0.0, 0.0, 1.0)):
    vector = Vector(value)
    if vector.length_squared <= 1.0e-10:
        vector = Vector(fallback)
    if vector.length_squared <= 1.0e-10:
        vector = Vector((0.0, 0.0, 1.0))
    return vector.normalized()


def _average(points):
    points = [Vector(point) for point in points]
    if not points:
        return Vector((0.0, 0.0, 0.0))
    result = Vector((0.0, 0.0, 0.0))
    for point in points:
        result += point
    return result / len(points)


def _chain_count_property(index):
    return f"asym_chain_bones_{int(index):02d}"


def _chain_reverse_property(index):
    return f"asym_chain_reverse_{int(index):02d}"


def _sheet_count_property(index):
    return f"chain_bones_{int(index):02d}"


def _sheet_reverse_property(index):
    return f"chain_reverse_{int(index):02d}"


def _main_counts(settings, count=None):
    count = int(count if count is not None else settings.asym_control_chain_count)
    return [
        int(_clamp(getattr(settings, _chain_count_property(index)), 1, 64))
        for index in range(count)
    ]


def _main_reverses(settings, count=None):
    count = int(count if count is not None else settings.asym_control_chain_count)
    return [
        bool(getattr(settings, _chain_reverse_property(index)))
        for index in range(count)
    ]


def _sheet_counts(settings, count=None):
    count = int(count if count is not None else settings.control_chain_count)
    return [
        int(_clamp(getattr(settings, _sheet_count_property(index)), 1, 64))
        for index in range(count)
    ]


def _sheet_reverses(settings, count=None):
    count = int(count if count is not None else settings.control_chain_count)
    return [
        bool(getattr(settings, _sheet_reverse_property(index)))
        for index in range(count)
    ]


def _set_main_counts(settings, values, count=None):
    count = int(count if count is not None else settings.asym_control_chain_count)
    values = list(values)
    fallback = int(_clamp(settings.asym_default_bone_count, 1, 64))
    for index in range(count):
        value = values[index] if index < len(values) else fallback
        setattr(settings, _chain_count_property(index), int(_clamp(round(float(value)), 1, 64)))


def _set_main_reverses(settings, values, count=None):
    count = int(count if count is not None else settings.asym_control_chain_count)
    values = list(values)
    for index in range(count):
        setattr(settings, _chain_reverse_property(index), bool(values[index]) if index < len(values) else False)


def _set_sheet_counts(settings, values, count=None):
    count = int(count if count is not None else settings.control_chain_count)
    values = list(values)
    fallback = int(_clamp(settings.default_bone_count, 1, 64))
    for index in range(count):
        value = values[index] if index < len(values) else fallback
        setattr(settings, _sheet_count_property(index), int(_clamp(round(float(value)), 1, 64)))


def _set_sheet_reverses(settings, values, count=None):
    count = int(count if count is not None else settings.control_chain_count)
    values = list(values)
    for index in range(count):
        setattr(settings, _sheet_reverse_property(index), bool(values[index]) if index < len(values) else False)


def _uses_open_surface(settings):
    """The public UI is one skirt generator; an unchecked loop uses the old sheet core."""
    return not bool(getattr(settings, "closed_skirt", True))


def _slerp_normal(first, second, factor):
    """Shortest-path spherical interpolation with an antipodal fallback."""
    first = _safe_normalized(first)
    second = _safe_normalized(second, first)
    if first.dot(second) < 0.0:
        second.negate()
    dot = _clamp(first.dot(second), -1.0, 1.0)
    factor = _clamp(factor)
    if dot > 0.9995:
        return _safe_normalized(first.lerp(second, factor), first)
    angle = math.acos(dot)
    sine = math.sin(angle)
    if abs(sine) <= 1.0e-8:
        return first
    return _safe_normalized(
        first * (math.sin((1.0 - factor) * angle) / sine)
        + second * (math.sin(factor * angle) / sine),
        first,
    )


def _vertical_tangent(chain, level):
    level = int(level)
    before = Vector(chain[max(0, level - 1)].matrix_world.translation)
    after = Vector(chain[min(len(chain) - 1, level + 1)].matrix_world.translation)
    return _safe_normalized(after - before, (0.0, 0.0, -1.0))


def _quaternion_for_tangent_normal(tangent, normal):
    """Create an object orientation with local +Y=tangent and local +Z=normal."""
    y_axis = _safe_normalized(tangent, (0.0, 0.0, -1.0))
    z_axis = Vector(normal) - y_axis * Vector(normal).dot(y_axis)
    if z_axis.length_squared <= 1.0e-10:
        z_axis = y_axis.cross(Vector((1.0, 0.0, 0.0)))
    if z_axis.length_squared <= 1.0e-10:
        z_axis = y_axis.cross(Vector((0.0, 1.0, 0.0)))
    z_axis = _safe_normalized(z_axis)
    x_axis = _safe_normalized(y_axis.cross(z_axis), (1.0, 0.0, 0.0))
    z_axis = _safe_normalized(x_axis.cross(y_axis), z_axis)
    matrix = Matrix((
        (x_axis.x, y_axis.x, z_axis.x),
        (x_axis.y, y_axis.y, z_axis.y),
        (x_axis.z, y_axis.z, z_axis.z),
    ))
    return matrix.to_quaternion()


def _set_control_normal(obj, tangent, normal, mark_auto=False):
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = _quaternion_for_tangent_normal(tangent, normal)
    obj[PROP_NORMAL_VERSION] = SCHEMA_VERSION
    if mark_auto:
        obj[PROP_AUTO_QUATERNION] = [
            float(obj.rotation_quaternion.w),
            float(obj.rotation_quaternion.x),
            float(obj.rotation_quaternion.y),
            float(obj.rotation_quaternion.z),
        ]


def _control_is_custom(obj):
    stored = obj.get(PROP_AUTO_QUATERNION)
    try:
        if stored is None or len(stored) != 4:
            return True
    except TypeError:
        return True
    current = obj.rotation_quaternion
    direct = sum((float(stored[index]) - float(current[index])) ** 2 for index in range(4))
    negated = sum((float(stored[index]) + float(current[index])) ** 2 for index in range(4))
    return min(direct, negated) > 1.0e-10


def _control_normal(obj, tangent):
    """Read only the twist component around a control chain tangent."""
    normal = obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    tangent = _safe_normalized(tangent, (0.0, 0.0, -1.0))
    normal -= tangent * normal.dot(tangent)
    if normal.length_squared <= 1.0e-10:
        normal = tangent.cross(Vector((1.0, 0.0, 0.0)))
    if normal.length_squared <= 1.0e-10:
        normal = tangent.cross(Vector((0.0, 1.0, 0.0)))
    return _safe_normalized(normal)


def _control_chain_normals(chains):
    result = []
    for chain in chains:
        values = [
            _control_normal(obj, _vertical_tangent(chain, level))
            for level, obj in enumerate(chain)
        ]
        for level in range(1, len(values)):
            if values[level - 1].dot(values[level]) < 0.0:
                values[level].negate()
        result.append(values)
    return result


def _interpolate_vertical_normal(values, factor):
    if len(values) < 2:
        return _safe_normalized(values[0] if values else (0.0, 0.0, 1.0))
    scaled = _clamp(factor) * (len(values) - 1)
    first = min(int(math.floor(scaled)), len(values) - 2)
    return _slerp_normal(values[first], values[first + 1], _smoothstep(scaled - first))


def _interpolate_control_normal(chain_normals, horizontal, vertical, closed):
    count = len(chain_normals)
    if count == 1:
        return _interpolate_vertical_normal(chain_normals[0], vertical)
    if closed:
        value = (float(horizontal) % 1.0) * count
        first = int(math.floor(value)) % count
        second = (first + 1) % count
        factor = value - math.floor(value)
    else:
        value = _clamp(horizontal) * (count - 1)
        first = min(int(math.floor(value)), count - 2)
        second = first + 1
        factor = value - first
    first_normal = _interpolate_vertical_normal(chain_normals[first], vertical)
    second_normal = _interpolate_vertical_normal(chain_normals[second], vertical)
    return _slerp_normal(first_normal, second_normal, _smoothstep(factor))


def _grid_default_normals(points, closed, invert_open=False):
    """Orient the cage normal outward, with a continuous fallback at pinched tips."""
    rows = len(points)
    columns = len(points[0]) if rows else 0
    result = [[Vector((0.0, 1.0, 0.0)) for _ in range(columns)] for _ in range(rows)]
    for row in range(rows):
        center = _average(points[row]) if closed else None
        before_row = max(0, row - 1)
        after_row = min(rows - 1, row + 1)
        for column in range(columns):
            if closed:
                before_column = (column - 1) % columns
                after_column = (column + 1) % columns
            else:
                before_column = max(0, column - 1)
                after_column = min(columns - 1, column + 1)
            horizontal = Vector(points[row][after_column]) - Vector(points[row][before_column])
            vertical = Vector(points[after_row][column]) - Vector(points[before_row][column])
            fallback = result[row - 1][column] if row else Vector((0, -1, 0))
            normal = vertical.cross(horizontal)
            if closed and normal.dot(points[row][column] - center) < 0:
                normal.negate()
            elif not closed and invert_open:
                normal.negate()
            result[row][column] = _safe_normalized(normal, fallback)
    return result


def _initialize_chain_orientations(chains, closed, invert=False):
    if not chains:
        return
    bpy.context.view_layer.update()
    grid = [
        [Vector(chains[column][row].matrix_world.translation) for column in range(len(chains))]
        for row in range(len(chains[0]))
    ]
    desired = _grid_default_normals(grid, closed, chains[0][0].get("invert_open_normal", False))
    for column, chain in enumerate(chains):
        for level, obj in enumerate(chain):
            normal = desired[level][column]
            if invert:
                normal.negate()
            _set_control_normal(
                obj,
                _vertical_tangent(chain, level),
                normal,
                mark_auto=True,
            )


def _mirror_vector(vector, axis):
    result = Vector(vector)
    if axis == 'X':
        result.x = -result.x
    else:
        result.y = -result.y
    return result


def _mirror_control_orientation(
    source,
    source_chain,
    source_level,
    target,
    target_chain,
    target_level,
    axis,
):
    source_normal = _control_normal(
        source,
        _vertical_tangent(source_chain, source_level),
    )
    _set_control_normal(
        target,
        _vertical_tangent(target_chain, target_level),
        _mirror_vector(source_normal, axis),
        mark_auto=not _control_is_custom(source),
    )


def _snapshot_value(obj):
    quaternion = obj.rotation_quaternion.copy()
    automatic = obj.get(PROP_AUTO_QUATERNION)
    try:
        valid_automatic = automatic is not None and len(automatic) == 4
    except TypeError:
        valid_automatic = False
    if not valid_automatic:
        automatic = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
    return [
        float(obj.location.x), float(obj.location.y), float(obj.location.z),
        float(quaternion.w), float(quaternion.x), float(quaternion.y), float(quaternion.z),
        *(float(value) for value in automatic),
    ]


def _restore_snapshot_object(obj, value):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return
    obj.location = Vector((float(value[0]), float(value[1]), float(value[2])))
    if len(value) >= 7:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = tuple(float(component) for component in value[3:7])
    if len(value) >= 11:
        obj[PROP_AUTO_QUATERNION] = [float(component) for component in value[7:11]]


def _snapshot_changed(first, second):
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        return False
    length = min(len(first), len(second))
    return (
        len(first) != len(second)
        or any(abs(float(first[index]) - float(second[index])) > 1.0e-6 for index in range(length))
    )

# ---------------------------------------------------------------------------
# Dynamic open-surface implementation
# ---------------------------------------------------------------------------


def _inverse_cumulative(cumulative, value):
    total = cumulative[-1] if cumulative else 0.0
    if total <= 1.0e-10:
        return _clamp(value)
    target = _clamp(value) * total
    low = 0
    high = len(cumulative) - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if cumulative[middle] <= target:
            low = middle
        else:
            high = middle
    span = cumulative[low + 1] - cumulative[low]
    factor = 0.0 if span <= 1.0e-10 else (target - cumulative[low]) / span
    return (low + factor) / max(len(cumulative) - 1, 1)


def _quad_strip_rows(component, width_axis):
    """Recover the rectangular quad lattice without relying on vertex numbering."""
    faces = component.get("faces", ())
    if not faces or any(len(face) != 4 for face in faces):
        return None
    points = {r["index"]: r["world"] for r in component["records"]}
    neighbors = {i: set() for i in points}
    edges = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edge = (min(a, b), max(a, b))
            edges[edge] = edges.get(edge, 0) + 1
            neighbors[a].add(b)
            neighbors[b].add(a)
    boundary = {}
    for (a, b), count in edges.items():
        if count > 2:
            return None
        if count == 1:
            boundary.setdefault(a, []).append(b)
            boundary.setdefault(b, []).append(a)
    corners = {i for i in boundary if len(neighbors[i]) == 2}
    if len(corners) != 4 or any(len(adj) != 2 for adj in boundary.values()):
        return None
    start = min(corners)
    previous, current = start, boundary[start][0]
    sides, side = [], [start]
    visited = {start}
    while current not in visited:
        visited.add(current)
        side.append(current)
        if current in corners:
            sides.append(side)
            side = [current]
        a, b = boundary[current]
        previous, current = current, b if a == previous else a
    sides.append(side + [start])
    if current != start or len(visited) != len(boundary) or len(sides) != 4:
        return None
    z = [sum(points[i].z for i in side) / len(side) for side in sides]
    top = max(range(4), key=lambda k: z[k] - z[(k + 2) % 4])
    row = sides[top]
    if (points[row[-1]] - points[row[0]]).dot(width_axis) < 0:
        row = row[::-1]
    rows = []
    used = set()
    while True:
        if len(set(row)) != len(row) or any(b not in neighbors[a] for a, b in zip(row, row[1:])):
            return None
        rows.append([points[i] for i in row])
        used.update(row)
        if len(used) == len(points):
            return rows if set(row) == set(sides[(top + 2) % 4]) else None
        choices = [neighbors[i] - used for i in row]
        if any(len(options) != 1 for options in choices):
            return None
        row = [next(iter(options)) for options in choices]


def _fit_open_region_grid(components, chain_count=4, levels=4):
    """Fit K open control chains at average horizontal arc-length positions."""
    chain_count = int(_clamp(chain_count, 4, 24))
    records = [record for component in components for record in component["records"]]
    if len(records) < 4:
        raise ValueError("至少需要 4 个有效顶点来拟合曲面控制网格")
    center, axis = _S.width_axis(records)
    projections = [(record["world"] - center).dot(axis) for record in records]
    minimum, maximum = min(projections), max(projections)
    top = max(record["world"].z for record in records)
    bottom = min(record["world"].z for record in records)
    if maximum - minimum <= _S.EPS:
        raise ValueError("选区横向宽度过小，无法生成控制链")
    if top - bottom <= _S.EPS:
        raise ValueError("选区缺少从上到下的高度差")

    if len(components) == 1:
        rows = _quad_strip_rows(components[0], axis)
        if rows:
            # Follow each fabric cross-section around bends, keeping both free edges.
            sections = [_S.resample_polyline(row, chain_count) for row in rows]
            rails = [_S.resample_polyline(rail, levels) for rail in zip(*sections)]
            radius = max((r["world"] - center).length for r in records)
            return list(map(list, zip(*rails))), max(radius * 0.04, (top - bottom) * 0.015, 0.002)

    parameters = []
    for record, projection in zip(records, projections):
        parameters.append((
            _clamp((projection - minimum) / (maximum - minimum)),
            _clamp((top - record["world"].z) / (top - bottom)),
        ))

    # Estimate one common width parameter by averaging arc lengths over all
    # four vertical levels.  This keeps the initial open controls easy to edit
    # even when the surface bends strongly in top view.
    resolution = 128
    dense_rows = []
    for level in range(_S.LEVELS):
        target_v = level / (_S.LEVELS - 1)
        dense_rows.append([
            _S.weighted_candidate_point(
                records,
                parameters,
                sample / resolution,
                target_v,
                level,
            )
            for sample in range(resolution + 1)
        ])
    cumulative = [0.0]
    for sample in range(1, resolution + 1):
        distance = sum(
            (dense_rows[level][sample] - dense_rows[level][sample - 1]).length
            for level in range(_S.LEVELS)
        ) / _S.LEVELS
        cumulative.append(cumulative[-1] + max(distance, 1.0e-10))
    raw_us = [
        _inverse_cumulative(cumulative, chain / (chain_count - 1))
        for chain in range(chain_count)
    ]

    tip_points = [
        Vector(component["tip_point"])
        for component in components
        if component.get("tip_point") is not None
    ]
    tip = _S.coordinate_median(tip_points) if tip_points else None
    grid = []
    for level in range(levels):
        target_v = level / (levels - 1)
        if level == levels - 1 and tip is not None:
            grid.append([tip.copy() for _chain in range(chain_count)])
        else:
            grid.append([
                _S.weighted_candidate_point(records, parameters, raw_u, target_v, level, levels)
                for raw_u in raw_us
            ])
    diagonal = max((record["world"] - center).length for record in records) * 2.0
    return grid, max(diagonal * 0.02, (top - bottom) * 0.015, 0.002)


def _sheet_create_controls(primary, settings, region_grids, display_size):
    previous = bpy.data.collections.get(str(primary.get(_S.PROP_COLLECTION, "")))
    old_prefix = str(previous.get("control_prefix", primary.get(_S.PROP_PREFIX, settings.prefix)) if previous else primary.get(_S.PROP_PREFIX, settings.prefix))
    _S.delete_controls(old_prefix, primary.name)
    if old_prefix != settings.prefix:
        _S.delete_controls(settings.prefix, primary.name)
    collection = _S.get_or_create_collection(
        _S.control_collection_name(settings.prefix, primary.name)
    )
    primary[_S.PROP_COLLECTION] = collection.name
    collection["control_prefix"] = settings.prefix
    collection["control_levels"] = len(region_grids[0])
    for region, grid in enumerate(region_grids):
        for row in range(len(grid)):
            for column in range(len(grid[row])):
                obj = bpy.data.objects.new(
                    _S.control_name(settings.prefix, region, row, column),
                    None,
                )
                collection.objects.link(obj)
                obj.empty_display_type = 'SPHERE'
                obj.empty_display_size = display_size
                obj.location = grid[row][column]
                obj.rotation_mode = 'QUATERNION'
                obj.show_in_front = True
                obj.color = CHAIN_COLORS[column % len(CHAIN_COLORS)]
                obj[_S.PROP_KIND] = _S.KIND_CONTROL
                obj["auto_sheet_target"] = primary.name
                obj["auto_sheet_region"] = region
                obj["auto_sheet_row"] = row
                obj["invert_open_normal"] = bool(primary.get("invert_open_normal", False))
                obj["auto_sheet_column"] = column
                _S.set_locked(obj, True)
    controls, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
    for grid in controls:
        _initialize_chain_orientations(_sheet_grid_chains(grid), False)
    return collection


def _sheet_find_controls(primary, prefix, require=True):
    collection_name = str(primary.get(
        _S.PROP_COLLECTION,
        _S.control_collection_name(prefix, primary.name),
    ))
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        if require:
            raise ValueError("找不到非闭合裙摆控制点集合，请先重新生成控制点")
        return [], None
    controls = {}
    for obj in collection.objects:
        if obj.get(_S.PROP_KIND) != _S.KIND_CONTROL or obj.get("auto_sheet_target") != primary.name:
            continue
        key = (
            int(obj.get("auto_sheet_region", -1)),
            int(obj.get("auto_sheet_row", -1)),
            int(obj.get("auto_sheet_column", -1)),
        )
        controls[key] = obj
    region_count = max((key[0] for key in controls), default=-1) + 1
    grids = []
    missing = []
    for region in range(region_count):
        keys = [key for key in controls if key[0] == region]
        column_count = max((key[2] for key in keys), default=-1) + 1
        grid = []
        for row in range(int(collection.get("control_levels", 4))):
            grid_row = []
            for column in range(column_count):
                obj = controls.get((region, row, column))
                if obj is None:
                    missing.append(_S.control_name(prefix, region, row, column))
                grid_row.append(obj)
            grid.append(grid_row)
        grids.append(grid)
    if require and (not grids or missing or any(len(row) < 4 for grid in grids for row in grid)):
        raise ValueError(
            "控制点版本不兼容或不完整，请删除后重新生成："
            + ", ".join(missing[:4])
        )
    return grids, collection


def _sheet_grid_chains(grid):
    if not grid:
        return []
    return [
        [row[column] for row in grid]
        for column in range(len(grid[0]))
    ]


def _sheet_position_map(grids):
    return {
        obj.name: _snapshot_value(obj)
        for obj in _S.flatten_controls(grids)
    }


def _sheet_restore_positions(grids, raw):
    snapshot = _S.parse_dict(raw)
    for obj in _S.flatten_controls(grids):
        value = snapshot.get(obj.name)
        if value is not None:
            _restore_snapshot_object(obj, value)


def _interpolate_open_curve(points, value):
    """Open centripetal Catmull-Rom through any 4–24 control points."""
    count = len(points)
    if count < 2:
        raise ValueError("横向控制曲线至少需要 2 个控制点")
    value = _clamp(value)
    if value <= _S.EPS:
        return Vector(points[0])
    if value >= 1.0 - _S.EPS:
        return Vector(points[-1])
    scaled = value * (count - 1)
    segment = min(int(math.floor(scaled)), count - 2)
    factor = scaled - segment
    first = Vector(points[segment])
    second = Vector(points[segment + 1])
    before = Vector(points[segment - 1]) if segment > 0 else first * 2.0 - second
    after = Vector(points[segment + 2]) if segment + 2 < count else second * 2.0 - first
    t0 = 0.0
    t1 = t0 + _S._centripetal_interval(before, first)
    t2 = t1 + _S._centripetal_interval(first, second)
    t3 = t2 + _S._centripetal_interval(second, after)
    target = t1 + (t2 - t1) * factor
    a1 = _S._knot_lerp(before, first, t0, t1, target)
    a2 = _S._knot_lerp(first, second, t1, t2, target)
    a3 = _S._knot_lerp(second, after, t2, t3, target)
    b1 = _S._knot_lerp(a1, a2, t0, t2, target)
    b2 = _S._knot_lerp(a2, a3, t1, t3, target)
    return _S._knot_lerp(b1, b2, t1, t2, target)


def _open_average_arc_mapping(control_grid, resolution=512):
    resolution = max(64, int(resolution))
    vertical_samples = tuple(index / 8 for index in range(9))
    previous = [
        _S.evaluate_patch(control_grid, 0.0, value)
        for value in vertical_samples
    ]
    cumulative = [0.0]
    for sample in range(1, resolution + 1):
        raw_u = sample / resolution
        current = [
            _S.evaluate_patch(control_grid, raw_u, value)
            for value in vertical_samples
        ]
        distance = sum(
            (current[index] - previous[index]).length
            for index in range(len(vertical_samples))
        ) / len(vertical_samples)
        cumulative.append(cumulative[-1] + max(distance, 1.0e-10))
        previous = current
    return cumulative


def _correct_open_chain_spacing(control_grid, raw_us, passes=2):
    raw_us = list(raw_us)
    if len(raw_us) <= 2:
        return raw_us
    vertical_samples = tuple(index / 8 for index in range(9))
    for _pass in range(max(0, int(passes))):
        segment_lengths = []
        for index in range(len(raw_us) - 1):
            segment_lengths.append(sum(
                (
                    _S.evaluate_patch(control_grid, raw_us[index + 1], value)
                    - _S.evaluate_patch(control_grid, raw_us[index], value)
                ).length
                for value in vertical_samples
            ) / len(vertical_samples))
        cumulative = [0.0]
        for length in segment_lengths:
            cumulative.append(cumulative[-1] + max(length, 1.0e-10))
        total = cumulative[-1]
        corrected = [0.0]
        for chain in range(1, len(raw_us) - 1):
            target = total * chain / (len(raw_us) - 1)
            source = 0
            while source < len(cumulative) - 2 and cumulative[source + 1] < target:
                source += 1
            span = cumulative[source + 1] - cumulative[source]
            factor = 0.0 if span <= 1.0e-10 else (target - cumulative[source]) / span
            corrected.append(raw_us[source] * (1.0 - factor) + raw_us[source + 1] * factor)
        corrected.append(1.0)
        raw_us = corrected
    return raw_us


def _sheet_interpolated_layout(settings, raw_us):
    anchor_counts = _sheet_counts(settings)
    anchor_reverses = _sheet_reverses(settings)
    control_count = len(anchor_counts)
    bone_counts = []
    reverse_flags = []
    anchor_columns = []
    used_columns = set()
    for anchor in range(control_count):
        target = anchor / max(control_count - 1, 1)
        if anchor == 0:
            column = 0
        elif anchor == control_count - 1:
            column = len(raw_us) - 1
        else:
            candidates = sorted(
                range(1, len(raw_us) - 1),
                key=lambda candidate: abs(raw_us[candidate] - target),
            )
            column = next(candidate for candidate in candidates if candidate not in used_columns)
        used_columns.add(column)
        anchor_columns.append(column)
    for raw_u in raw_us:
        value = _clamp(raw_u) * (control_count - 1)
        first = min(int(math.floor(value)), control_count - 2)
        second = first + 1
        factor = value - first
        bone_counts.append(int(_clamp(round(
            anchor_counts[first] * (1.0 - factor) + anchor_counts[second] * factor
        ), 1, 64)))
        nearest = int(_clamp(round(value), 0, control_count - 1))
        reverse_flags.append(bool(anchor_reverses[nearest]))
    for anchor, column in enumerate(anchor_columns):
        bone_counts[column] = anchor_counts[anchor]
        reverse_flags[column] = anchor_reverses[anchor]
    return bone_counts, reverse_flags, anchor_columns


def _build_open_surface_data(control_grid, settings, control_objects=None, low_resolution=False):
    chains = int(settings.chains)
    if chains < int(settings.control_chain_count):
        raise ValueError("骨骼链数不能少于控制点链数")
    cumulative = _open_average_arc_mapping(
        control_grid,
        resolution=max(64, chains * (4 if low_resolution else 24)),
    )
    raw_us = [
        _inverse_cumulative(cumulative, chain / max(chains - 1, 1))
        for chain in range(chains)
    ]
    if not low_resolution:
        raw_us = _correct_open_chain_spacing(control_grid, raw_us, passes=2)
    bone_counts, reverse_flags, anchor_columns = _sheet_interpolated_layout(settings, raw_us)
    maximum_bones = max(bone_counts)
    dense_v = max(17 if low_resolution else 49, maximum_bones * (3 if low_resolution else 10) + 1)
    bone_chains = []
    preview_chains = []
    for column, raw_u in enumerate(raw_us):
        dense = [
            _S.evaluate_patch(control_grid, raw_u, step / (dense_v - 1))
            for step in range(dense_v)
        ]
        bone_chains.append(_S.resample_polyline(dense, bone_counts[column] + 1))
        preview_chains.append(_S.resample_polyline(dense, maximum_bones + 1))
    preview_grid = [
        [preview_chains[column][row] for column in range(chains)]
        for row in range(maximum_bones + 1)
    ]
    if _S.is_collapsed_free_end(preview_grid):
        tip = _average(preview_grid[-1])
        preview_grid[-1] = [tip.copy() for _column in range(chains)]
        for column, chain in enumerate(bone_chains):
            chain[-1] = tip.copy()

    proxy_normals = _S.proxy_normals(preview_grid)
    geometric_z_axes = []
    for column, count in enumerate(bone_counts):
        column_normals = [-row[column] for row in proxy_normals]
        geometric_z_axes.append([
            _M._interpolate_vector_samples(column_normals, joint / max(count, 1))
            for joint in range(count + 1)
        ])
    has_custom_normals = bool(control_objects) and any(
        _control_is_custom(obj)
        for chain in control_objects
        for obj in chain
    )
    if has_custom_normals:
        current_normals = _control_chain_normals(control_objects)
        anchor_normals = []
        for anchor, chain in enumerate(control_objects):
            generated_column = anchor_columns[anchor]
            anchor_normals.append([
                current_normals[anchor][level]
                if _control_is_custom(obj)
                else _M._interpolate_vector_samples(
                    geometric_z_axes[generated_column],
                    level / 3.0,
                )
                for level, obj in enumerate(chain)
            ])
        bone_z_axes = [
            [
                _interpolate_control_normal(
                    anchor_normals,
                    raw_us[column],
                    joint / max(bone_counts[column], 1),
                    False,
                )
                for joint in range(bone_counts[column] + 1)
            ]
            for column in range(chains)
        ]
    else:
        bone_z_axes = geometric_z_axes
    return {
        "bone_chains": bone_chains,
        "preview_grid": preview_grid,
        "bone_counts": bone_counts,
        "reverse_flags": reverse_flags,
        "anchor_columns": anchor_columns,
        "raw_us": raw_us,
        "bone_z_axes": bone_z_axes,
    }


def _open_weights(u, v, settings, data, region=0):
    del region
    columns = len(data["bone_counts"])
    column_value = _clamp(u) * (columns - 1)
    column0 = min(int(math.floor(column_value)), columns - 1)
    column1 = min(column0 + 1, columns - 1)
    column_factor = _smoothstep(column_value - column0)
    column_weights = ((column0, 1.0 - column_factor), (column1, column_factor))
    weights = {}
    parent_weight = 0.0
    for column, column_weight in column_weights:
        if column_weight <= 1.0e-10:
            continue
        count = int(data["bone_counts"][column])
        logical_v = 1.0 - _clamp(v) if data["reverse_flags"][column] else _clamp(v)
        local_parent = 0.0
        if logical_v < 0.15:
            local_parent = 1.0 - _smoothstep(logical_v / 0.15)
        parent_weight += column_weight * local_parent
        row_value = _clamp(logical_v * count - 0.5, 0.0, count - 1.0)
        row0 = min(int(math.floor(row_value)), count - 1)
        row1 = min(row0 + 1, count - 1)
        factor = row_value - math.floor(row_value)
        row1_weight = (1.0 - math.cos(math.pi * factor)) * 0.5
        row0_weight = 1.0 - row1_weight
        for row, row_weight in ((row0, row0_weight), (row1, row1_weight)):
            name = _S.bone_name(settings.prefix, 0, column, row, False, False, columns)
            weights[name] = weights.get(name, 0.0) + (
                column_weight * (1.0 - local_parent) * row_weight
            )
    weights = {name: value for name, value in weights.items() if value >= _S.MIN_WEIGHT}
    slots = 3 if parent_weight >= _S.MIN_WEIGHT else 4
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:slots])
    if parent_weight >= _S.MIN_WEIGHT:
        weights[_S.PARENT_KEY] = parent_weight
    total = sum(weights.values())
    if total <= _S.EPS:
        return {}
    return {name: value / total for name, value in weights.items()}


def _create_open_bones(armature, data, settings):
    edit_bones = armature.data.edit_bones
    parent = edit_bones.get(settings.parent_bone)
    if parent is None:
        raise ValueError(f"找不到父骨骼：{settings.parent_bone}")
    collection = _S.get_or_create_bone_collection(armature.data, settings.bone_collection)
    inverse = armature.matrix_world.inverted()
    normal_to_local = armature.matrix_world.to_3x3().transposed()
    created = 0
    for column, physical_chain in enumerate(data["bone_chains"]):
        points = [Vector(point) for point in physical_chain]
        normals = [Vector(normal) for normal in data["bone_z_axes"][column]]
        if data["reverse_flags"][column]:
            points.reverse()
            normals.reverse()
        previous = None
        for row in range(len(points) - 1):
            name = _S.bone_name(
                settings.prefix, 0, column, row, False, False,
                len(data["bone_chains"]),
            )
            if edit_bones.get(name) is not None:
                raise ValueError(f"已存在同名生成骨骼：{name}")
            bone = edit_bones.new(name)
            bone.head = inverse @ points[row]
            bone.tail = inverse @ points[row + 1]
            if (bone.tail - bone.head).length_squared <= _S.EPS:
                raise ValueError(f"控制曲面产生了零长度骨骼：{name}")
            bone.parent = parent if row == 0 else previous
            bone.use_connect = row > 0
            bone.use_deform = True
            if collection is not None:
                collection.assign(bone)
            _S.align_bone_roll(bone, normal_to_local @ normals[row])
            previous = bone
            created += 1
    return created


def _sheet_validate_build(context, settings):
    _S.validate_prefix(settings.prefix)
    primary = _S.primary_from_settings(settings)
    if primary is None:
        raise ValueError("请先在编辑模式选择顶点并生成控制点")
    if int(primary.get(PROP_SCHEMA, 0)) != SCHEMA_VERSION:
        raise ValueError("旧版控制点数据不兼容，请删除控制点后重新生成")
    if str(primary.get(PROP_GENERATOR_TYPE, "")) != 'SURFACE':
        raise ValueError("当前控制点不是非闭合裙摆类型，请重新生成")
    armature = settings.target_armature
    if armature is None or armature.type != 'ARMATURE' or armature.name not in context.view_layer.objects:
        raise ValueError("请选择有效的目标骨架")
    if not settings.parent_bone or armature.data.bones.get(settings.parent_bone) is None:
        raise ValueError("请选择有效的父骨骼")
    if settings.parent_bone.startswith(settings.prefix):
        raise ValueError("父骨骼不能使用生成骨骼前缀")
    if int(settings.control_chain_count) < 4 or int(settings.control_chain_count) > 24:
        raise ValueError("控制点链数必须在 4 到 24 之间")
    if int(settings.chains) < int(settings.control_chain_count):
        raise ValueError("骨骼链数不能少于控制点链数")
    grids, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
    if len(grids) != 1 or len(grids[0][0]) != int(settings.control_chain_count):
        raise ValueError("控制点链数已改变，请重新生成控制点")
    regions = _S.read_regions(primary)
    if len(regions) != 1:
        raise ValueError("旧版分区控制点不兼容，请重新生成控制点")
    return primary, armature, grids, regions


def _sheet_build(context, settings):
    global _ACTIVE_SHEET_DATA
    primary, armature, control_grids, regions = _sheet_validate_build(context, settings)
    control_grid = [
        [obj.matrix_world.translation.copy() for obj in row]
        for row in control_grids[0]
    ]
    control_chains = _sheet_grid_chains(control_grids[0])
    main = context.scene.auto_skirt_settings
    data = rail_data(control_chains, main) if main.workflow_enabled else _build_open_surface_data(control_grid, settings, control_chains)
    bind = not main.workflow_enabled or main.bind_weights
    _ACTIVE_SHEET_DATA = data
    state = _S.ContextState(context)
    created_bones = assignments = 0
    modifier_added = False
    indices_by_mesh = {}
    projector = _S.make_projector(data["preview_grid"]) if bind else None
    weights_by_mesh = {}
    for part in regions[0].get("parts", []) if bind else ():
        mesh_obj = bpy.data.objects.get(str(part.get("mesh", "")))
        if mesh_obj is None or mesh_obj.type != 'MESH':
            raise ValueError("记录的目标网格不存在，请重新生成控制点")
        mesh_weights = weights_by_mesh.setdefault(mesh_obj.name, {})
        mesh_indices = indices_by_mesh.setdefault(mesh_obj.name, set())
        for raw_index in part.get("indices", []):
            index = int(raw_index)
            if not 0 <= index < len(mesh_obj.data.vertices):
                raise ValueError("选区顶点索引已失效，请重新生成控制点")
            u, v = _S.project_uv(_S.vertex_world(mesh_obj, index), projector)
            mesh_weights[index] = _open_weights(u, v, settings, data)
            mesh_indices.add(index)

    try:
        _S.activate(context, armature, mode='EDIT')
        _S.cleanup_bones(armature, settings.prefix)
        created_bones = _create_open_bones(armature, data, settings)
        bpy.ops.object.mode_set(mode='OBJECT')

        old_prefix = str(primary.get(_S.PROP_PREFIX, settings.prefix))
        _S.delete_proxies(old_prefix, primary.name)
        if old_prefix != settings.prefix:
            _S.delete_proxies(settings.prefix, primary.name)

        for mesh_name, mesh_weights in weights_by_mesh.items():
            mesh_obj = bpy.data.objects.get(mesh_name)
            _S.activate(context, mesh_obj, mode='OBJECT')
            assignments += _S.apply_weights(
                mesh_obj,
                indices_by_mesh[mesh_name],
                mesh_weights,
                settings,
            )
            modifier_added |= _S.ensure_modifier(mesh_obj, armature)

        if settings.create_proxy_mesh:
            _S.create_proxy(data["preview_grid"], primary, settings, 0, bind=bind)
        primary[_S.PROP_PREFIX] = settings.prefix
        primary[PROP_REVERSES] = json.dumps(data["reverse_flags"], separators=(",", ":"))
    finally:
        state.restore(context)
    return (
        created_bones,
        sum(len(value) for value in indices_by_mesh.values()),
        assignments,
        modifier_added,
    )


def _sheet_generate(context, settings):
    _S.validate_prefix(settings.prefix)
    settings.group_mode = 'MERGED'
    meshes, region_components = _S.collect_selection(context, settings)
    primary = meshes[0]
    settings.target_mesh = primary
    components = [component for region in region_components for component in region]
    main = context.scene.auto_skirt_settings
    levels = main.control_levels if main.workflow_enabled else 4
    grid, display_size = _fit_open_region_grid(components, settings.control_chain_count, levels)
    fabric_normal = _average([r["normal"] for c in components for r in c["records"]])
    cage_normals = _grid_default_normals(grid, False)
    primary["invert_open_normal"] = sum(n.dot(fabric_normal) for row in cage_normals for n in row) < 0
    collection = _sheet_create_controls(primary, settings, [grid], display_size)

    parts_by_mesh = {}
    for component in components:
        parts_by_mesh.setdefault(component["mesh"], set()).update(component["indices"])
    regions = [{
        "parts": [
            {"mesh": name, "indices": sorted(indices)}
            for name, indices in parts_by_mesh.items()
        ]
    }]
    target_names = [mesh.name for mesh in meshes]
    primary[_S.PROP_COLLECTION] = collection.name
    primary[_S.PROP_TARGETS] = json.dumps(target_names, separators=(",", ":"))
    primary[_S.PROP_REGIONS] = json.dumps(regions, separators=(",", ":"))
    primary[_S.PROP_PREFIX] = settings.prefix
    primary[_S.PROP_MODE] = 'MERGED'
    primary[PROP_SCHEMA] = SCHEMA_VERSION
    primary[PROP_GENERATOR_TYPE] = 'SURFACE'
    primary[PROP_REVERSES] = json.dumps(_sheet_reverses(settings), separators=(",", ":"))
    for mesh_obj in meshes:
        indices = sorted(parts_by_mesh.get(mesh_obj.name, set()))
        mesh_obj[_S.PROP_INDICES] = json.dumps(indices, separators=(",", ":"))
    _S.enter_edit(context, settings, collection)
    return sum(len(component["records"]) for component in components), 1


def _sheet_compute_weights(u, v, settings, region):
    if _ACTIVE_SHEET_DATA is None:
        raise ValueError("曲面权重数据尚未建立")
    return _open_weights(u, v, settings, _ACTIVE_SHEET_DATA, region)


def _sheet_create_bones_compat(armature, region_nodes, settings):
    del region_nodes
    if _ACTIVE_SHEET_DATA is None:
        raise ValueError("曲面骨骼数据尚未建立")
    return _create_open_bones(armature, _ACTIVE_SHEET_DATA, settings)

# ---------------------------------------------------------------------------
# Closed-skirt reversal and editable normals
# ---------------------------------------------------------------------------

_OLD_MAIN_BUILD_DATA = None
_OLD_MAIN_CREATE_BONES = None
_OLD_MAIN_COMPUTE_WEIGHTS = None
_OLD_MAIN_BUILD = None
_OLD_MAIN_INITIAL_PARAMETERS = None
_OLD_MAIN_LIVE_PROXY = None
_OLD_MAIN_DRAW_MARKERS = None
_OLD_SHEET_LIVE_PROXY = None
_OLD_SHEET_DRAW_MARKERS = None


def _main_control_objects(settings, require=True):
    primary = settings.target_mesh
    if primary is None and settings.control_edit_mesh_name:
        primary = bpy.data.objects.get(settings.control_edit_mesh_name)
    if primary is None or primary.type != 'MESH':
        if require:
            raise ValueError("找不到控制点对应的目标网格")
        return None, []
    collection_name = str(primary.get(
        _M.PROP_COLLECTION,
        _M._control_collection_name(settings.prefix, primary.name),
    ))
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        if require:
            raise ValueError("找不到控制点集合，请先生成控制点")
        return primary, []
    controls = _M.find_control_points(collection, settings.prefix, require_all=require)
    if not _M._controls_are_asymmetric(controls):
        if require:
            raise ValueError("旧版控制点不兼容，请重新生成控制点")
        return primary, []
    return primary, controls


def _main_position_map(control_objects):
    return {
        obj.name: _snapshot_value(obj)
        for obj in _M._flatten_control_objects(control_objects)
    }


def _main_restore_positions(control_objects, snapshot_raw):
    data = _M._parse_json_dict(snapshot_raw)
    for obj in _M._flatten_control_objects(control_objects):
        if obj.name in data:
            _restore_snapshot_object(obj, data[obj.name])


def _main_changed_names(current, previous):
    return {
        name for name, value in current.items()
        if name in previous and _snapshot_changed(value, previous[name])
    }


def _nearest_closed_anchor_reverses(raw_us, settings):
    anchor_reverses = _main_reverses(settings)
    count = len(anchor_reverses)
    result = []
    for raw_u in raw_us:
        value = (float(raw_u) % 1.0) * count
        nearest = int(math.floor(value + 0.5)) % count
        result.append(bool(anchor_reverses[nearest]))
    return result


def _parameter_cache_key(surface_points, control_points):
    # Selection, generator mode and K are the only invalidators requested for
    # the live parameterization cache. Positions are rounded to avoid noise.
    return (
        len(control_points),
        tuple(
            tuple(round(float(component), 5) for component in point)
            for point in surface_points
        ),
    )


def _cached_initial_asym_parameters(surface_points, control_points, profile, arc_mapping):
    if not _PREVIEW_MODE:
        return _OLD_MAIN_INITIAL_PARAMETERS(surface_points, control_points, profile, arc_mapping)
    key = _parameter_cache_key(surface_points, control_points)
    cached = _PARAMETER_CACHE.get(key)
    if cached is None:
        computed = _OLD_MAIN_INITIAL_PARAMETERS(surface_points, control_points, profile, arc_mapping)
        raw_values = [
            _M._raw_skirt_u(point, profile["center"], profile["right"], profile["back"])
            for point in surface_points
        ]
        cached = [(raw_u, value[1]) for raw_u, value in zip(raw_values, computed)]
        _PARAMETER_CACHE.clear()
        _PARAMETER_CACHE[key] = cached
    return [
        (_M._arc_s_from_raw_u(arc_mapping, raw_u), local_v)
        for raw_u, local_v in cached
    ]


def _main_build_variable_data(control_points, surface_points, settings, anchor_bone_counts):
    global _ACTIVE_SKIRT_DATA
    if settings.workflow_enabled:
        _, controls = _main_control_objects(settings)
        data = rail_data(controls, settings)
        if settings.bind_weights:
            data["projector"] = _M._build_asymmetric_parameter_projector(data["preview_grid"])
        _ACTIVE_SKIRT_DATA = data
        return data
    data = _OLD_MAIN_BUILD_DATA(control_points, surface_points, settings, anchor_bone_counts)
    primary, controls = _main_control_objects(settings, require=False)
    del primary
    reverse_flags = _nearest_closed_anchor_reverses(data["raw_us"], settings)
    data["reverse_flags"] = reverse_flags
    geometric_z_axes = [
        [Vector(normal) for normal in column]
        for column in data["bone_z_axes"]
    ]
    has_custom_normals = (
        controls
        and len(controls) == len(control_points)
        and any(_control_is_custom(obj) for chain in controls for obj in chain)
    )
    if has_custom_normals:
        current_normals = _control_chain_normals(controls)
        anchor_normals = []
        for anchor, chain in enumerate(controls):
            generated_column = data["anchor_columns"][anchor]
            anchor_normals.append([
                current_normals[anchor][level]
                if _control_is_custom(obj)
                else _M._interpolate_vector_samples(
                    geometric_z_axes[generated_column],
                    level / 3.0,
                )
                for level, obj in enumerate(chain)
            ])
        data["bone_z_axes"] = [
            [
                _interpolate_control_normal(
                    anchor_normals,
                    raw_u,
                    joint / max(int(count), 1),
                    True,
                )
                for joint in range(int(count) + 1)
            ]
            for raw_u, count in zip(data["raw_us"], data["bone_counts"])
        ]
    data["geometric_z_axes"] = geometric_z_axes
    _ACTIVE_SKIRT_DATA = data
    return data


def _main_create_variable_bones(
    armature,
    bone_chains,
    bone_z_axes,
    parent_bone_name,
    prefix,
    bone_collection_name="",
):
    if _ACTIVE_SKIRT_DATA is None or len(_ACTIVE_SKIRT_DATA.get("reverse_flags", [])) != len(bone_chains):
        return _OLD_MAIN_CREATE_BONES(
            armature, bone_chains, bone_z_axes, parent_bone_name, prefix,
            bone_collection_name=bone_collection_name,
        )
    logical_chains = []
    logical_normals = []
    for column, chain in enumerate(bone_chains):
        points = [Vector(point) for point in chain]
        normals = [Vector(normal) for normal in bone_z_axes[column]]
        if _ACTIVE_SKIRT_DATA["reverse_flags"][column]:
            points.reverse()
            normals.reverse()
        logical_chains.append(points)
        logical_normals.append(normals)
    return _OLD_MAIN_CREATE_BONES(
        armature,
        logical_chains,
        logical_normals,
        parent_bone_name,
        prefix,
        bone_collection_name=bone_collection_name,
    )


def _main_variable_weights(arc_s, local_v, bone_counts, prefix, settings, parent_weight):
    if _ACTIVE_SKIRT_DATA is None or len(_ACTIVE_SKIRT_DATA.get("reverse_flags", [])) != len(bone_counts):
        return _OLD_MAIN_COMPUTE_WEIGHTS(
            arc_s, local_v, bone_counts, prefix, settings, parent_weight
        )
    columns = len(bone_counts)
    value = (float(arc_s) % 1.0) * columns
    base = int(math.floor(value))
    column0 = base % columns
    column1 = (column0 + 1) % columns
    factor = _smoothstep(value - base)
    column_weights = ((column0, 1.0 - factor), (column1, factor))
    lengths = _ACTIVE_SKIRT_DATA.get("chain_lengths", [1.0] * columns)
    reference = _M._median_value(lengths)
    blend_distance = max(reference * 0.15, _M.EPSILON)
    weights = {}
    combined_parent = 0.0
    for column, column_weight in column_weights:
        if column_weight <= 1.0e-10:
            continue
        count = int(bone_counts[column])
        logical_v = 1.0 - _clamp(local_v) if _ACTIVE_SKIRT_DATA["reverse_flags"][column] else _clamp(local_v)
        local_distance = logical_v * float(lengths[column])
        local_parent = 0.0
        if local_distance < blend_distance:
            local_parent = 1.0 - _smoothstep(local_distance / blend_distance)
        combined_parent += column_weight * local_parent
        row_value = _clamp(logical_v * count - 0.5, 0.0, count - 1.0)
        row0 = min(int(math.floor(row_value)), count - 1)
        row1 = min(row0 + 1, count - 1)
        row_factor = row_value - math.floor(row_value)
        row_weight0, row_weight1 = _M._vertical_weights(row_factor, settings.weight_curve)
        for row, row_weight in ((row0, row_weight0), (row1, row_weight1)):
            name = _M._plain_bone_name(prefix, column, row)
            weights[name] = weights.get(name, 0.0) + (
                column_weight * (1.0 - local_parent) * row_weight
            )
    weights = {name: value for name, value in weights.items() if value >= _M.MIN_WEIGHT}
    slots = int(settings.max_influences) - (1 if combined_parent >= _M.MIN_WEIGHT else 0)
    weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:max(0, slots)])
    if combined_parent >= _M.MIN_WEIGHT:
        weights[_M.PARENT_WEIGHT_KEY] = combined_parent
    if settings.normalize_weights:
        total = sum(weights.values())
        if total > _M.EPSILON:
            weights = {name: value / total for name, value in weights.items()}
    return weights


def _main_build_wrapper(context, settings, build_bones, build_weights):
    settings.skirt_shape_mode = 'ASYMMETRIC'
    primary, _controls = _main_control_objects(settings, require=True)
    if int(primary.get(PROP_SCHEMA, 0)) != SCHEMA_VERSION:
        raise ValueError("旧版控制点数据不兼容，请删除后重新生成")
    if str(primary.get(PROP_GENERATOR_TYPE, "")) != 'SKIRT':
        raise ValueError("当前控制点不是闭合裙摆类型，请重新生成")
    result = _OLD_MAIN_BUILD(context, settings, build_bones, build_weights)
    if primary is not None:
        primary[PROP_SCHEMA] = SCHEMA_VERSION
        primary[PROP_GENERATOR_TYPE] = 'SKIRT'
        primary[PROP_REVERSES] = json.dumps(_main_reverses(settings), separators=(",", ":"))
    return result


def _normal_line_geometry(chains, normals=None):
    if chains and chains[0] and hasattr(chains[0][0], "matrix_world"):
        grid = [[obj.matrix_world.translation for obj in row] for row in zip(*chains)]
        closed = "auto_skirt_chain_index" in chains[0][0]
        normals = list(zip(*_grid_default_normals(grid, closed, chains[0][0].get("invert_open_normal", False))))
    lines = []
    for column, chain in enumerate(chains):
        for level, item in enumerate(chain):
            if hasattr(item, "matrix_world"):
                point = item.matrix_world.translation.copy()
                normal = (_control_normal(item, _vertical_tangent(chain, level))
                          if _control_is_custom(item) else normals[column][level])
                size = max(float(item.empty_display_size) * 3.5, 0.01)
            else:
                point = Vector(item)
                normal = Vector(normals[column][level]) if normals else Vector((0.0, 0.0, 1.0))
                size = 0.03
            tip = point + _safe_normalized(normal) * size
            side = _safe_normalized(normal.cross(Vector((0.0, 0.0, 1.0))), (1.0, 0.0, 0.0))
            lines.extend((point, tip, tip, tip - normal * size * 0.25 + side * size * 0.10))
            lines.extend((tip, tip - normal * size * 0.25 - side * size * 0.10))
    return lines


def _chain_direction_arrow_geometry(chain, reverse_direction=False):
    """Build a four-fin arrow along the logical root-to-tip control direction."""
    ordered = list(reversed(chain)) if reverse_direction else list(chain)
    if len(ordered) < 2:
        return []
    points = [obj.matrix_world.translation.copy() for obj in ordered]
    lengths = [
        (points[index + 1] - points[index]).length
        for index in range(len(points) - 1)
    ]
    total = sum(lengths)
    if total <= 1.0e-10:
        return []
    # Keep the marker at the physical midpoint so reversing changes only its direction.
    distance = total * 0.5
    accumulated = 0.0
    segment = 0
    for index, length in enumerate(lengths):
        segment = index
        if accumulated + length >= distance or index == len(lengths) - 1:
            break
        accumulated += length
    length = lengths[segment]
    if length <= 1.0e-10:
        return []
    factor = _clamp((distance - accumulated) / length)
    tangent = _safe_normalized(points[segment + 1] - points[segment])
    tip = points[segment].lerp(points[segment + 1], factor)
    display_size = max(
        (float(getattr(obj, "empty_display_size", 0.0)) for obj in ordered),
        default=0.0,
    )
    size = max(0.005, min(total * 0.12, max(display_size * 2.2, total * 0.05)))
    return _arrow_lines(tip, tangent, size)


def generation_arrow_lines(points, bone_count, reverse):
    """Place one arrow on each chord used by the generated bone chain."""
    joints = _S.resample_polyline(points, bone_count + 1)
    if reverse:
        joints.reverse()
    lines = []
    for head, tail in zip(joints, joints[1:]):
        delta = tail - head
        length = delta.length
        if length > 1.0e-10:
            lines.extend(_arrow_lines(head.lerp(tail, 0.66), delta / length, length * 0.26))
    return lines


def _arrow_lines(tip, tangent, size):
    reference = Vector((0.0, 0.0, 1.0))
    if abs(tangent.dot(reference)) > 0.9:
        reference = Vector((1.0, 0.0, 0.0))
    side_a = _safe_normalized(tangent.cross(reference), (0.0, 1.0, 0.0))
    side_b = _safe_normalized(tangent.cross(side_a), (1.0, 0.0, 0.0))
    back = tip - tangent * size
    shaft = tip - tangent * size * 1.65
    wing = size * 0.46
    return [
        shaft, tip,
        tip, back + side_a * wing,
        tip, back - side_a * wing,
        tip, back + side_b * wing,
        tip, back - side_b * wing,
    ]


def _draw_chain_direction_arrows(chains, reverse_flags):
    if not chains:
        return
    reverse_flags = list(reverse_flags)
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        try:
            gpu.state.line_width_set(2.8)
        except (AttributeError, RuntimeError):
            pass
        shader.bind()
        for column, chain in enumerate(chains):
            reverse_direction = bool(reverse_flags[column]) if column < len(reverse_flags) else False
            lines = _chain_direction_arrow_geometry(chain, reverse_direction)
            if not lines:
                continue
            shader.uniform_float("color", display_color(column))
            batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
    finally:
        try:
            gpu.state.line_width_set(1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.blend_set('NONE')
        except (AttributeError, RuntimeError):
            pass


def _draw_normal_lines(lines, color=(0.32, 1.0, 0.78, 1.0)):
    if not lines:
        return
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.bind()
        shader.uniform_float("color", color)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        try:
            gpu.state.line_width_set(2.0)
        except (AttributeError, RuntimeError):
            pass
        batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
    finally:
        try:
            gpu.state.line_width_set(1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.blend_set('NONE')
        except (AttributeError, RuntimeError):
            pass

_CHAIN_RESAMPLE_GUARD = False
_SHEET_MIRROR_GUARD = False


def _resample_open_values(values, new_count, discrete=False):
    values = list(values)
    if not values:
        return [False if discrete else 6] * new_count
    if new_count == 1:
        return [values[0]]
    result = []
    for index in range(new_count):
        value = index * (len(values) - 1) / (new_count - 1)
        if discrete:
            result.append(values[int(_clamp(round(value), 0, len(values) - 1))])
        else:
            first = min(int(math.floor(value)), len(values) - 2)
            factor = value - first
            result.append(int(round(values[first] * (1.0 - factor) + values[first + 1] * factor)))
    return result


def _update_control_chain_count(settings, context):
    global _CHAIN_RESAMPLE_GUARD
    if _CHAIN_RESAMPLE_GUARD or context is None or context.scene is None:
        return
    new_count = int(_clamp(settings.asym_control_chain_count, 4, 24))
    if _uses_open_surface(settings):
        sheet = getattr(context.scene, "auto_sheet_settings", None)
        if sheet is None:
            return
        primary = _S.primary_from_settings(sheet)
        grids = []
        if primary is not None and int(primary.get(PROP_SCHEMA, 0)) == SCHEMA_VERSION:
            grids, _collection = _sheet_find_controls(primary, sheet.prefix, require=False)
        old_count = len(grids[0][0]) if grids and grids[0] else int(sheet.control_chain_count)
        old_count = int(_clamp(old_count, 4, 24))
        old_counts = _main_counts(settings, old_count)
        old_reverses = _main_reverses(settings, old_count)
        new_counts = _resample_open_values(old_counts, new_count)
        new_reverses = _resample_open_values(old_reverses, new_count, discrete=True)
        _CHAIN_RESAMPLE_GUARD = True
        try:
            sheet.control_chain_count = new_count
            settings.columns = max(int(settings.columns), new_count)
            _set_main_counts(settings, new_counts, new_count)
            _set_main_reverses(settings, new_reverses, new_count)
            _set_sheet_counts(sheet, new_counts, new_count)
            _set_sheet_reverses(sheet, new_reverses, new_count)
            if old_count == new_count or not grids:
                _invalidate_preview('SURFACE')
                return
            grid = grids[0]
            old_chains = _sheet_grid_chains(grid)
            old_normals = _control_chain_normals(old_chains)
            had_custom_normals = any(
                _control_is_custom(obj) for chain in old_chains for obj in chain
            )
            points = [[obj.matrix_world.translation.copy() for obj in row] for row in grid]
            new_grid = [
                [_interpolate_open_curve(points[row], index / (new_count - 1)) for index in range(new_count)]
                for row in range(_S.LEVELS)
            ]
            display_size = max((obj.empty_display_size for obj in _S.flatten_controls(grids)), default=0.05)
            collection = _sheet_create_controls(primary, sheet, [new_grid], display_size)
            new_grids, _ = _sheet_find_controls(primary, sheet.prefix, require=True)
            new_chains = _sheet_grid_chains(new_grids[0])
            for column, chain in enumerate(new_chains):
                horizontal = column / max(new_count - 1, 1)
                for level, obj in enumerate(chain):
                    normal = _interpolate_control_normal(old_normals, horizontal, level / 3.0, False)
                    _set_control_normal(
                        obj,
                        _vertical_tangent(chain, level),
                        normal,
                        mark_auto=not had_custom_normals,
                    )
            primary[_S.PROP_COLLECTION] = collection.name
            primary[PROP_SCHEMA] = SCHEMA_VERSION
            if sheet.edit_active:
                controls = _S.flatten_controls(new_grids)
                for obj in controls:
                    _S.set_locked(obj, False)
                    obj.select_set(True)
                if controls:
                    context.view_layer.objects.active = controls[0]
                sheet.edit_snapshot = json.dumps(_sheet_position_map(new_grids), separators=(",", ":"))
                sheet.edit_last_positions = sheet.edit_snapshot
            _invalidate_preview('SURFACE')
        finally:
            _CHAIN_RESAMPLE_GUARD = False
        return

    settings.skirt_shape_mode = 'ASYMMETRIC'
    primary, controls = _main_control_objects(settings, require=False)
    if not controls or int(primary.get(PROP_SCHEMA, 0)) != SCHEMA_VERSION:
        return
    old_count = len(controls)
    if old_count == new_count:
        return
    _CHAIN_RESAMPLE_GUARD = True
    try:
        old_points = _M._control_world_locations(controls)
        old_normals = _control_chain_normals(controls)
        had_custom_normals = any(
            _control_is_custom(obj) for chain in controls for obj in chain
        )
        new_points = _M._resample_asym_control_points(old_points, new_count)
        old_counts = _main_counts(settings, old_count)
        old_reverses = _main_reverses(settings, old_count)
        new_counts = _M._resample_cyclic_counts(old_counts, new_count, settings.asym_default_bone_count)
        new_reverses = []
        for chain in range(new_count):
            old_value = chain * old_count / new_count
            new_reverses.append(old_reverses[int(math.floor(old_value + 0.5)) % old_count])
        display_size = max((obj.empty_display_size for obj in _M._flatten_control_objects(controls)), default=0.05)
        for obj in list(_M._flatten_control_objects(controls)):
            bpy.data.objects.remove(obj, do_unlink=True)
        _set_main_counts(settings, new_counts, new_count)
        _set_main_reverses(settings, new_reverses, new_count)
        settings.columns = max(int(settings.columns), new_count)
        collection = _M._create_or_update_asymmetric_controls(
            primary, settings, new_points, display_size, update_existing=False
        )
        new_controls = _M.find_control_points(collection, settings.prefix, require_all=True)
        for column, chain in enumerate(new_controls):
            horizontal = column / new_count
            for level, obj in enumerate(chain):
                normal = _interpolate_control_normal(old_normals, horizontal, level / 3.0, True)
                _set_control_normal(
                    obj,
                    _vertical_tangent(chain, level),
                    normal,
                    mark_auto=not had_custom_normals,
                )
        primary[PROP_SCHEMA] = SCHEMA_VERSION
        primary[PROP_GENERATOR_TYPE] = 'SKIRT'
        if settings.control_edit_active:
            _M._set_controls_locked(new_controls, False)
            _M._select_only_controls(context, new_controls)
            settings.control_edit_snapshot = json.dumps(_main_position_map(new_controls), separators=(",", ":"))
            settings.control_edit_last_positions = settings.control_edit_snapshot
        _invalidate_preview('SKIRT')
    finally:
        _CHAIN_RESAMPLE_GUARD = False


def _mirror_pairs_from_points(points, axis):
    """Pair reflected controls while allowing any number of on-axis self pairs."""
    if not points:
        return [], 0.0
    center = _average(points.values())
    center_value = center.x if axis == 'X' else center.y
    scale = max(((point - center).length for point in points.values()), default=1.0)
    tolerance = max(scale, 1.0) * 1.0e-5
    tolerance_squared = tolerance * tolerance
    unpaired = set(points)
    pairs = []

    # A chain already on the mirror plane must map to itself.  The previous
    # parity-based rule failed for cardinal four-chain cages, which have two
    # on-axis chains for each mirror plane.
    for index in sorted(points):
        if index not in unpaired:
            continue
        reflected = _M._mirror_location(points[index], axis, center_value)
        self_error = (points[index] - reflected).length_squared
        other_errors = [
            (points[other] - reflected).length_squared
            for other in unpaired
            if other != index
        ]
        nearest_other = min(other_errors) if other_errors else float('inf')
        if self_error <= nearest_other + tolerance_squared:
            pairs.append((index, index))
            unpaired.remove(index)

    # A noisy/asymmetric odd set can still leave one control.  Choose the
    # control closest to the plane rather than forcing two unrelated chains
    # to collapse onto one another.
    if len(unpaired) % 2:
        middle = min(
            unpaired,
            key=lambda index: (
                points[index] - _M._mirror_location(points[index], axis, center_value)
            ).length_squared,
        )
        pairs.append((middle, middle))
        unpaired.remove(middle)

    # Pair the remaining points by the globally best symmetric reflection
    # error instead of depending on control-chain numbering order.
    while unpaired:
        candidates = []
        ordered = sorted(unpaired)
        for offset, first in enumerate(ordered[:-1]):
            reflected_first = _M._mirror_location(points[first], axis, center_value)
            for second in ordered[offset + 1:]:
                reflected_second = _M._mirror_location(points[second], axis, center_value)
                error = (
                    (points[second] - reflected_first).length_squared
                    + (points[first] - reflected_second).length_squared
                )
                candidates.append((error, first, second))
        _error, first, second = min(candidates)
        pairs.append((first, second))
        unpaired.remove(first)
        unpaired.remove(second)
    return pairs, center_value


def _fixed_dynamic_mirror_pairs(control_objects, snapshot_raw, level, axis):
    snapshot = _M._parse_json_dict(snapshot_raw)
    points = {}
    for chain, objects in enumerate(control_objects):
        point = _M._vector_from_json(snapshot.get(objects[level].name))
        if point is not None:
            points[chain] = point
    return _mirror_pairs_from_points(points, axis)


def _pair_lookup(pairs):
    result = {}
    for first, second in pairs:
        result[first] = second
        result[second] = first
    return result


def _symmetrize_self_mirrored_normal(normal, axis):
    normal = _safe_normalized(normal)
    combined = normal + _mirror_vector(normal, axis)
    return _safe_normalized(combined, normal) if combined.length_squared > 1.0e-10 else normal


def _apply_atomic_chain_mirror(chains, level, changed_names, active_name, snapshot_points, axes, reference=None):
    """Apply X/Y from original user sources without reusing generated targets as sources."""
    axes = tuple(axis for axis in axes if axis in {'X', 'Y'})
    if not axes or not snapshot_points:
        return set()
    maps = {}
    centers = {}
    for axis in axes:
        pairs, _ = _mirror_pairs_from_points(reference or snapshot_points, axis)
        center = _average(snapshot_points.values())
        centers[axis] = center.x if axis == "X" else center.y
        maps[axis] = _pair_lookup(pairs)
    changed_indices = {
        index for index, chain in enumerate(chains)
        if chain[level].name in changed_names
    }
    if not changed_indices:
        return set()
    active_index = next(
        (index for index in changed_indices if chains[index][level].name == active_name),
        None,
    )
    remaining = set(changed_indices)
    modified = set()
    first_orbit = True
    while remaining:
        source_index = active_index if first_orbit and active_index in remaining else min(remaining)
        first_orbit = False
        source = chains[source_index][level]
        base = source.location.copy()
        source_normal = _control_normal(source, _vertical_tangent(chains[source_index], level))
        self_axes = [axis for axis in axes if maps[axis].get(source_index) == source_index]
        for axis in self_axes:
            if axis == 'X':
                base.x = centers[axis]
            else:
                base.y = centers[axis]
            source_normal = _symmetrize_self_mirrored_normal(source_normal, axis)

        variants = [()]
        for axis in axes:
            variants += [variant + (axis,) for variant in list(variants)]
        plans = {}
        for variant in variants:
            target_index = source_index
            point = base.copy()
            normal = source_normal.copy()
            for axis in variant:
                target_index = maps[axis].get(target_index, target_index)
                point = _M._mirror_location(point, axis, centers[axis])
                normal = _mirror_vector(normal, axis)
            plans.setdefault(target_index, (variant, point, normal))

        # Write all positions before deriving target tangents for orientation.
        for target_index, (_variant, point, _normal) in plans.items():
            chains[target_index][level].location = point
        source_custom = _control_is_custom(source)
        for target_index, (variant, _point, normal) in plans.items():
            target = chains[target_index][level]
            if target_index != source_index or variant or self_axes:
                _set_control_normal(
                    target,
                    _vertical_tangent(chains[target_index], level),
                    normal,
                    mark_auto=not source_custom,
                )
            modified.add(target.name)
        remaining.difference_update(plans)
    return modified


def _main_apply_control_mirror(context, scene=None):
    if _M._MIRROR_GUARD:
        return
    scene = scene or context.scene
    settings = getattr(scene, "auto_skirt_settings", None)
    if (
        settings is None or not settings.control_edit_active
        or (not settings.control_mirror_x and not settings.control_mirror_y)
    ):
        return
    try:
        _mesh, _collection, chains = _M._control_edit_collection(context, settings)
    except (ValueError, RuntimeError):
        return
    if not _M._controls_are_asymmetric(chains):
        return
    current = _main_position_map(chains)
    previous = _M._parse_json_dict(settings.control_edit_last_positions)
    changed = _main_changed_names(current, previous)
    if not changed:
        return
    snapshot = _M._parse_json_dict(settings.control_edit_snapshot)
    active_name = _M._preferred_changed_control_name(
        context, chains, changed, previous
    )
    axes = tuple(
        axis for axis, enabled in (
            ('X', settings.control_mirror_x),
            ('Y', settings.control_mirror_y),
        ) if enabled
    )
    _M._MIRROR_GUARD = True
    try:
        reference = mirror_reference(chains, snapshot)
        for level in range(len(chains[0])):
            points = {
                index: Vector(snapshot[chain[level].name][:3])
                for index, chain in enumerate(chains)
                if chain[level].name in snapshot
            }
            _apply_atomic_chain_mirror(
                chains, level, changed, active_name, points, axes, reference,
            )
        settings.control_edit_last_positions = json.dumps(
            _main_position_map(chains), separators=(",", ":")
        )
        _invalidate_preview('SKIRT')
    finally:
        _M._MIRROR_GUARD = False


def _main_dynamic_mirror(
    control_objects,
    current,
    changed,
    active_name,
    level,
    axis,
    snapshot_raw,
):
    mirrored = set()
    pairs, center_value = _fixed_dynamic_mirror_pairs(control_objects, snapshot_raw, level, axis)
    for first, second in pairs:
        obj_a = control_objects[first][level]
        obj_b = control_objects[second][level]
        if first == second:
            if obj_a.name not in changed:
                continue
            point = _M._mirror_location(obj_a.location, axis, center_value).lerp(obj_a.location, 0.5)
            if axis == 'X':
                point.x = center_value
            else:
                point.y = center_value
            obj_a.location = point
            _mirror_control_orientation(
                obj_a, control_objects[first], level,
                obj_a, control_objects[first], level, axis,
            )
            current[obj_a.name] = _snapshot_value(obj_a)
            mirrored.add(obj_a.name)
            continue
        source_name, target_name = _M._choose_mirror_source(
            obj_a.name, obj_b.name, changed, active_name
        )
        if source_name is None:
            continue
        source = bpy.data.objects.get(source_name)
        target = bpy.data.objects.get(target_name)
        if source is None or target is None:
            continue
        target.location = _M._mirror_location(source.location, axis, center_value)
        source_chain_index = first if source == obj_a else second
        target_chain_index = first if target == obj_a else second
        _mirror_control_orientation(
            source, control_objects[source_chain_index], level,
            target, control_objects[target_chain_index], level, axis,
        )
        current[target.name] = _snapshot_value(target)
        mirrored.add(target.name)
    return mirrored


def _surface_mirror(context):
    global _SHEET_MIRROR_GUARD
    settings = getattr(context.scene, "auto_sheet_settings", None)
    if (
        _SHEET_MIRROR_GUARD or settings is None or not settings.edit_active
        or (not settings.control_mirror_x and not settings.control_mirror_y)
    ):
        return
    primary = _S.primary_from_settings(settings)
    try:
        grids, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
    except ValueError:
        return
    current = _sheet_position_map(grids)
    previous = _S.parse_dict(settings.edit_last_positions)
    changed = {
        name for name, value in current.items()
        if name in previous and _snapshot_changed(value, previous[name])
    }
    if not changed:
        return
    snapshot = _S.parse_dict(settings.edit_snapshot)
    active_name = context.active_object.name if context.active_object else ""
    axes = tuple(
        axis for axis, enabled in (
            ('X', settings.control_mirror_x),
            ('Y', settings.control_mirror_y),
        ) if enabled
    )
    _SHEET_MIRROR_GUARD = True
    try:
        for grid in grids:
            chains = _sheet_grid_chains(grid)
            reference = mirror_reference(chains, snapshot)
            for level in range(len(chains[0])):
                points = {
                    index: Vector(snapshot[chain[level].name][:3])
                    for index, chain in enumerate(chains)
                    if chain[level].name in snapshot
                }
                _apply_atomic_chain_mirror(
                    chains, level, changed, active_name, points, axes, reference,
                )
        settings.edit_last_positions = json.dumps(
            _sheet_position_map(grids), separators=(",", ":")
        )
        _invalidate_preview('SURFACE')
    finally:
        _SHEET_MIRROR_GUARD = False


# ---------------------------------------------------------------------------
# Two-tier live preview and viewport normal overlays
# ---------------------------------------------------------------------------


def _invalidate_preview(kind=None):
    kinds = (kind,) if kind else tuple(_PREVIEW_CACHE)
    for name in kinds:
        cache = _PREVIEW_CACHE[name]
        cache["input_signature"] = None
        cache["quick"] = ([], [])
        cache["full"] = ([], [])
        cache["normal_lines"] = []


def _control_signature(objects, extra=()):
    values = list(extra)
    for obj in objects:
        values.extend(round(float(value), 5) for value in _snapshot_value(obj))
    return tuple(values)


def _main_preview_signature(settings, controls):
    return _control_signature(
        _M._flatten_control_objects(controls),
        (
            settings.prefix,
            int(settings.asym_control_chain_count),
            int(settings.columns),
            tuple(_main_counts(settings)),
            tuple(_main_reverses(settings)),
            bool(settings.smooth_bone_chains),
        ),
    )


def _surface_preview_signature(settings, grids):
    return _control_signature(
        _S.flatten_controls(grids),
        (
            settings.prefix,
            int(settings.control_chain_count),
            int(settings.chains),
            tuple(_sheet_counts(settings)),
            tuple(_sheet_reverses(settings)),
        ),
    )


def _main_quick_geometry(settings, controls):
    points = _M._control_world_locations(controls)
    columns = min(max(len(points), 12), int(settings.columns))
    samples = [
        [
            _M._variable_asym_surface_point(points, column / columns, row / 3.0)
            for column in range(columns)
        ]
        for row in range(4)
    ]
    return _M._proxy_draw_geometry(samples, True)


def _surface_quick_geometry(settings, grids):
    triangles = []
    lines = []
    columns = min(max(int(settings.control_chain_count), 12), int(settings.chains))
    for grid_objects in grids:
        grid = [[obj.matrix_world.translation.copy() for obj in row] for row in grid_objects]
        samples = [
            [
                _S.evaluate_patch(grid, column / max(columns - 1, 1), row / 3.0)
                for column in range(columns)
            ]
            for row in range(4)
        ]
        local_triangles, local_lines = _M._proxy_draw_geometry(samples, False)
        triangles.extend(local_triangles)
        lines.extend(local_lines)
    return triangles, lines


def _bone_normal_lines(data):
    lines = []
    for column, chain in enumerate(data.get("bone_chains", [])):
        axes = data.get("bone_z_axes", [])[column]
        lengths = [
            (Vector(chain[index + 1]) - Vector(chain[index])).length
            for index in range(len(chain) - 1)
        ]
        size = max(sum(lengths) / max(len(lengths), 1) * 0.35, 0.005)
        for joint, point in enumerate(chain):
            normal = _safe_normalized(axes[min(joint, len(axes) - 1)])
            start = Vector(point)
            tip = start + normal * size
            lines.extend((start, tip))
    return lines


def _tag_view3d_redraw():
    context = bpy.context
    for window in context.window_manager.windows if context.window_manager else []:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _schedule_idle_preview(kind):
    cache = _PREVIEW_CACHE[kind]
    if cache["timer"]:
        return
    cache["timer"] = True

    def refresh():
        cache["timer"] = False
        scene = bpy.context.scene
        if scene is None:
            return None
        active = (
            getattr(scene, "auto_skirt_settings", None).control_edit_active
            if kind == 'SKIRT' and hasattr(scene, "auto_skirt_settings")
            else getattr(scene, "auto_sheet_settings", None).edit_active
            if kind == 'SURFACE' and hasattr(scene, "auto_sheet_settings")
            else False
        )
        if not active:
            return None
        remaining = 0.205 - (time.monotonic() - cache["changed_at"])
        if remaining > 0.0:
            cache["timer"] = True
            return max(remaining, 0.02)
        try:
            if kind == 'SKIRT':
                _main_full_preview()
            else:
                _surface_full_preview()
            _tag_view3d_redraw()
        except (ValueError, RuntimeError, ReferenceError):
            pass
        return None

    try:
        bpy.app.timers.register(refresh, first_interval=0.205)
    except (RuntimeError, ValueError):
        cache["timer"] = False


def _main_full_preview():
    global _PREVIEW_MODE
    cache = _PREVIEW_CACHE['SKIRT']
    now = time.monotonic()
    if now - cache["full_at"] < 0.25:
        return cache["full"]
    _M._LIVE_PROXY_CACHE["signature"] = None
    _PREVIEW_MODE = True
    try:
        geometry = _OLD_MAIN_LIVE_PROXY(bpy.context.scene)
    finally:
        _PREVIEW_MODE = False
    cache["full"] = geometry
    cache["full_at"] = now
    if _ACTIVE_SKIRT_DATA is not None:
        cache["normal_lines"] = _bone_normal_lines(_ACTIVE_SKIRT_DATA)
    return geometry


def _surface_full_preview():
    global _ACTIVE_SHEET_DATA
    cache = _PREVIEW_CACHE['SURFACE']
    now = time.monotonic()
    if now - cache["full_at"] < 0.25:
        return cache["full"]
    settings = getattr(bpy.context.scene, "auto_sheet_settings", None)
    if settings is None or not settings.edit_active:
        return [], []
    primary = _S.primary_from_settings(settings)
    grids, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
    triangles = []
    lines = []
    normal_lines = []
    for grid_objects in grids:
        grid = [[obj.matrix_world.translation.copy() for obj in row] for row in grid_objects]
        data = _build_open_surface_data(grid, settings, _sheet_grid_chains(grid_objects))
        _ACTIVE_SHEET_DATA = data
        vertices, faces, _uv = _S.proxy_topology(data["preview_grid"], triangulated=False)
        edges = set()
        for face in faces:
            for offset in range(1, len(face) - 1):
                triangles.extend((vertices[face[0]], vertices[face[offset]], vertices[face[offset + 1]]))
            for offset, first in enumerate(face):
                second = face[(offset + 1) % len(face)]
                edges.add(tuple(sorted((int(first), int(second)))))
        for first, second in edges:
            lines.extend((vertices[first], vertices[second]))
        normal_lines.extend(_bone_normal_lines(data))
    cache["full"] = (triangles, lines)
    cache["normal_lines"] = normal_lines
    cache["full_at"] = now
    return cache["full"]


def _main_live_proxy(scene):
    settings = getattr(scene, "auto_skirt_settings", None)
    cache = _PREVIEW_CACHE['SKIRT']
    if settings is None or not settings.control_edit_active or not settings.control_live_proxy_preview:
        _invalidate_preview('SKIRT')
        return [], []
    _primary, controls = _main_control_objects(settings, require=True)
    if settings.workflow_enabled:
        return rail_preview(controls, settings)
    signature = _main_preview_signature(settings, controls)
    now = time.monotonic()
    if signature != cache["input_signature"]:
        cache["input_signature"] = signature
        cache["changed_at"] = now
        cache["normal_lines"] = []
        cache["quick"] = _main_quick_geometry(settings, controls)
        _schedule_idle_preview('SKIRT')
        return cache["quick"]
    if now - cache["changed_at"] < 0.2:
        return cache["quick"]
    if cache["full"] == ([], []) or cache["full_at"] < cache["changed_at"]:
        return _main_full_preview()
    return cache["full"]


def _surface_live_proxy():
    settings = getattr(bpy.context.scene, "auto_sheet_settings", None)
    cache = _PREVIEW_CACHE['SURFACE']
    if settings is None or not settings.edit_active or not settings.live_proxy_preview:
        _invalidate_preview('SURFACE')
        return [], []
    primary = _S.primary_from_settings(settings)
    grids, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
    main = bpy.context.scene.auto_skirt_settings
    if main.workflow_enabled:
        return rail_preview(_sheet_grid_chains(grids[0]), main)
    signature = _surface_preview_signature(settings, grids)
    now = time.monotonic()
    if signature != cache["input_signature"]:
        cache["input_signature"] = signature
        cache["changed_at"] = now
        cache["normal_lines"] = []
        cache["quick"] = _surface_quick_geometry(settings, grids)
        _schedule_idle_preview('SURFACE')
        return cache["quick"]
    if now - cache["changed_at"] < 0.2:
        return cache["quick"]
    if cache["full"] == ([], []) or cache["full_at"] < cache["changed_at"]:
        return _surface_full_preview()
    return cache["full"]


def _sheet_marker_positions():
    settings = getattr(bpy.context.scene, "auto_sheet_settings", None)
    if settings is None or not settings.edit_active:
        return [], []
    primary = _S.primary_from_settings(settings)
    if primary is None:
        return [], []
    grids, _collection = _sheet_find_controls(primary, settings.prefix, require=False)
    control_chains = [
        [obj.matrix_world.translation.copy() for obj in chain if obj is not None]
        for grid in grids
        for chain in _sheet_grid_chains(grid)
    ]
    if not bpy.context.scene.auto_skirt_settings.show_mesh_points:
        return control_chains, []
    selected = []
    for mesh_obj in _S.stored_targets(primary):
        try:
            indices = json.loads(mesh_obj.get(_S.PROP_INDICES, "[]"))
        except (TypeError, json.JSONDecodeError):
            indices = []
        selected.extend(
            mesh_obj.matrix_world @ mesh_obj.data.vertices[int(index)].co
            for index in indices
            if 0 <= int(index) < len(mesh_obj.data.vertices)
        )
    return control_chains, selected


def generation_preview(scene):
    """Read the recorded mesh selection and the locked control cage for stage three."""
    global _display_active
    main = scene.auto_skirt_settings
    primary = main.target_mesh
    if primary is None:
        raise ValueError("找不到裙摆网格")
    if main.closed_skirt:
        _, chains = _main_control_objects(main)
        targets = _M._stored_target_meshes(primary)
        indices_key = _M.PROP_INDICES
    else:
        grids, _ = _sheet_find_controls(primary, main.prefix)
        chains = _sheet_grid_chains(grids[0])
        targets = _S.stored_targets(primary)
        indices_key = _S.PROP_INDICES
    _display_active = set(range(len(chains)))
    positions = []
    for mesh in targets:
        matrix = mesh.matrix_world
        verts = mesh.data.vertices
        indices = json.loads(mesh.get(indices_key, "[]"))
        positions.extend(matrix @ verts[index].co for index in indices if 0 <= index < len(verts))
    lines = {f"C{i:02d}": [obj.matrix_world.translation.copy() for obj in chain]
             for i, chain in enumerate(chains)}
    return lines, positions, rail_preview(chains, main)


def _main_draw_markers_with_normals():
    update_display_colors(bpy.context, True)
    _OLD_MAIN_DRAW_MARKERS()
    settings = getattr(bpy.context.scene, "auto_skirt_settings", None)
    if settings is None or not settings.control_edit_active:
        return
    try:
        _primary, controls = _main_control_objects(settings, require=True)
        _draw_chain_direction_arrows(controls, _main_reverses(settings))
        if settings.show_control_normals:
            _draw_normal_lines(_normal_line_geometry(controls))
            if settings.control_live_proxy_preview:
                _draw_normal_lines(_PREVIEW_CACHE['SKIRT']["normal_lines"], (1.0, 0.86, 0.28, 0.9))
    except (ValueError, RuntimeError, ReferenceError):
        pass


def _sheet_draw_markers_with_normals():
    update_display_colors(bpy.context, False)
    _OLD_SHEET_DRAW_MARKERS()
    settings = getattr(bpy.context.scene, "auto_sheet_settings", None)
    if settings is None or not settings.edit_active:
        return
    try:
        primary = _S.primary_from_settings(settings)
        grids, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
        reverse_flags = _sheet_reverses(settings)
        for grid in grids:
            chains = _sheet_grid_chains(grid)
            _draw_chain_direction_arrows(chains, reverse_flags)
            if settings.show_control_normals:
                _draw_normal_lines(_normal_line_geometry(chains))
        if settings.show_control_normals and settings.live_proxy_preview:
            _draw_normal_lines(_PREVIEW_CACHE['SURFACE']["normal_lines"], (1.0, 0.86, 0.28, 0.9))
    except (ValueError, RuntimeError, ReferenceError):
        pass


# ---------------------------------------------------------------------------
# Unified operators and panel dispatch
# ---------------------------------------------------------------------------


def _sync_common_to_sheet(context):
    """Copy the single public settings model into the legacy open-surface core."""
    main = context.scene.auto_skirt_settings
    sheet = context.scene.auto_sheet_settings
    chain_count = int(_clamp(main.asym_control_chain_count, 4, 24))
    sheet.target_armature = main.target_armature
    sheet.parent_bone = main.parent_bone
    sheet.bone_collection = main.bone_collection
    sheet.prefix = main.prefix
    sheet.create_proxy_mesh = main.create_proxy_mesh
    sheet.group_mode = 'MERGED'
    sheet.control_chain_count = chain_count
    sheet.chains = max(int(main.columns), chain_count)
    sheet.default_bone_count = int(main.asym_default_bone_count)
    _set_sheet_counts(sheet, _main_counts(main, chain_count), chain_count)
    _set_sheet_reverses(sheet, _main_reverses(main, chain_count), chain_count)
    if main.target_mesh is not None:
        sheet.target_mesh = main.target_mesh
    return sheet


def _current_control_chains(context):
    main = context.scene.auto_skirt_settings
    if _uses_open_surface(main):
        sheet = _sync_common_to_sheet(context)
        primary = _S.primary_from_settings(sheet)
        grids, _collection = _sheet_find_controls(primary, sheet.prefix, require=True)
        return 'SURFACE', _sheet_grid_chains(grids[0]), main
    _primary, controls = _main_control_objects(main, require=True)
    return 'SKIRT', controls, main


def _persist_chain_options(context):
    main = context.scene.auto_skirt_settings
    if _uses_open_surface(main):
        sheet = _sync_common_to_sheet(context)
        primary = _S.primary_from_settings(sheet)
        if primary is not None:
            primary[PROP_REVERSES] = json.dumps(_main_reverses(main), separators=(",", ":"))
        _invalidate_preview('SURFACE')
    else:
        primary = main.target_mesh
        if primary is not None:
            primary[_M.PROP_ASYM_BONE_COUNTS] = json.dumps(_main_counts(main), separators=(",", ":"))
            primary[PROP_REVERSES] = json.dumps(_main_reverses(main), separators=(",", ":"))
        _invalidate_preview('SKIRT')


def _set_all_reverses(context, value):
    main = context.scene.auto_skirt_settings
    _set_main_reverses(main, [value] * main.asym_control_chain_count)
    _persist_chain_options(context)


def _auto_chain_counts(context):
    kind, chains, settings = _current_control_chains(context)
    lengths = []
    for chain in chains:
        points = [obj.matrix_world.translation.copy() for obj in chain]
        dense = [
            _M.interpolate_vertical_control(points, step / 64.0)
            for step in range(65)
        ]
        lengths.append(sum((dense[index + 1] - dense[index]).length for index in range(64)))
    average = sum(lengths) / max(len(lengths), 1)
    default = int(settings.asym_default_bone_count)
    values = [
        int(_clamp(round(default * length / average), 1, 64))
        if average > 1.0e-10 else default
        for length in lengths
    ]
    _set_main_counts(settings, values)
    _persist_chain_options(context)
    return values


def _reset_control_normals(context, mode):
    kind, chains, settings = _current_control_chains(context)
    if mode == 'INVERT':
        for chain in chains:
            for level, obj in enumerate(chain):
                normal = -_control_normal(obj, _vertical_tangent(chain, level))
                _set_control_normal(obj, _vertical_tangent(chain, level), normal)
    elif mode == 'CLEAR':
        _initialize_chain_orientations(chains, kind == 'SKIRT')
    else:
        if kind == 'SKIRT':
            _main_full_preview()
            data = _ACTIVE_SKIRT_DATA
            proxy_normals = _M._proxy_grid_vertex_normals(data["preview_grid"], True)
            desired = [[-normal for normal in row] for row in proxy_normals]
        else:
            _surface_full_preview()
            data = _ACTIVE_SHEET_DATA
            proxy_normals = _S.proxy_normals(data["preview_grid"])
            desired = [[-normal for normal in row] for row in proxy_normals]
        for anchor, chain in enumerate(chains):
            column = data["anchor_columns"][anchor]
            for level, obj in enumerate(chain):
                row_value = level / 3.0 * (len(desired) - 1)
                first = min(int(math.floor(row_value)), len(desired) - 1)
                second = min(first + 1, len(desired) - 1)
                normal = _slerp_normal(
                    desired[first][column],
                    desired[second][column],
                    row_value - first,
                )
                _set_control_normal(
                    obj,
                    _vertical_tangent(chain, level),
                    normal,
                    mark_auto=True,
                )
    _invalidate_preview(kind)
    if kind == 'SURFACE':
        primary = _S.primary_from_settings(settings)
        grids, _collection = _sheet_find_controls(primary, settings.prefix, require=True)
        settings.edit_last_positions = json.dumps(_sheet_position_map(grids), separators=(",", ":"))
    else:
        settings.control_edit_last_positions = json.dumps(_main_position_map(chains), separators=(",", ":"))
    _tag_view3d_redraw()


class AUTOUNIFIED_OT_reverse_all(bpy.types.Operator):
    bl_idname = "auto_skirt.reverse_all_chains"
    bl_label = "全部反转骨骼方向"
    bl_description = "将所有链的骨骼头尾、父子顺序、编号和权重方向同步反转"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _set_all_reverses(context, True)
        self.report({'INFO'}, "所有骨骼链方向已反转")
        return {'FINISHED'}


class AUTOUNIFIED_OT_reset_reverses(bpy.types.Operator):
    bl_idname = "auto_skirt.reset_chain_reverses"
    bl_label = "全部正向"
    bl_description = "恢复所有骨骼链为 L0 到 L3 的正向顺序"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _set_all_reverses(context, False)
        self.report({'INFO'}, "所有骨骼链已恢复正向")
        return {'FINISHED'}


class AUTOUNIFIED_OT_auto_counts(bpy.types.Operator):
    bl_idname = "auto_skirt.auto_chain_counts"
    bl_label = "按长度计算"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            values = _auto_chain_counts(context)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "逐链骨骼数：" + ", ".join(map(str, values)))
        return {'FINISHED'}


class AUTOUNIFIED_OT_uniform_counts(bpy.types.Operator):
    bl_idname = "auto_skirt.uniform_chain_counts"
    bl_label = "全部相同"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        main = context.scene.auto_skirt_settings
        _set_main_counts(main, [main.asym_default_bone_count] * main.asym_control_chain_count)
        _persist_chain_options(context)
        return {'FINISHED'}


class AUTOUNIFIED_OT_reset_counts(bpy.types.Operator):
    bl_idname = "auto_skirt.reset_chain_counts"
    bl_label = "恢复 6 根"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        main = context.scene.auto_skirt_settings
        main.asym_default_bone_count = 6
        _set_main_counts(main, [6] * main.asym_control_chain_count)
        _persist_chain_options(context)
        return {'FINISHED'}


class AUTOUNIFIED_OT_recalculate_normals(bpy.types.Operator):
    bl_idname = "auto_skirt.recalculate_control_normals"
    bl_label = "按代理重算法向"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            _reset_control_normals(context, 'RECALCULATE')
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class AUTOUNIFIED_OT_clear_normals(bpy.types.Operator):
    bl_idname = "auto_skirt.clear_custom_normals"
    bl_label = "清除自定义法向"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            _reset_control_normals(context, 'CLEAR')
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class AUTOUNIFIED_OT_invert_normals(bpy.types.Operator):
    bl_idname = "auto_skirt.invert_control_normals"
    bl_label = "全部反转法向"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            _reset_control_normals(context, 'INVERT')
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


def fit_closed_rails(records, cage, profile, levels):
    """Sample the outer fabric radius at each rail and height of the cage."""
    center, up = profile["center"], profile["up"]
    sectors = [[] for _ in cage]
    for record in records:
        delta = record["world"] - center
        height = delta.dot(up)
        radial = delta - up * height
        u = _M._raw_skirt_u(record["world"], center, profile["right"], profile["back"])
        sectors[round(u * len(cage)) % len(cage)].append((height, radial.length))
    result = []
    for index, rail in enumerate(cage):
        samples = sectors[index]
        if not samples:
            nearest = min((i for i, sector in enumerate(sectors) if sector),
                          key=lambda i: min((i-index) % len(cage), (index-i) % len(cage)))
            samples = sectors[nearest]
        direction = _M._radial_at_u(index / len(cage), profile["right"], profile["back"])
        points = []
        for level in range(levels):
            height = (rail[0].lerp(rail[-1], level / (levels - 1)) - center).dot(up)
            closest = min(abs(h - height) for h, radius in samples)
            band = closest + (rail[-1] - rail[0]).length / max(16, levels * 2)
            radius = max(r for h, r in samples if abs(h - height) <= band)
            points.append(center + up * height + direction * radius)
        result.append(points)
    return result


def rail_grid(chains, count):
    sampled = [_S.resample_polyline([obj.matrix_world.translation for obj in chain], count + 1)
               for chain in chains]
    return list(map(list, zip(*sampled)))


def rail_data(chains, settings):
    count = settings.asym_default_bone_count
    grid = rail_grid(chains, count)
    bones = list(zip(*grid))
    normals = _grid_default_normals(grid, settings.closed_skirt, chains[0][0].get("invert_open_normal", False))
    axes = list(zip(*normals))
    for index, chain in enumerate(chains):
        if any(_control_is_custom(obj) for obj in chain):
            points = [obj.matrix_world.translation for obj in chain]
            # Match authored orientations to physical distance along the control rail.
            distances = [0.0]
            for first, second in zip(points, points[1:]):
                distances.append(distances[-1] + (second - first).length)
            values = [_control_normal(obj, _vertical_tangent(chain, i)) for i, obj in enumerate(chain)]
            axes[index] = [_interpolate_vertical_normal(values, _inverse_cumulative(distances, i / count))
                           for i in range(count + 1)]
    for chain in bones:
        if any((b - a).length_squared <= _M.EPSILON for a, b in zip(chain, chain[1:])):
            raise ValueError("控制链过短或控制点重合，请返回编辑并拉开控制点")
    return dict(bone_chains=bones, preview_grid=grid, bone_counts=[count] * len(chains),
                bone_z_axes=axes, reverse_flags=_main_reverses(settings),
                chain_lengths=[sum((b - a).length for a, b in zip(chain, chain[1:])) for chain in bones])


def rail_preview(chains, settings):
    kind = 'SKIRT' if settings.closed_skirt else 'SURFACE'
    cache = _PREVIEW_CACHE[kind]
    signature = (settings.asym_default_bone_count,
                 tuple(tuple(obj.matrix_world.translation) for chain in chains for obj in chain))
    if signature != cache["input_signature"]:
        grid = rail_grid(chains, settings.asym_default_bone_count)
        cache["full"] = _M._proxy_draw_geometry(grid, settings.closed_skirt)
        cache["input_signature"] = signature
    return cache["full"]


def mirror_reference(chains, snapshot):
    return {i: _average([Vector(snapshot[obj.name][:3]) for obj in chain])
            for i, chain in enumerate(chains) if all(obj.name in snapshot for obj in chain)}


def highlighted_indices(chains, settings, snapshot):
    selected = {i for i, chain in enumerate(chains) if any(obj.select_get() for obj in chain)}
    points = mirror_reference(chains, snapshot)
    for axis, enabled in (('X', settings.control_mirror_x), ('Y', settings.control_mirror_y)):
        if enabled:
            mapping = _pair_lookup(_mirror_pairs_from_points(points, axis)[0])
            selected.update(mapping.get(i, i) for i in tuple(selected))
    return selected


_display_active = set()


def display_color(index):
    color = CHAIN_COLORS[index % len(CHAIN_COLORS)]
    if index in _display_active:
        return color
    mean = sum(color[:3]) / 3
    return tuple((mean * 0.82 + channel * 0.18) * 0.55 for channel in color[:3]) + (1.0,)


def update_display_colors(context, closed):
    global _display_active
    main = getattr(context.scene, "auto_skirt_settings", None)
    sheet = getattr(context.scene, "auto_sheet_settings", None)
    if not main or not sheet:
        return
    if closed and main.control_edit_active:
        _, chains = _main_control_objects(main, require=False)
        settings, raw = main, main.control_edit_snapshot
    elif not closed and sheet.edit_active:
        grids, _ = _sheet_find_controls(_S.primary_from_settings(sheet), sheet.prefix, require=False)
        chains = _sheet_grid_chains(grids[0]) if grids else []
        settings, raw = sheet, sheet.edit_snapshot
    else:
        return
    if chains and all(all(chain) for chain in chains):
        _display_active = highlighted_indices(chains, settings, _M._parse_json_dict(raw))


def workflow_stage(context):
    main = context.scene.auto_skirt_settings
    if main.control_edit_active or context.scene.auto_sheet_settings.edit_active:
        return 2
    return 3 if main.workflow_stage == 3 else 1


class AUTOSKIRT_OT_workflow_next(bpy.types.Operator):
    bl_idname = "auto_skirt.workflow_next"
    bl_label = "下一阶段"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        main = context.scene.auto_skirt_settings
        stage = workflow_stage(context)
        try:
            if stage == 1:
                rig = main.target_armature
                if rig is None or rig.name not in context.view_layer.objects or not rig.data.bones.get(main.parent_bone):
                    raise ValueError("请选择当前视图层中的目标骨架及其父骨骼")
                if not main.prefix.strip():
                    main.property_unset("prefix")
                if context.mode != "EDIT_MESH":
                    raise ValueError("请在网格编辑模式选择裙摆顶点")
                main.workflow_enabled = True
                main.closed_skirt = main.skirt_type == 'CLOSED'
                main.columns = main.asym_control_chain_count
                outcome = _generate_execute(self, context)
                if outcome == {'FINISHED'}:
                    _set_main_reverses(main, [False] * main.columns)
                    _invalidate_preview()
                    main.workflow_stage = 2
            elif stage == 2:
                if context.scene.auto_sheet_settings.edit_active:
                    outcome = bpy.ops.auto_sheet.finish_edit()
                else:
                    outcome = bpy.ops.auto_skirt.finish_control_edit()
                if outcome == {'FINISHED'}:
                    main["edit_stage_prefix"] = main.prefix
                    main.workflow_stage = 3
            else:
                if main.workflow_enabled:
                    _set_main_counts(main, [main.asym_default_bone_count] * main.asym_control_chain_count)
                outcome = _build_execute(self, context)
                if outcome == {'FINISHED'}:
                    main.workflow_stage = 1
            return outcome
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class AUTOSKIRT_OT_workflow_back(bpy.types.Operator):
    bl_idname = "auto_skirt.workflow_back"
    bl_label = "返回上一步"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        main = context.scene.auto_skirt_settings
        stage = workflow_stage(context)
        if stage in (2, 3) and context.mode != 'OBJECT':
            if not bpy.ops.object.mode_set.poll():
                self.report({'ERROR'}, "请先退出当前物体的编辑模式")
                return {'CANCELLED'}
            bpy.ops.object.mode_set(mode='OBJECT')
        if stage == 3:
            main.prefix = main.get("edit_stage_prefix", main.prefix)
            outcome = _edit_execute(self, context)
            if outcome == {'FINISHED'}:
                for key in ("asym_default_bone_count", "create_proxy_mesh", "bind_weights"):
                    main.property_unset(key)
                main.workflow_stage = 2
            return outcome
        if stage != 2:
            return {'CANCELLED'}
        sheet = context.scene.auto_sheet_settings
        edit = sheet if sheet.edit_active else main
        targets = (_S.stored_targets(main.target_mesh) if sheet.edit_active
                   else _M._stored_target_meshes(main.target_mesh))
        primary = main.target_mesh
        outcome = _delete_controls_execute(self, context)
        if outcome == {'FINISHED'}:
            for key in ("control_mirror_x", "control_mirror_y", "show_control_normals",
                        "live_proxy_preview" if edit == sheet else "control_live_proxy_preview"):
                edit.property_unset(key)
            main.property_unset("show_mesh_points")
            _set_main_reverses(main, [False] * main.asym_control_chain_count)
            main.workflow_stage = 1
            for obj in list(context.selected_objects):
                obj.select_set(False)
            for obj in targets or [primary]:
                if obj and obj.name in context.view_layer.objects:
                    obj.select_set(True)
            if primary and primary.name in context.view_layer.objects:
                context.view_layer.objects.active = primary
                if bpy.ops.object.mode_set.poll():
                    bpy.ops.object.mode_set(mode='EDIT')
        return outcome


class AUTOSKIRT_OT_reverse_selected(bpy.types.Operator):
    bl_idname = "auto_skirt.reverse_selected"
    bl_label = "反转选中链"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _, chains, main = _current_control_chains(context)
        active = [i for i, chain in enumerate(chains) if any(obj.select_get() for obj in chain)]
        if not active:
            self.report({'WARNING'}, "请先选中链上的控制点")
            return {'CANCELLED'}
        for i in active:
            key = _chain_reverse_property(i)
            setattr(main, key, not getattr(main, key))
        _persist_chain_options(context)
        return {'FINISHED'}


def workflow_panel(self, context):
    layout = self.layout
    main = context.scene.auto_skirt_settings
    stage = workflow_stage(context)
    layout.label(text=("1 / 3   生成准备", "2 / 3   控制点编辑", "3 / 3   骨骼生成")[stage - 1])
    box = layout.box()
    if stage == 1:
        if context.mode != "EDIT_MESH":
            box.label(text="在网格编辑模式中选择裙摆顶点", icon="INFO")
        box.prop(main, "target_armature")
        if main.target_armature:
            box.prop_search(main, "parent_bone", main.target_armature.data, "bones")
        else:
            box.prop(main, "parent_bone")
        box.label(text="生成设置")
        box.prop(main, "skirt_type")
        box.prop(main, "asym_control_chain_count", text="控制链数量")
        box.prop(main, "control_levels", text="每链控制点数")
    elif stage == 2:
        sheet = context.scene.auto_sheet_settings
        edit = sheet if sheet.edit_active else main
        row = box.row(align=True)
        row.prop(edit, "control_mirror_x", text="X 镜像", toggle=True)
        row.prop(edit, "control_mirror_y", text="Y 镜像", toggle=True)
        box.prop(edit, "live_proxy_preview" if sheet.edit_active else "control_live_proxy_preview", text="显示代理网格")
        box.prop(main, "show_mesh_points")
        box.prop(edit, "show_control_normals", text="显示轴向")
        box.operator("auto_skirt.reverse_selected")
    else:
        box.prop(main, "prefix", text="骨骼前缀")
        box.prop(main, "create_proxy_mesh")
        box.prop(main, "bind_weights")
        box.prop(main, "asym_default_bone_count", text="每链骨骼数")
    label = ("生成控制点", "完成控制点编辑", "生成并绑定骨骼" if main.bind_weights else "生成骨骼")[stage - 1]
    layout.operator("auto_skirt.workflow_next", text=label)
    back = layout.row()
    back.enabled = stage > 1
    back.operator("auto_skirt.workflow_back")


EXTRA_CLASSES = (
    AUTOSKIRT_OT_workflow_next,
    AUTOSKIRT_OT_workflow_back,
    AUTOSKIRT_OT_reverse_selected,
    AUTOUNIFIED_OT_reverse_all,
    AUTOUNIFIED_OT_reset_reverses,
    AUTOUNIFIED_OT_auto_counts,
    AUTOUNIFIED_OT_uniform_counts,
    AUTOUNIFIED_OT_reset_counts,
    AUTOUNIFIED_OT_recalculate_normals,
    AUTOUNIFIED_OT_clear_normals,
    AUTOUNIFIED_OT_invert_normals,
)

_OLD_GENERATE_EXECUTE = None
_OLD_EDIT_EXECUTE = None
_OLD_BUILD_EXECUTE = None
_OLD_DELETE_CONTROLS_EXECUTE = None
_OLD_DELETE_GENERATED_EXECUTE = None


def _generate_execute(self, context):
    settings = context.scene.auto_skirt_settings
    try:
        if _uses_open_surface(settings):
            sheet = _sync_common_to_sheet(context)
            count, regions = _S.generate(context, sheet)
            settings.target_mesh = sheet.target_mesh
            self.report({'INFO'}, f"已从 {count} 个顶点生成非闭合裙摆控制点并进入编辑")
            return {'FINISHED'}
        settings.skirt_shape_mode = 'ASYMMETRIC'
        collection, count = _M.generate_control_points(context, settings)
        primary, controls = _main_control_objects(settings, require=True)
        _initialize_chain_orientations(controls, True)
        primary[PROP_SCHEMA] = SCHEMA_VERSION
        primary[PROP_GENERATOR_TYPE] = 'SKIRT'
        primary[PROP_REVERSES] = json.dumps(_main_reverses(settings), separators=(",", ":"))
        _M._enter_control_edit_mode(context, settings, collection)
        self.report({'INFO'}, f"已从 {count} 个顶点生成闭合裙摆控制点并进入编辑")
        return {'FINISHED'}
    except (ValueError, RuntimeError) as exc:
        self.report({'ERROR'}, str(exc))
        return {'CANCELLED'}


def _edit_execute(self, context):
    settings = context.scene.auto_skirt_settings
    try:
        if _uses_open_surface(settings):
            sheet = _sync_common_to_sheet(context)
            primary = _S.primary_from_settings(sheet)
            if primary is None or str(primary.get(PROP_GENERATOR_TYPE, "")) != 'SURFACE':
                raise ValueError("当前没有可编辑的非闭合裙摆控制点，请先重新生成")
            _S.enter_edit(context, sheet)
        else:
            primary, _controls = _main_control_objects(settings, require=True)
            if int(primary.get(PROP_SCHEMA, 0)) != SCHEMA_VERSION:
                raise ValueError("旧版控制点不兼容，请删除后重新生成")
            if str(primary.get(PROP_GENERATOR_TYPE, "")) != 'SKIRT':
                raise ValueError("当前控制点不是闭合裙摆类型，请重新生成")
            _M._enter_control_edit_mode(context, settings)
    except (ValueError, RuntimeError) as exc:
        self.report({'ERROR'}, str(exc))
        return {'CANCELLED'}
    self.report({'INFO'}, "已进入控制点编辑")
    return {'FINISHED'}


def _hide_built_controls(context):
    _, chains, settings = _current_control_chains(context)
    for chain in chains:
        for obj in chain:
            if obj.name in context.view_layer.objects:
                obj.hide_set(True, view_layer=context.view_layer)
    settings.workflow_stage = 1


def _build_execute(self, context):
    settings = context.scene.auto_skirt_settings
    try:
        if _uses_open_surface(settings):
            sheet = _sync_common_to_sheet(context)
            if sheet.edit_active:
                _S.apply_local_mirror(context)
                _S.leave_edit(context, sheet, reset=False)
            bones, vertices, assignments, modifier_added = _S.build(context, sheet)
            _hide_built_controls(context)
            message = f"已生成 {bones} 根骨骼"
            if not settings.workflow_enabled or settings.bind_weights:
                message += f"，为 {vertices} 个顶点写入 {assignments} 次权重"
            if modifier_added:
                message += "；已添加 Armature 修改器"
            self.report({'INFO'}, message)
            return {'FINISHED'}
        settings.skirt_shape_mode = 'ASYMMETRIC'
        if settings.control_edit_active:
            _M._apply_control_edit_mirror(context)
            _M._leave_control_edit_mode(context, settings, reset=False)
        _M._force_fixed_weight_defaults(settings)
        result = _M._build(context, settings, True, not settings.workflow_enabled or settings.bind_weights)
        _hide_built_controls(context)
        message = f"已生成 {result['bones']} 根骨骼"
        if not settings.workflow_enabled or settings.bind_weights:
            message += f"，为 {result['vertices']} 个顶点写入 {result['assignments']} 次权重"
        if result["modifier_added"]:
            message += "；已添加 Armature 修改器"
        self.report({'INFO'}, message)
        if result.get("warning"):
            self.report({'WARNING'}, result["warning"])
        return {'FINISHED'}
    except (ValueError, RuntimeError) as exc:
        self.report({'ERROR'}, str(exc))
        return {'CANCELLED'}


def _delete_controls_execute(self, context):
    settings = context.scene.auto_skirt_settings
    if not _uses_open_surface(settings):
        primary = settings.target_mesh
        result = _OLD_DELETE_CONTROLS_EXECUTE(self, context)
        if result == {'FINISHED'} and primary is not None:
            for key in (PROP_SCHEMA, PROP_GENERATOR_TYPE, PROP_REVERSES):
                if key in primary:
                    del primary[key]
        return result
    sheet = _sync_common_to_sheet(context)
    primary = _S.primary_from_settings(sheet)
    if primary is None:
        self.report({'ERROR'}, "找不到非闭合裙摆控制点记录")
        return {'CANCELLED'}
    if sheet.edit_active:
        _S.leave_edit(context, sheet, reset=False)
    prefixes = {sheet.prefix, str(primary.get(_S.PROP_PREFIX, sheet.prefix))}
    removed = sum(_S.delete_controls(prefix, primary.name) for prefix in prefixes)
    for key in (_S.PROP_COLLECTION, _S.PROP_REGIONS, _S.PROP_TARGETS, PROP_SCHEMA, PROP_GENERATOR_TYPE):
        if key in primary:
            del primary[key]
    self.report({'INFO'}, f"已删除 {removed} 个非闭合裙摆控制点")
    return {'FINISHED'}


def _delete_generated_execute(self, context):
    settings = context.scene.auto_skirt_settings
    if not _uses_open_surface(settings):
        return _OLD_DELETE_GENERATED_EXECUTE(self, context)
    try:
        sheet = _sync_common_to_sheet(context)
        bones, groups = _S.delete_generated(context, sheet, include_controls=False)
    except (ValueError, RuntimeError) as exc:
        self.report({'ERROR'}, str(exc))
        return {'CANCELLED'}
    self.report({'INFO'}, f"已删除 {bones} 根骨骼和 {groups} 个顶点组")
    return {'FINISHED'}


def _sheet_header_noop(_self, _context):
    return None


def _update_public_generator_setting(settings, context):
    if context is None or context.scene is None:
        return
    if _uses_open_surface(settings):
        _sync_common_to_sheet(context)
        _invalidate_preview('SURFACE')
    else:
        _invalidate_preview('SKIRT')
    _tag_view3d_redraw()


def configure(main_module, sheet_module):
    """Patch the two existing generators before Blender class registration."""
    global _M, _S, _CONFIGURED
    global _OLD_MAIN_BUILD_DATA, _OLD_MAIN_CREATE_BONES, _OLD_MAIN_COMPUTE_WEIGHTS
    global _OLD_MAIN_BUILD, _OLD_MAIN_INITIAL_PARAMETERS, _OLD_MAIN_LIVE_PROXY
    global _OLD_MAIN_DRAW_MARKERS, _OLD_SHEET_LIVE_PROXY, _OLD_SHEET_DRAW_MARKERS
    global _OLD_GENERATE_EXECUTE, _OLD_EDIT_EXECUTE, _OLD_BUILD_EXECUTE
    global _OLD_DELETE_CONTROLS_EXECUTE, _OLD_DELETE_GENERATED_EXECUTE
    if _CONFIGURED:
        return
    _M, _S = main_module, sheet_module
    _CONFIGURED = True

    _M.ASYM_CONTROL_CHAIN_COLORS = CHAIN_COLORS
    _S.CONTROL_CHAIN_COLORS = CHAIN_COLORS

    main_annotations = _M.AUTOSKIRT_PG_settings.__annotations__
    main_annotations["workflow_enabled"] = BoolProperty(default=False, options={'HIDDEN'})
    main_annotations["workflow_stage"] = IntProperty(default=1, min=1, max=3)
    main_annotations["bind_weights"] = BoolProperty(name="绑定权重", default=True)
    main_annotations["skirt_type"] = EnumProperty(name="裙摆类型", items=(
        ('CLOSED', "环绕式裙摆", "首尾闭合"), ('OPEN', "开放式裙摆", "围裙、披风等"),
    ))
    main_annotations["closed_skirt"] = BoolProperty(
        name="闭合裙摆",
        default=True,
        description="开启时使用闭环裙摆外包络拟合；关闭时使用原面片的开环边界逻辑",
        update=_update_public_generator_setting,
    )
    main_annotations["columns"] = IntProperty(
        name="骨骼链数",
        default=12,
        min=4,
        max=256,
        update=_update_public_generator_setting,
    )
    main_annotations["asym_default_bone_count"] = IntProperty(
        name="默认每链骨骼数",
        default=6,
        min=1,
        max=64,
        description="自动计算时与平均裙长对应的骨骼数量",
        update=_update_public_generator_setting,
    )
    main_annotations["asym_control_chain_count"] = IntProperty(
        name="控制点链数",
        default=6,
        min=4,
        max=24,
        description="沿裙摆宽度或周向布置的控制链数量",

    )
    main_annotations["show_mesh_points"] = BoolProperty(
        name="显示网格点", default=False, description="显示生成控制点时选中的网格顶点",
        update=lambda self, context: _tag_view3d_redraw(),
    )
    main_annotations["show_control_normals"] = BoolProperty(
        name="显示自定义法向",
        default=False,
        description="显示每个控制点的目标骨骼 +Z；旋转控制点可调整扭转",
    )
    for index in range(24):
        main_annotations[_chain_count_property(index)] = IntProperty(
            name=f"C{index:02d}",
            default=6,
            min=1,
            max=64,
            update=_update_public_generator_setting,
        )
        main_annotations[_chain_reverse_property(index)] = BoolProperty(
            name=f"C{index:02d} 反转骨骼方向",
            default=False,
            description="反转本链的骨骼头尾、父子顺序、R00 编号和权重方向；不移动控制点",
            update=_update_public_generator_setting,
        )

    sheet_annotations = _S.AUTOSHEET_PG_settings.__annotations__
    sheet_annotations["control_chain_count"] = IntProperty(
        name="控制点链数", default=6, min=4, max=24
    )
    sheet_annotations["default_bone_count"] = IntProperty(
        name="默认每链骨骼数", default=6, min=1, max=64
    )
    sheet_annotations["control_mirror_x"] = BoolProperty(name="X 镜像", default=False)
    sheet_annotations["control_mirror_y"] = BoolProperty(name="Y 镜像", default=False)
    sheet_annotations["show_control_normals"] = BoolProperty(
        name="显示自定义法向", default=False
    )
    for index in range(24):
        sheet_annotations[_sheet_count_property(index)] = IntProperty(
            name=f"C{index:02d}", default=6, min=1, max=64
        )
        sheet_annotations[_sheet_reverse_property(index)] = BoolProperty(
            name=f"C{index:02d} 反向", default=False
        )

    _OLD_MAIN_BUILD_DATA = _M.build_variable_asymmetric_data
    _OLD_MAIN_CREATE_BONES = _M.create_variable_asymmetric_bones
    _OLD_MAIN_COMPUTE_WEIGHTS = _M.compute_variable_asymmetric_vertex_weights
    _OLD_MAIN_BUILD = _M._build
    _OLD_MAIN_INITIAL_PARAMETERS = _M._initial_variable_asym_parameters
    _OLD_MAIN_LIVE_PROXY = _M._control_edit_live_proxy_geometry
    _OLD_MAIN_DRAW_MARKERS = _M._draw_control_edit_selection_markers
    _OLD_SHEET_LIVE_PROXY = _S.live_proxy_geometry
    _OLD_SHEET_DRAW_MARKERS = _S.draw_markers
    _OLD_GENERATE_EXECUTE = _M.AUTOSKIRT_OT_generate_control_points.execute
    _OLD_EDIT_EXECUTE = _M.AUTOSKIRT_OT_edit_control_points.execute
    _OLD_BUILD_EXECUTE = _M.AUTOSKIRT_OT_build.execute
    _OLD_DELETE_CONTROLS_EXECUTE = _M.AUTOSKIRT_OT_delete_controls.execute
    _OLD_DELETE_GENERATED_EXECUTE = _M.AUTOSKIRT_OT_delete_bones_groups.execute

    _M._control_position_map = _main_position_map
    _M._restore_control_positions = _main_restore_positions
    _M._changed_control_names = _main_changed_names
    _M._apply_dynamic_mirror_axis = _main_dynamic_mirror
    _M._apply_control_edit_mirror = _main_apply_control_mirror
    _M._initial_variable_asym_parameters = _cached_initial_asym_parameters
    _M.build_variable_asymmetric_data = _main_build_variable_data
    _M.create_variable_asymmetric_bones = _main_create_variable_bones
    _M.compute_variable_asymmetric_vertex_weights = _main_variable_weights
    _M._build = _main_build_wrapper
    _M._control_edit_live_proxy_geometry = _main_live_proxy
    _M._draw_control_edit_selection_markers = _main_draw_markers_with_normals
    _M._draw_control_edit_header = _sheet_header_noop

    _S.chain_display_color = display_color
    _S.fit_region_grid = _fit_open_region_grid
    _S.create_controls = _sheet_create_controls
    _S.find_controls = _sheet_find_controls
    _S.position_map = _sheet_position_map
    _S.restore_positions = _sheet_restore_positions
    _S.interpolate_horizontal_curve = _interpolate_open_curve
    _S.compute_weights = _sheet_compute_weights
    _S.create_bones = _sheet_create_bones_compat
    _S.validate_build = _sheet_validate_build
    _S.build = _sheet_build
    _S.generate = _sheet_generate
    _S.apply_local_mirror = _surface_mirror
    _S.marker_positions = _sheet_marker_positions
    _S.live_proxy_geometry = _surface_live_proxy
    _S.draw_markers = _sheet_draw_markers_with_normals
    _S.draw_header = _sheet_header_noop

    _M.AUTOSKIRT_OT_generate_control_points.execute = _generate_execute
    _M.AUTOSKIRT_OT_edit_control_points.execute = _edit_execute
    _M.AUTOSKIRT_OT_build.execute = _build_execute
    _M.AUTOSKIRT_OT_delete_controls.execute = _delete_controls_execute
    _M.AUTOSKIRT_OT_delete_bones_groups.execute = _delete_generated_execute
    _M.AUTOSKIRT_PT_main.draw = workflow_panel

    # The sheet implementation remains registered for its internal operators
    # and edit session, but only the unified public panel is shown.
    _S.CLASSES = tuple(
        cls for cls in _S.CLASSES
        if cls is not _S.AUTOSHEET_PT_main
    )
    if not all(cls in _M.classes for cls in EXTRA_CLASSES):
        _M.classes = tuple(_M.classes) + EXTRA_CLASSES
