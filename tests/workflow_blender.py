"""Exercise the public workflow in a factory-startup Blender process."""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import skirt_bone_generator as addon

ui = addon.unified_generator


def make_cloth(kind):
    closed = kind in {"skirt", "pleats"}
    cols, rows = 36, 9
    vertices, faces = [], []
    for j in range(rows):
        t = j / (rows - 1)
        for i in range(cols):
            u = i / (cols - 1)
            if closed:
                a = math.tau * i / cols
                r = 0.45 + 0.35*t
                if kind == "pleats":
                    r += 0.08 * math.cos(12*a) * t
                point = (r*math.cos(a), r*math.sin(a), 1-t + 0.12*t*math.cos(2*a))
            else:
                x = (2*u-1) * (0.55+0.2*t)
                if kind == "tip":
                    x *= 1-t
                if kind == "narrow":
                    x *= 0.012
                point = (x, 0.1*x*x + (0.35*t*t if kind == "cape" else 0), 0 if kind == "flat" else 1-t)
            vertices.append(point)
    for j in range(rows-1):
        for i in range(cols if closed else cols-1):
            a = j*cols+i
            b = j*cols+(i+1)%cols
            faces.append((a,b,b+cols,a+cols))
    mesh = bpy.data.meshes.new(kind)
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(kind, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    return obj, closed


def exercise(kind, n, levels, bind=True, proxy=True, axes=("X",)):
    # Each case runs in its own Blender scene.
    ctx = bpy.context
    bpy.ops.object.armature_add()
    rig = ctx.object
    root = rig.data.bones[0].name
    rig.select_set(False)
    cloth, closed = make_cloth(kind)
    s = ctx.scene.auto_skirt_settings
    s.target_armature, s.parent_bone = rig, root
    s.skirt_type = "CLOSED" if closed else "OPEN"
    s.asym_control_chain_count, s.control_levels = n, levels
    next_step, back = bpy.ops.auto_skirt.workflow_next, bpy.ops.auto_skirt.workflow_back
    if kind == "flat":
        try:
            next_step()
        except RuntimeError:
            pass
        assert ui.workflow_stage(ctx) == 1 and len(rig.data.bones) == 1
        return
    assert next_step() == {"FINISHED"}
    assert back() == {"FINISHED"} and ctx.mode == "EDIT_MESH"
    assert s.control_levels == levels and s.asym_control_chain_count == n
    assert next_step() == {"FINISHED"}
    _, chains, _ = ui._current_control_chains(ctx)
    assert len(chains) == n and all(len(chain) == levels for chain in chains)
    edit = s if closed else ctx.scene.auto_sheet_settings
    snap = addon._parse_json_dict(edit.control_edit_snapshot if closed else edit.edit_snapshot)
    # Front is index 0 on a ring, and index 0 is the left edge on a sheet.
    for axis in ("X", "Y"):
        if axis == "Y" and (not closed or n % 2):
            continue
        pairs, _ = ui._mirror_pairs_from_points(ui.mirror_reference(chains, snap), axis)
        mapping = ui._pair_lookup(pairs)
        for index in range(n):
            if closed:
                partner = (-index if axis == "X" else n // 2 - index) % n
            else:
                partner = n - 1 - index
            assert mapping[index] == partner, (kind, n, axis, index, mapping[index], partner)
    normal_lines = ui._normal_line_geometry(chains)
    for i, chain in enumerate(chains):
        j = levels // 2
        origin, tip = normal_lines[(i*levels+j)*6:(i*levels+j)*6+2]
        reference = Vector((origin.x, origin.y, 0)) if closed else cloth.data.polygons[len(cloth.data.polygons)//2].normal
        assert (tip-origin).dot(reference) > 0, (kind, i, "outward axis")
    selected = 1
    level = levels // 2
    for chain in chains:
        for obj in chain:
            obj.select_set(False)
    src = chains[selected][level]
    src.select_set(True)
    ctx.view_layer.objects.active = src
    ref = ui.mirror_reference(chains, snap)
    expected = {selected: src.location.copy() + Vector((0.04, 0.02, 0.03))}
    source = expected[selected].copy()
    maps = {}
    centers = {}
    for axis in axes:
        setattr(edit, 'control_mirror_' + axis.lower(), True)
        pairs, _ = ui._mirror_pairs_from_points(ref, axis)
        maps[axis] = ui._pair_lookup(pairs)
        coords = [Vector(snap[c[level].name][:3]) for c in chains]
        centers[axis] = sum(p.x if axis == "X" else p.y for p in coords)/n
        if maps[axis][selected] == selected:
            source[0 if axis == "X" else 1] = centers[axis]
    expected = {selected: source}
    for axis in axes:
        for index, value in tuple(expected.items()):
            expected.setdefault(maps[axis][index], addon._mirror_location(value, axis, centers[axis]))
    assert ui.highlighted_indices(chains, edit, snap) == set(expected)
    if closed:
        edit.control_live_proxy_preview = True
    else:
        edit.live_proxy_preview = True
    def read_preview():
        return ui._main_live_proxy(ctx.scene) if closed else ui._surface_live_proxy()
    def mesh_points():
        return addon._control_edit_selected_world_positions(ctx.scene) if closed else ui._sheet_marker_positions()[1]
    assert not mesh_points()
    s.show_mesh_points = True
    assert len(mesh_points()) == len(cloth.data.vertices)
    s.show_mesh_points = False
    assert not mesh_points()
    old_lines = tuple(tuple(p) for p in read_preview()[1])
    src.location += Vector((0.04, 0.02, 0.03))
    # Trigger the registered dependency-graph mirror handler just as a viewport move does.
    ctx.view_layer.update()
    ctx.view_layer.update()
    new_lines = tuple(tuple(p) for p in read_preview()[1])
    assert len(old_lines) == len(new_lines) and old_lines != new_lines
    preview_grid = ui.rail_grid(chains, s.asym_default_bone_count)
    assert new_lines == tuple(tuple(p) for p in addon._proxy_draw_geometry(preview_grid, closed)[1])
    for index, value in expected.items():
        assert (chains[index][level].location - value).length < 1e-4, (kind, index, 'mirror')
    assert bpy.ops.auto_skirt.reverse_selected() == {"FINISHED"}
    assert next_step() == {"FINISHED"} and ui.workflow_stage(ctx) == 3
    paths, selected_points, (faces, edges) = ui.generation_preview(ctx.scene)
    assert len(selected_points) == len(cloth.data.vertices)
    assert len(paths) == n and all(len(points) == levels for points in paths.values())
    assert all(ui.display_color(i) == ui.CHAIN_COLORS[i] for i in range(n))
    assert faces and edges
    s.asym_default_bone_count = 11
    assert len(ui.generation_preview(ctx.scene)[2][1]) > len(edges)
    assert back() == {"FINISHED"} and s.asym_default_bone_count == 6
    assert ui._main_reverses(s)[selected]
    assert next_step() == {"FINISHED"}
    s.asym_default_bone_count = 5
    s.create_proxy_mesh, s.bind_weights = proxy, bind
    sampled = ui.rail_grid(chains, 5)
    from mathutils.geometry import intersect_point_line
    for i, chain in enumerate(chains):
        points = [obj.matrix_world.translation for obj in chain]
        assert (sampled[0][i] - points[0]).length < 1e-6
        assert (sampled[-1][i] - points[-1]).length < 1e-6
        for row in sampled:
            distances = []
            for a, b in zip(points, points[1:]):
                if (a-b).length < 1e-8:
                    distances.append((row[i]-a).length)
                    continue
                _, factor = intersect_point_line(row[i], a, b)
                distances.append((row[i]-a.lerp(b, max(0, min(1, factor)))).length)
            assert min(distances) < 1e-5
    assert all(not obj.hide_get() for chain in chains for obj in chain)
    before = set(ctx.scene.objects)
    assert next_step() == {"FINISHED"} and ui.workflow_stage(ctx) == 1
    assert all(obj.hide_get() for chain in chains for obj in chain)
    assert not cloth.hide_get() and not rig.hide_get()
    assert len(rig.data.bones) == n*5+1
    for i in range(n):
        rail = [row[i] for row in sampled]
        if i == selected:
            rail.reverse()
        for j in range(5):
            b = rig.data.bones[f"{s.prefix}C{i:02d}_R{j:02d}"]
            assert (rig.matrix_world @ b.head_local - rail[j]).length < 1e-4
            assert (rig.matrix_world @ b.tail_local - rail[j+1]).length < 1e-4
    proxies = [obj for obj in ctx.scene.objects if obj not in before and obj.type == "MESH"]
    assert len(proxies) == int(proxy)
    for obj in [cloth] + proxies:
        assert bool(obj.vertex_groups) == bind
        assert any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers) == bind
        if bind:
            assert all(1 <= len(v.groups) <= 4 and abs(sum(g.weight for g in v.groups)-1) < 0.0001
                       for v in obj.data.vertices)

    assert bpy.ops.auto_skirt.edit_control_points() == {"FINISHED"}
    assert all(not obj.hide_get() for chain in chains for obj in chain)
    assert bpy.ops.auto_skirt.workflow_next() == {"FINISHED"}


CASES = [('skirt',4,2,True,True,('X','Y')), ('skirt',7,7,True,False,('X',)),
         ('pleats',12,32,True,True,('Y',)), ('cape',5,6,True,True,('X',)),
         ('apron',6,3,False,True,('X',)), ('tip',9,8,True,True,('X',)),
         ('narrow',8,4,False,False,('Y',)), ('flat',6,4), ('skirt',10,4,False,True,('X','Y')),
         ('cape',11,9,True,False,('X',))]


if __name__ == '__main__':
    addon.register()
    for case in CASES:
        bpy.ops.scene.new(type='NEW')
        exercise(*case)
        print('PASS',case,flush=True)
    addon.unregister()
