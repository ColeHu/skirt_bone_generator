bl_info = {
    "name": "skirt_bone_generator",
    "author": "Akazure",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > 侧栏 > 裙子骨骼",
    "description": "根据选择的水平边环自动生成裙子骨骼链",
    "category": "Rigging",
}

import bpy
import bmesh
import math
import string
from mathutils import Vector
from bpy.props import (
    StringProperty, IntProperty, PointerProperty,
)


# ===================== 工具函数 =====================

def _get_selected_edge_loops(bm):
    """
    从 bmesh 中获取已选择的边，按连通环分组。
    返回 list[list[BMVert]]，每个子列表是一个环上的顶点（有序）。
    """
    sel_edges = [e for e in bm.edges if e.select]
    if not sel_edges:
        return []

    adj = {}
    for e in sel_edges:
        for v in e.verts:
            if v not in adj:
                adj[v] = []
        adj[e.verts[0]].append(e.verts[1])
        adj[e.verts[1]].append(e.verts[0])

    visited = set()
    loops = []
    for start_v in adj:
        if start_v in visited:
            continue
        ring = []
        queue = [start_v]
        while queue:
            v = queue.pop(0)
            if v in visited:
                continue
            visited.add(v)
            ring.append(v)
            for nb in adj.get(v, []):
                if nb not in visited:
                    queue.append(nb)
        if ring:
            loops.append(ring)

    return loops


def _sort_loop_verts(loop_verts):
    """对一个环上的顶点按角度排序（绕环中心），返回有序列表。"""
    if len(loop_verts) <= 2:
        return loop_verts

    center = Vector((0, 0, 0))
    for v in loop_verts:
        center += v.co
    center /= len(loop_verts)

    ref = (loop_verts[0].co - center).normalized()
    v0 = loop_verts[0].co - center
    v1 = loop_verts[1].co - center if len(loop_verts) > 1 else Vector((0, 0, 1))
    normal = v0.cross(v1).normalized()
    if normal.length < 1e-6:
        normal = Vector((0, 0, 1))

    def angle_key(v):
        d = v.co - center
        x = d.dot(ref)
        y = d.dot(normal.cross(ref))
        return math.atan2(y, x)

    return sorted(loop_verts, key=angle_key)


def _sort_loops_top_to_bottom(loops):
    """按平均 Z 坐标从高到低排序环"""
    def avg_z(loop):
        return sum(v.co.z for v in loop) / len(loop)
    return sorted(loops, key=avg_z, reverse=True)


def _fit_ellipse_coords(loop_verts, shared_frame=None):
    """
    为环上的顶点拟合一个最佳椭圆，然后将每个顶点投影到椭圆上
    对应角度的位置。返回 (投影坐标列表, 帧信息字典)。

    如果提供 shared_frame，则使用共享的坐标系和PCA轴方向，
    确保多个环的椭圆对齐。每个环仍然独立计算中心和半轴长度。
    """
    if len(loop_verts) <= 2:
        return [v.co.copy() for v in loop_verts], None

    center = Vector((0, 0, 0))
    for v in loop_verts:
        center += v.co
    center /= len(loop_verts)

    # 坐标系：使用共享帧或自行计算
    if shared_frame:
        ref = shared_frame['ref']
        tangent = shared_frame['tangent']
        normal = shared_frame['normal']
    else:
        ref = (loop_verts[0].co - center).normalized()
        v0 = loop_verts[0].co - center
        v1 = loop_verts[1].co - center if len(loop_verts) > 1 else Vector((0, 0, 1))
        normal = v0.cross(v1).normalized()
        if normal.length < 1e-6:
            normal = Vector((0, 0, 1))
        tangent = normal.cross(ref).normalized()

    # 投影到2D
    pts_2d = []
    heights = []
    for v in loop_verts:
        d = v.co - center
        x = d.dot(ref)
        y = d.dot(tangent)
        h = d.dot(normal)
        pts_2d.append((x, y))
        heights.append(h)

    n = len(pts_2d)
    mean_x = sum(p[0] for p in pts_2d) / n
    mean_y = sum(p[1] for p in pts_2d) / n

    # PCA轴：使用共享帧或自行计算
    if shared_frame and 'ex1' in shared_frame:
        ex1 = shared_frame['ex1']
        ex2 = shared_frame['ex2']
    else:
        # 协方差矩阵
        cxx = sum((p[0] - mean_x) ** 2 for p in pts_2d) / n
        cxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pts_2d) / n
        cyy = sum((p[1] - mean_y) ** 2 for p in pts_2d) / n

        # 特征值分解 2x2
        trace = cxx + cyy
        det = cxx * cyy - cxy * cxy
        disc = math.sqrt(max(trace * trace / 4 - det, 0))
        lambda1 = trace / 2 + disc

        # 主方向
        if abs(cxy) > 1e-10:
            ex1 = (lambda1 - cyy, cxy)
            mag1 = math.sqrt(ex1[0] ** 2 + ex1[1] ** 2)
            ex1 = (ex1[0] / mag1, ex1[1] / mag1)
        elif cxx >= cyy:
            ex1 = (1.0, 0.0)
        else:
            ex1 = (0.0, 1.0)
        ex2 = (-ex1[1], ex1[0])

    # 每个环独立计算半轴长度
    max_a = 0
    max_b = 0
    for p in pts_2d:
        dx, dy = p[0] - mean_x, p[1] - mean_y
        proj_a = abs(dx * ex1[0] + dy * ex1[1])
        proj_b = abs(dx * ex2[0] + dy * ex2[1])
        if proj_a > max_a:
            max_a = proj_a
        if proj_b > max_b:
            max_b = proj_b

    if max_a < 1e-6:
        max_a = 1e-6
    if max_b < 1e-6:
        max_b = 1e-6

    # 将每个顶点按其角度投影到椭圆上
    projected = []
    for i, (px, py) in enumerate(pts_2d):
        dx, dy = px - mean_x, py - mean_y
        u = dx * ex1[0] + dy * ex1[1]
        v = dx * ex2[0] + dy * ex2[1]
        angle = math.atan2(v, u)

        eu = max_a * math.cos(angle)
        ev = max_b * math.sin(angle)

        nx = mean_x + eu * ex1[0] + ev * ex2[0]
        ny = mean_y + eu * ex1[1] + ev * ex2[1]

        new_co = center + ref * nx + tangent * ny + normal * heights[i]
        projected.append(new_co)

    frame = {
        'ref': ref, 'tangent': tangent, 'normal': normal,
        'ex1': ex1, 'ex2': ex2,
        'max_a': max_a, 'max_b': max_b, 'center': center.copy(),
    }
    return projected, frame


def _ellipse_equal_arc_angles(a, b, t_start, t_end, n_points, steps=512):
    """
    在椭圆上从参数角 t_start 到 t_end 返回 n_points 个等弧长的参数角度。
    a, b 是椭圆半轴长度。包含 t_start 和 t_end。
    """
    if n_points <= 1:
        return [t_start]
    if n_points == 2:
        return [t_start, t_end]

    dt = (t_end - t_start) / steps
    cum_len = [0.0]
    for i in range(steps):
        t = t_start + (i + 0.5) * dt
        ds = math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2) * abs(dt)
        cum_len.append(cum_len[-1] + ds)

    total = cum_len[-1]
    if total < 1e-12:
        return [t_start + (t_end - t_start) * k / (n_points - 1)
                for k in range(n_points)]

    angles = [t_start]
    for k in range(1, n_points - 1):
        target = total * k / (n_points - 1)
        lo, hi = 0, steps
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_len[mid + 1] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo < steps:
            seg_len = cum_len[lo + 1] - cum_len[lo]
            frac = (target - cum_len[lo]) / max(seg_len, 1e-12)
            angle = t_start + (lo + frac) * dt
        else:
            angle = t_end
        angles.append(angle)
    angles.append(t_end)
    return angles


def _chain_label(chain_idx, total_chains):
    """
    生成骨骼链标签：
    - 链数 <= 26 时用字母 a~z
    - 链数 > 26 时用三位数字 001~
    """
    if total_chains <= 26:
        return string.ascii_lowercase[chain_idx]
    else:
        return f"{chain_idx + 1:03d}"


def _bone_name(prefix, chain_label, bone_idx, side=""):
    """生成骨骼名称: prefix.chain_label.bone_idx[.L/.R]"""
    name = f"{prefix}.{chain_label}.{bone_idx + 1:03d}"
    if side:
        name += f".{side}"
    return name


def _find_closest_vert_in_loop(loop_verts, target_co):
    """在环上找到距离 target_co 最近的顶点"""
    best = None
    best_dist = float('inf')
    for v in loop_verts:
        d = (v.co - target_co).length
        if d < best_dist:
            best_dist = d
            best = v
    return best


# ===================== 标记中轴操作 =====================

class SKIRT_OT_MarkFrontAxis(bpy.types.Operator):
    """标记当前选中的顶点为前中轴"""
    bl_idname = "skirt.mark_front_axis"
    bl_label = "标记前中轴"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        props = context.scene.skirt_bone_props
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        sel_verts = [v for v in bm.verts if v.select]
        if len(sel_verts) != 1:
            self.report({'ERROR'}, "请选择恰好 1 个顶点作为前中轴")
            return {'CANCELLED'}

        props.front_axis_vert_index = sel_verts[0].index
        self.report({'INFO'}, f"前中轴已标记: 顶点 {sel_verts[0].index}")
        return {'FINISHED'}


class SKIRT_OT_MarkBackAxis(bpy.types.Operator):
    """标记当前选中的顶点为后中轴"""
    bl_idname = "skirt.mark_back_axis"
    bl_label = "标记后中轴"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        props = context.scene.skirt_bone_props
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        sel_verts = [v for v in bm.verts if v.select]
        if len(sel_verts) != 1:
            self.report({'ERROR'}, "请选择恰好 1 个顶点作为后中轴")
            return {'CANCELLED'}

        props.back_axis_vert_index = sel_verts[0].index
        self.report({'INFO'}, f"后中轴已标记: 顶点 {sel_verts[0].index}")
        return {'FINISHED'}


# ===================== 主操作 =====================

class SKIRT_OT_GenerateBones(bpy.types.Operator):
    """根据选择的水平边环和中轴标记生成裙子骨骼"""
    bl_idname = "skirt.generate_bones"
    bl_label = "生成裙子骨骼"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        props = context.scene.skirt_bone_props
        obj = context.active_object

        # 检查中轴标记
        if props.front_axis_vert_index < 0 or props.back_axis_vert_index < 0:
            self.report({'ERROR'}, "请先标记前中轴和后中轴顶点")
            return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        # 1) 提取选中的边环
        loops = _get_selected_edge_loops(bm)
        if len(loops) < 2:
            self.report({'ERROR'}, "至少需要选择 2 条边环")
            return {'CANCELLED'}

        # 2) 排序
        sorted_loops = []
        for loop in loops:
            sorted_loops.append(_sort_loop_verts(loop))
        sorted_loops = _sort_loops_top_to_bottom(sorted_loops)

        # 2.5) 百褶裙处理：将每个环的顶点投影到拟合椭圆上
        #      使用顶层环的坐标系和PCA轴作为共享帧，确保所有椭圆对齐
        vert_ellipse_co = {}
        shared_frame = None
        for loop in sorted_loops:
            ellipse_coords, frame = _fit_ellipse_coords(loop, shared_frame)
            if shared_frame is None:
                shared_frame = frame
            for v, co in zip(loop, ellipse_coords):
                vert_ellipse_co[v] = co

        # 3) 获取前/后中轴顶点坐标
        front_axis_co = bm.verts[props.front_axis_vert_index].co.copy()
        back_axis_co = bm.verts[props.back_axis_vert_index].co.copy()

        # 4) 构建网格完整邻接表
        full_adj = {}
        for e in bm.edges:
            for v in e.verts:
                if v not in full_adj:
                    full_adj[v] = []
            full_adj[e.verts[0]].append(e.verts[1])
            full_adj[e.verts[1]].append(e.verts[0])

        # 5) BFS 找上下环对应顶点
        loop_connections = []
        for i in range(len(sorted_loops) - 1):
            lower_set = set(sorted_loops[i + 1])
            all_loop_verts = set()
            for loop in sorted_loops:
                all_loop_verts.update(loop)
            conn = {}
            for start_v in sorted_loops[i]:
                visited = {start_v}
                queue = [start_v]
                found = None
                while queue:
                    v = queue.pop(0)
                    if v in lower_set:
                        found = v
                        break
                    for nb in full_adj.get(v, []):
                        if nb not in visited:
                            if nb in lower_set or nb not in all_loop_verts:
                                visited.add(nb)
                                queue.append(nb)
                if found:
                    conn[start_v] = found
            loop_connections.append(conn)

        # 6) 构建完整骨骼链
        chains = []
        for v in sorted_loops[0]:
            chain = [v]
            current = v
            for conn in loop_connections:
                if current not in conn:
                    break
                current = conn[current]
                chain.append(current)
            if len(chain) == len(sorted_loops):
                chains.append(chain)

        if not chains:
            self.report({'ERROR'}, "上下环之间没有找到相连的顶点对")
            return {'CANCELLED'}

        num_segments = len(sorted_loops) - 1

        # 7) 使用椭圆PCA角度排序链，实现弧长均匀分布
        top_loop = sorted_loops[0]
        top_center = shared_frame['center']

        front_on_top = _find_closest_vert_in_loop(top_loop, front_axis_co)
        back_on_top = _find_closest_vert_in_loop(top_loop, back_axis_co)

        world_mat = obj.matrix_world

        # 计算每个环的中心（基于椭圆投影坐标）
        loop_centers = []
        for loop in sorted_loops:
            c = sum((world_mat @ vert_ellipse_co[v] for v in loop),
                    Vector((0, 0, 0)))
            c /= len(loop)
            loop_centers.append(c)

        # 使用顶层椭圆的PCA坐标系计算角度
        ef = shared_frame

        def _pca_angle(v):
            """计算顶点在顶层椭圆PCA坐标系中的参数角度"""
            d = v.co - top_center
            px = d.dot(ef['ref'])
            py = d.dot(ef['tangent'])
            u = px * ef['ex1'][0] + py * ef['ex1'][1]
            w = px * ef['ex2'][0] + py * ef['ex2'][1]
            return math.atan2(w, u)

        pca_front = _pca_angle(front_on_top)
        pca_back = _pca_angle(back_on_top)
        if pca_back <= pca_front:
            pca_back += 2 * math.pi

        # 确保前后中轴链
        front_chain = None
        back_chain = None
        for chain in chains:
            if chain[0] == front_on_top:
                front_chain = chain
            elif chain[0] == back_on_top:
                back_chain = chain

        # 收集前中轴到后中轴之间(+X侧)的所有链，按PCA角度排序
        all_left_chains = []
        for chain in chains:
            head_v = chain[0]
            if head_v == front_on_top or head_v == back_on_top:
                continue
            angle = _pca_angle(head_v)
            if angle < pca_front:
                angle += 2 * math.pi
            if pca_front < angle < pca_back:
                all_left_chains.append((angle, chain))

        all_left_chains.sort(key=lambda x: x[0])

        # 8) 按 bone_count 在椭圆上等弧长选取 L 侧链
        bone_count = max(1, props.bone_count)
        if len(all_left_chains) <= bone_count:
            left_chains = [chain for _, chain in all_left_chains]
        else:
            # 计算等弧长目标角度
            target_pca = _ellipse_equal_arc_angles(
                ef['max_a'], ef['max_b'],
                pca_front, pca_back, bone_count + 2)

            left_chains = []
            used = set()
            for i in range(1, bone_count + 1):
                target = target_pca[i]
                best_idx = None
                best_diff = float('inf')
                for j, (angle, chain) in enumerate(all_left_chains):
                    if j in used:
                        continue
                    diff = abs(angle - target)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = j
                if best_idx is not None:
                    used.add(best_idx)
                    left_chains.append(all_left_chains[best_idx][1])

        # 9) 编排标签：前中轴(a) → 左侧按角度(b,c,d...+.L,镜像.R) → 后中轴
        total_labels = (1 if front_chain else 0) + len(left_chains) + (1 if back_chain else 0)
        label_idx = 0
        labeled_chains = []

        # 前中轴
        if front_chain:
            coords = [world_mat @ vert_ellipse_co[v] for v in front_chain]
            labeled_chains.append(
                (_chain_label(label_idx, total_labels), coords, ""))
            label_idx += 1

        # 左侧链(.L) + 镜像(.R)
        for chain in left_chains:
            label = _chain_label(label_idx, total_labels)
            coords = [world_mat @ vert_ellipse_co[v] for v in chain]
            labeled_chains.append((label, coords, "L"))
            mirrored = [Vector((-c.x, c.y, c.z)) for c in coords]
            labeled_chains.append((label, mirrored, "R"))
            label_idx += 1

        # 后中轴
        if back_chain:
            coords = [world_mat @ vert_ellipse_co[v] for v in back_chain]
            labeled_chains.append(
                (_chain_label(label_idx, total_labels), coords, ""))
            label_idx += 1

        if not labeled_chains:
            self.report({'ERROR'}, "没有找到可生成的骨骼链")
            return {'CANCELLED'}

        # 10) 退出编辑模式，创建骨架
        bpy.ops.object.mode_set(mode='OBJECT')

        arm_data = bpy.data.armatures.new(f"{props.name_prefix}")
        arm_obj = bpy.data.objects.new(
            f"{props.name_prefix}", arm_data)
        context.collection.objects.link(arm_obj)
        context.view_layer.objects.active = arm_obj
        arm_obj.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')

        # 11) 生成骨骼
        created = 0
        for label, coords, side in labeled_chains:
            parent_bone = None

            centers = loop_centers
            if side == "R":
                centers = [Vector((-c.x, c.y, c.z)) for c in loop_centers]

            for seg in range(num_segments):
                head_pos = coords[seg]
                tail_pos = coords[seg + 1]

                name = _bone_name(props.name_prefix, label, seg, side)
                bone = arm_data.edit_bones.new(name)
                bone.head = head_pos
                bone.tail = tail_pos

                if bone.length < 1e-5:
                    bone.tail = bone.head.copy()
                    bone.tail.z -= 0.02

                # Z轴从环中心向外发散
                outward = head_pos - centers[seg]
                if outward.length > 1e-6:
                    bone.align_roll(outward)

                if parent_bone:
                    bone.parent = parent_bone
                    bone.use_connect = True

                parent_bone = bone
                created += 1

        bpy.ops.object.mode_set(mode='OBJECT')

        total_with_mirror = len(labeled_chains)
        info = (f"{total_labels} 条链(含镜像 {total_with_mirror} 条)"
                f" × {num_segments} 段 = {created} 根骨骼")
        self.report({'INFO'}, f"生成完成: {info}")
        props.last_info = info
        return {'FINISHED'}


# ===================== 属性 =====================

class SkirtBoneProperties(bpy.types.PropertyGroup):
    name_prefix: StringProperty(
        name="命名前缀",
        default="skirt",
        description="骨骼命名前缀，如 skirt → skirt.a.001",
    )
    bone_count: IntProperty(
        name="L侧骨骼链数量(不包含中轴)",
        default=5,
        min=1,
        description="前后中轴之间L侧(+x方向)生成的骨骼链数量（R侧自动镜像同等数量）",
    )
    front_axis_vert_index: IntProperty(
        name="前中轴顶点索引",
        default=-1,
    )
    back_axis_vert_index: IntProperty(
        name="后中轴顶点索引",
        default=-1,
    )
    last_info: StringProperty(
        name="上次生成信息",
        default="",
    )


# ===================== 面板 =====================

class SKIRT_PT_Panel(bpy.types.Panel):
    bl_label = "裙子骨骼生成器"
    bl_idname = "SKIRT_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "裙子骨骼"

    def draw(self, context):
        layout = self.layout
        props = context.scene.skirt_bone_props

        box = layout.box()
        box.label(text="命名", icon='FONT_DATA')
        box.prop(props, "name_prefix")

        # 中轴标记
        box_axis = layout.box()
        box_axis.label(text="中轴标记", icon='ORIENTATION_NORMAL')
        row_f = box_axis.row(align=True)
        row_f.operator("skirt.mark_front_axis", icon='TRACKING_FORWARDS',
                       text="标记前中轴")
        if props.front_axis_vert_index >= 0:
            row_f.label(text=f"✓ 顶点 {props.front_axis_vert_index}")
        else:
            row_f.label(text="未标记")
        row_b = box_axis.row(align=True)
        row_b.operator("skirt.mark_back_axis", icon='TRACKING_BACKWARDS',
                       text="标记后中轴")
        if props.back_axis_vert_index >= 0:
            row_b.label(text=f"✓ 顶点 {props.back_axis_vert_index}")
        else:
            row_b.label(text="未标记")

        # 骨骼分布
        box2 = layout.box()
        box2.label(text="骨骼分布", icon='MOD_ARRAY')
        box2.prop(props, "bone_count")

        layout.separator()

        row = layout.row()
        row.scale_y = 1.6
        row.operator("skirt.generate_bones", icon='BONE_DATA',
                     text="⚡ 生成裙子骨骼")

        if props.last_info:
            box3 = layout.box()
            box3.label(text="上次生成结果:", icon='CHECKMARK')
            box3.label(text=props.last_info)

        layout.separator()
        layout.label(text="使用方法:", icon='INFO')
        col = layout.column(align=True)
        col.label(text="1. 进入裙子网格编辑模式")
        col.label(text="2. 选顶点标记前/后中轴")
        col.label(text="3. 环选多条水平边")
        col.label(text="4. 设置参数后点击生成")


# ===================== 注册 =====================

classes = (
    SkirtBoneProperties,
    SKIRT_OT_MarkFrontAxis,
    SKIRT_OT_MarkBackAxis,
    SKIRT_OT_GenerateBones,
    SKIRT_PT_Panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.skirt_bone_props = PointerProperty(type=SkirtBoneProperties)


def unregister():
    del bpy.types.Scene.skirt_bone_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
