"""Build a persistent 16-garment gallery and exercise the registered add-on via Blender MCP."""
import math
import traceback
import bpy
from mathutils import Vector
import skirt_bone_generator as addon

CASES = [
    ("斜裁单侧长裙", "bias", True, 7, 7),
    ("前短后长燕尾裙", "train", True, 9, 9),
    ("偏心腰口椭圆裙", "offset", True, 5, 5),
    ("螺旋扭褶裙", "spiral", True, 12, 12),
    ("前开衩筒裙", "slit", True, 8, 6),
    ("双侧开衩双片裙", "fork", True, 10, 9),
    ("四瓣分叉花裙", "petals", True, 12, 8),
    ("双层错缝荷叶裙", "ruffle", True, 11, 8),
    ("斜肩单侧长披风", "shoulder", False, 7, 6),
    ("偏置燕尾双叉披风", "swallow", False, 9, 10),
    ("三叉破边斗篷", "torn", False, 10, 8),
    ("镂空披肩", "hole", False, 6, 6),
    ("马蹄形环肩披风", "horseshoe", False, 12, 7),
    ("偏襟交叠围裙", "wrap", False, 8, 7),
    ("窄长蛇形飘带", "ribbon", False, 4, 16),
    ("疏密混合三角披风", "tri", False, 5, 9),
]
TAU = math.tau


def geometry(kind, closed):
    n = 96 if kind in ("spiral","ruffle") else 40
    rows = 25 if kind in ("ribbon","spiral","tri") else 17
    vertices, faces = [], []
    for layer in range(2 if kind in ("ruffle","wrap") else 1):
        base = len(vertices)
        for j in range(rows):
            t = j/(rows-1)
            for i in range(n):
                u = i/(n if closed else n-1)
                a = TAU*u
                if closed:
                    r = .30 + .33*t
                    height = 1.16
                    xoff = 0
                    yscale = .88
                    if kind == "bias": height += .47*math.sin(a+.45)
                    if kind == "train":
                        train = ((1+math.sin(a))/2)**3
                        height = .76+.78*train
                        r += .31*t*train
                    if kind == "offset":
                        xoff = .28*(1-t)
                        yscale = .51+.2*t
                    if kind == "spiral":
                        r += (.035+.085*t)*math.cos(16*a-4*t)
                        a += .48*t
                    if kind == "slit": r = .34+.07*t
                    if kind == "fork": height += .14*math.sin(a)
                    if kind == "petals": height -= .24*math.cos(2*a)**4
                    if kind == "ruffle":
                        r += layer*.085+.064*t*math.sin(12*a+layer*.65)
                        height -= .22*layer
                    p = (xoff+r*math.cos(a),yscale*r*math.sin(a),1.6-t*height)
                else:
                    if kind == "tri": u = u**1.75
                    x = (2*u-1)*(.48+.21*t)
                    y = .22*(2*u-1)**2+.14*t*t
                    height = 1.15
                    top = 1.55
                    if kind == "shoulder":
                        height += .57*u
                        top += .27*u
                        x += .22*t
                    if kind == "swallow": height += .28*math.cos(TAU*u+.8)
                    if kind == "torn": height += .15*math.cos(3*TAU*u)+.08*math.sin(9*TAU*u)
                    if kind == "horseshoe":
                        angle = (u-.5)*math.radians(270)
                        x,y = (.40+.31*t)*math.sin(angle),(.40+.31*t)*math.cos(angle)
                    if kind == "wrap":
                        x = .66*(u-.5)+(.17 if layer else -.17)+.13*t
                        y += .065*layer
                        height += .2*layer*u
                    if kind == "ribbon":
                        x = .055*(2*u-1)+.26*math.sin(TAU*t)
                        y = .20*math.sin(TAU*t+.5)
                        height = 1.49
                    p = (x,y,top-t*height)
                vertices.append(p)
        for j in range(rows-1):
            t = (j+.5)/(rows-1)
            for i in range(n if closed else n-1):
                u = (i+.5)/(n if closed else n-1)
                cuts = {"slit":(.75,), "fork":(.25,.75), "petals":(0,.25,.50,.75),
                        "swallow":(.43,), "torn":(.28,.69)}.get(kind,())
                if t>.45 and any(min(abs(u-c),1-abs(u-c)) < .016+.078*(t-.45) for c in cuts):
                    continue
                if kind=="hole" and ((u-.5)/.22)**2+((t-.48)/.19)**2<1:
                    continue
                a = base+j*n+i
                b = base+j*n+(i+1)%n
                face = (a,a+n,b+n,b) if closed else (a,b,b+n,a+n)
                if kind=="tri" and (i+j)%3:
                    faces.extend(((face[0],face[1],face[2]),(face[0],face[2],face[3])))
                else:
                    faces.append(face)
    used = sorted({index for face in faces for index in face})
    remap = {v:i for i,v in enumerate(used)}
    return [vertices[i] for i in used], [tuple(remap[i] for i in f) for f in faces]


def exercise(cloth, rig, closed, n, levels, index):
    ctx=bpy.context
    st=ctx.scene.auto_skirt_settings
    st.target_armature,st.parent_bone=rig,"测试父骨"
    st.skirt_type="CLOSED" if closed else "OPEN"
    st.prefix=f"Gallery{index:02d}_"
    st.asym_control_chain_count,st.control_levels=n,levels
    ctx.view_layer.objects.active=cloth
    cloth.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    step=bpy.ops.auto_skirt.workflow_next
    assert step()=={"FINISHED"}, "生成控制点"
    ui=addon.unified_generator
    _,chains,_=ui._current_control_chains(ctx)
    assert len(chains)==n and all(len(chain)==levels for chain in chains)
    edit=st if closed else ctx.scene.auto_sheet_settings
    snap=addon._parse_json_dict(edit.control_edit_snapshot if closed else edit.edit_snapshot)
    for chain in chains:
        for obj in chain: obj.select_set(False)
    axis="Y" if closed and index%2==0 else "X"
    dim=0 if axis=="X" else 1
    j=levels//2
    ref=ui.mirror_reference(chains,snap)
    pairs,_=ui._mirror_pairs_from_points(ref,axis)
    first,second=next((a,b) for a,b in pairs if a!=b)
    source=chains[first][j]
    source.select_set(True)
    ctx.view_layer.objects.active=source
    setattr(edit,"control_mirror_"+axis.lower(),True)
    assert ui.highlighted_indices(chains,edit,snap)=={first,second}
    center=sum(Vector(snap[c[j].name][:3])[dim] for c in chains)/n
    source.location+=Vector((.016,.012,.018))
    ctx.view_layer.update()
    ctx.view_layer.update()
    reflected=source.location.copy()
    reflected[dim]=2*center-reflected[dim]
    assert (reflected-chains[second][j].location).length<2e-4, "镜像移动"
    assert step()=={"FINISHED"}
    count=3+index%7
    st.asym_default_bone_count=count
    st.bind_weights=True
    st.create_proxy_mesh=False
    paths,points,(tris,edges)=ui.generation_preview(ctx.scene)
    assert len(paths)==n and len(points)==len(cloth.data.vertices) and tris and edges
    sampled=ui.rail_grid(chains,count)
    assert step()=={"FINISHED"}
    assert len(rig.data.bones)==1+n*count
    for col in range(n):
        for row in range(count):
            bone=rig.data.bones[f"{st.prefix}C{col:02d}_R{row:02d}"]
            assert (rig.matrix_world@bone.head_local-sampled[row][col]).length<2e-4
            assert (rig.matrix_world@bone.tail_local-sampled[row+1][col]).length<2e-4
    assert all(o.hide_get() for c in chains for o in c)
    assert any(m.type=="ARMATURE" and m.object==rig for m in cloth.modifiers)
    assert all(1<=len(v.groups)<=4 and abs(sum(g.weight for g in v.groups)-1)<1e-4 for v in cloth.data.vertices)
    pose=rig.pose.bones[f"{st.prefix}C{first:02d}_R00"]
    pose.rotation_mode="XYZ"
    pose.rotation_euler.x=.25
    ctx.view_layer.update()
    evaluated=cloth.evaluated_get(ctx.evaluated_depsgraph_get())
    delta=max((v.co-evaluated.data.vertices[v.index].co).length for v in cloth.data.vertices)
    pose.rotation_euler.x=0
    ctx.view_layer.update()
    assert delta>1e-4, "姿态变形"
    return {"bones":n*count,"verts":len(points),"mirror":axis,"displacement":round(delta,4)}


def add_case(index, gallery, origin):
    label,kind,closed,n,levels=CASES[index-1]
    ctx=bpy.context
    window=ctx.window
    original=window.scene
    scene=bpy.data.scenes.new("临时裙骨测试")
    window.scene=scene
    label=f"T{index:02d} {label}"
    group=bpy.data.collections.new(label)
    scene.collection.children.link(group)
    try:
        vs,fs=geometry(kind,closed)
        mesh=bpy.data.meshes.new(label)
        mesh.from_pydata(vs,[],fs)
        mesh.update()
        assert not mesh.validate() and min(f.area for f in mesh.polygons)>1e-10
        cloth=bpy.data.objects.new(label,mesh)
        group.objects.link(cloth)
        pos=(origin[0]+((index-1)%4)*2.2,origin[1]+((index-1)//4)*2.6,.02)
        cloth.location=pos
        cloth.show_name=True
        cloth["控制链数"]=n
        cloth["每链控制点数"]=levels
        cloth["裙摆类型"]="环绕式" if closed else "开放式"
        ad=bpy.data.armatures.new(label+" 骨架")
        rig=bpy.data.objects.new(ad.name,ad)
        group.objects.link(rig)
        rig.location=pos
        ad.display_type="STICK"
        ctx.view_layer.objects.active=rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        root=ad.edit_bones.new("测试父骨")
        root.head=(0,0,1.55)
        root.tail=(0,0,1.72)
        bpy.ops.object.mode_set(mode='OBJECT')
        rig.select_set(False)
        stats=exercise(cloth,rig,closed,n,levels,index)
        cloth["测试结果"]="通过"
        return stats
    except Exception:
        cloth["测试结果"]="待检查"
        return {"error":traceback.format_exc()}
    finally:
        if ctx.mode!="OBJECT": bpy.ops.object.mode_set(mode="OBJECT")
        controls=[]
        for child in list(scene.collection.children):
            if child!=group:
                group.children.link(child)
                controls.extend(o for o in child.all_objects if o.type=="EMPTY")
        gallery.children.link(group)
        window.scene=original
        bpy.data.scenes.remove(scene)
        ctx.view_layer.update()
        for o in controls:
            if o.name in ctx.view_layer.objects: o.hide_set(True)
