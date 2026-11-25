# -*- coding: utf-8 -*-
"""Two-bone cylinder study for Maya 2024. Run show() in Maya, or --check in Python.
Video channel values set the scale of the labelled frame-pair excursions. Units are bind-cylinder radii, angles are degrees.
"""
from math import atan2, cos, sin, tan, exp, hypot, sqrt, pi, radians, degrees, dist, isfinite

HALF = 6.0
REACH = 5.1
HALF_RINGS = 9
SIDES = 24
CHANNELS = (
    "Center Slide In", "Center Slide Out", "Slide In Prev", "Slide In Next",
    "Slide Out Prev", "Slide Out Next", "Slide Ortho In Prev", "Slide Ortho In Next",
    "Slide Ortho Out Prev", "Slide Ortho Out Next", "Bulge",
)
PAIRS = (
    ("Center In    404 / 409", (0,)), ("Center Out    429 / 438", (1,)),
    ("Slide In    650 / 654", (2, 3)), ("Slide Out    673 / 676", (4, 5)),
    ("Ortho In    695 / 697", (6, 7)), ("Ortho Out    714 / 716", (8, 9)),
    ("Bulge", (10,)),
)
VIDEO = (0.33,0.29,0.19,0.19,0.21,0.21,0.10,0.10,0.10,0.10,0.10)
ZERO = (0.0,) * len(CHANNELS)
RING_STEP = HALF/HALF_RINGS


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t*t*(3.0 - 2.0*t)


def fade(distance, reach):
    return 1.0 - smooth(distance/reach)


def shoulder(distance, peak, end):
    """Zero displacement at the center and at the rigid end."""
    if distance < peak:
        return smooth(distance/peak)
    return fade(distance-peak, end-peak)


IN_TRAVEL = (0.37674, 0.30413, 4.9902)


def skin_weight(station):
    return smooth(0.5 + station/5.2)


def mix(a, b, t):
    return a + (b-a)*t


class Elbow:
    """Inner and outer longitudinal rails define a common section for each ring."""
    def __init__(self, bend=118.0, twist=0.0, direction=0.0, values=ZERO, avoid_overlap=False):
        if len(values) != len(CHANNELS) or not all(isfinite(v) for v in (bend, twist, direction, *values)):
            raise ValueError("Invalid pose or compensation values")
        if abs(bend) > 150.00001:
            raise ValueError("Supported bend range is -150 to 150 degrees")
        plane = radians(direction) + (pi if bend < 0.0 else 0.0)
        beta = radians(abs(bend))
        self.cp, self.sp = cos(plane), sin(plane)
        self.cb, self.sb = cos(beta), sin(beta)
        self.h = beta*0.5
        self.ch, self.sh = cos(self.h), sin(self.h)
        self.twist = radians(twist)
        self.values = tuple(v/unit for v,unit in zip(values,VIDEO))
        self.avoid_overlap = avoid_overlap
        self.edited = any(values)
        self.edit_scale = 1.0
        self.response = min(1.15, self.sh*self.sh/0.75)
        self.fraction = abs(bend)/150.0
        compression = (1-self.ch)/(1-cos(radians(75)))
        self.tip_y = 1-0.355*compression**3
        self.tip_x = -self.tip_y*tan(self.h)
        self.inner_reach = 3.15+1.90*compression
        self.tangent_x = -0.22*tan(self.h)
        # Clearance attenuates secondary inside edits independently of the default shape.
        raw = 1.12/self.ch-0.12
        excess = max(0.0,raw-1.95)
        self.inner_clearance = 1.0/(1.0+3.0*(excess+0.60*exp(-excess/0.60)-0.60))
        if not avoid_overlap:
            self.inner_clearance = 1.0
        if beta and any(abs(v)>1.000001 for v in self.values):
            self.edit_scale = self._safe_edit_scale()

    def _safe_edit_scale(self):
        # Beyond the measured excursion, shorten the proposed displacement only
        # if a triangle would turn through its default tangent plane. This guards
        # local inversion, not contact between the two limbs.
        bind,faces = _CHECK_MESH
        bind = [(x,self.cp*y-self.sp*z,self.sp*y+self.cp*z) for x,y,z in bind]
        edited = self.mesh(bind)
        self.edited = False
        try:
            neutral = self.mesh(bind)
        finally:
            self.edited = True

        def edges(points,a,b):
            return tuple(x-y for x,y in zip(points[b],points[a]))

        def cross(a,b):
            x,y,z = a
            u,v,w = b
            return y*w-z*v,z*u-x*w,x*v-y*u

        def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

        limit = 1.0
        for face in faces:
            a = face[0]
            for b,c in zip(face[1:],face[2:]):
                u,v = edges(neutral,a,b),edges(neutral,a,c)
                n = cross(u,v)
                e,f = edges(edited,a,b),edges(edited,a,c)
                du = tuple(x-y for x,y in zip(e,u))
                dv = tuple(x-y for x,y in zip(f,v))
                # Signed area of the interpolated triangle is quadratic in the gain.
                A = dot(n,cross(du,dv))
                B = dot(n,cross(du,v))+dot(n,cross(u,dv))
                C = 0.95*dot(n,n)
                end = limit
                if A>0 and 0 < -B/(2*A) < end:
                    end = -B/(2*A)
                if (A*end+B)*end+C < 0:
                    lo,hi = 0.0,end
                    for _ in range(24):
                        t = (lo+hi)*0.5
                        if (A*t+B)*t+C>0:
                            lo=t
                        else:
                            hi=t
                    limit=lo
        return limit

    def _rail(self, station, offsets, pole):
        u = abs(station)
        row = min(int(u/RING_STEP),HALF_RINGS-1)
        t = min(1.0,u/RING_STEP-row)
        # Linear interpolation on a mesh edge also handles off-ring Slide samples.
        dx = mix(offsets[row],offsets[row+1],t)
        x = -u+self.fraction*dx
        tip = self.tip_x if pole>0 else self.tangent_x+1.22*self.sh
        if pole>0:
            # Anchor the distribution on ring zero itself.  A convex spatial
            # falloff orders the gaps on both limbs and vanishes at the end.
            x = -u+tip*max(0.0,1.0-u/HALF)**2
        elif u<RING_STEP:
            x += (1.0-u/RING_STEP)*(tip-self.fraction*offsets[0])
        # Evaluate the accepted pose's actual silhouette at this material ring.
        if pole>0:
            a = min(1.0,max(0.0,(tip-x)/(self.inner_reach+tip)))
            remainder = (1.0-a)**2*(1.0+0.87*a) if self.avoid_overlap else (1.0-a)**4
            y = 1.0-(1.0-self.tip_y)*remainder
        else:
            run = max(0.0,x-self.tangent_x)
            y = -1.0+1.22-sqrt(max(0.0,1.22**2-run**2))
        if station > 0.0:
            return -self.cb*x-self.sb*y, -self.sb*x+self.cb*y
        return x, y

    def section(self, station):
        ox, oy = self._rail(station, OUTER,-1)
        ix, iy = self._rail(station, INNER_CLEAR if self.avoid_overlap else INNER,1)
        return ox, oy, 0.5*(ix-ox), 0.5*(iy-oy)

    def _point(self, s, y, z, sec):
        z0 = z
        ox, oy, rx, ry = sec
        x0, y0 = ox+(1+y)*rx, oy+(1+y)*ry
        u = abs(s)
        if not self.edited or u >= REACH or self.response == 0.0:
            return x0, y0, z
        ci, co, ip, inn, op, on, nip, nin, nop, non, bulge = self.values
        wi = self.response*max(y, 0.0)**1.7
        wo = self.response*max(-y, 0.0)**1.7
        branch = -1.0 if s < 0.0 else 1.0
        inner, outer = (ip, op) if s < 0.0 else (inn, on)
        lift_i, lift_o = (nip, nop) if s < 0.0 else (nin, non)
        gain, peak, end = IN_TRAVEL
        travel = branch*(gain*wi*inner*shoulder(u,peak,end)
                         - 0.48*wo*outer*shoulder(u,2.05,4.8))
        if travel:
            ax, ay, sx, sy = self.section(s+travel)
            x, py = x0+ax+(1+y)*sx-(ox+(1+y)*rx), y0+ay+(1+y)*sy-(oy+(1+y)*ry)
        else:
            x, py = x0, y0
        # A unilateral Slide Out contributes half the paired excursion at the apex.
        outer_at_crease = mix(op, on, smooth(0.5+s/1.6))
        shift = (0.62*self.inner_clearance*wi*ci*fade(u,3.6) - 0.392*wo*co*fade(u,3.55)
                 - 0.251*wo*outer_at_crease*fade(u,2.75))
        x -= self.sh*shift
        py += self.ch*shift
        lift = (0.186*self.inner_clearance*wi*lift_i*shoulder(u,1.45,4.8)
                + 0.120*wo*lift_o*shoulder(u,1.35,4.8))
        if lift:
            # Tangent of a material meridian at constant circumferential coordinate.
            left, right = self.section(s-0.002), self.section(s+0.002)
            tx = right[0]-left[0] + (1+y)*(right[2]-left[2])
            ty = right[1]-left[1] + (1+y)*(right[3]-left[3])
            nx, ny, nz = -y*ty, y*tx, z*(ry*tx-rx*ty)
            norm = sqrt(nx*nx+ny*ny+nz*nz)
            if norm > 1e-12:
                x += lift*nx/norm
                py += lift*ny/norm
                z += lift*nz/norm
        z += 0.42*bulge*self.response*fade(u,3.65)*z**3
        # Keep combined shoulder edits from passing through the fold bisector.
        if self.avoid_overlap and u > 1e-8:
            room = branch*(self.ch*x0+self.sh*y0)
            gap = branch*(self.ch*x+self.sh*py)
            minimum = max(0.0, 0.12*room)
            if gap < minimum:
                delta = branch*(minimum-gap)
                x += self.ch*delta
                py += self.sh*delta
        return mix(x0,x,self.edit_scale),mix(y0,py,self.edit_scale),mix(z0,z,self.edit_scale)

    def point(self, p, section=None):
        s, by, bz = p
        angle = self.twist*skin_weight(s)
        ct, st = cos(angle), sin(angle)
        cy, cz = by*self.cp+bz*self.sp, -by*self.sp+bz*self.cp
        y, z = cy*ct-cz*st, cy*st+cz*ct
        x, y, z = self._point(s, y, z, section or self.section(s))
        return x, self.cp*y-self.sp*z, self.sp*y+self.cp*z

    def mesh(self, bind):
        sections = {s:self.section(s) for s in dict.fromkeys(p[0] for p in bind)}
        return [self.point(p, sections[p[0]]) for p in bind]


# Limb-local displacements sampled from the approved 150-degree default.
OUTER = (0.3573783304, 0.5395253322, 0.4023315278, 0.242476591, 0.1149518661, 0.0305340904, 0.0257631388, 0.0152670452, 0.0047709516, 0.0)
INNER_CLEAR = (-2.4071727709, -2.1190059439, -1.743053118, -1.3217063759, -0.8973578005, -0.5123994744, -0.2092234804, -0.1549803559, -0.0542431246, -0.0)
INNER = (-2.4071727709, -2.0916385862, -1.7084857196, -1.2864088566, -0.86373149, -0.481721735, -0.1748798316, -0.129540616, -0.0453392156, -0.0)


def cylinder():
    points, faces = [], []
    rings = 2*HALF_RINGS+1
    for row in range(rings):
        s = HALF*(row-HALF_RINGS)/HALF_RINGS
        for j in range(SIDES):
            a = 2*pi*j/SIDES
            points.append((s, cos(a), sin(a)))
    for row in range(rings-1):
        for j in range(SIDES):
            a, b = row*SIDES+j, row*SIDES+(j+1)%SIDES
            faces.append((a,b,b+SIDES,a+SIDES))
    start, end = len(points), len(points)+1
    points.extend(((-HALF,0.0,0.0), (HALF,0.0,0.0)))
    last = (rings-1)*SIDES
    for j in range(SIDES):
        k = (j+1)%SIDES
        faces.extend(((start,k,j), (end,last+j,last+k)))
    return points, faces


_CHECK_MESH = cylinder()


# Fitted viewing geometry for the labelled cylinder shots. Yaw, elevation, pixels
# per radius, distance and bend are estimates; before levels are visual offsets.
SHOTS = (
    (-34.7,22.4,51.6,29.1,117.6), (-1.1,16.2,50.9,44.9,117.9),
    (-5.3,8.5,48.1,23.4,125.6), (-5.7,9.6,48.1,23.3,125.6),
    (-28.7,6.8,48.2,23.5,126.0), (-14.1,11.8,47.3,20.2,125.4),
    (-30,25,48,25,124),
)
# Each shot starts from a different existing shape. These are fitted starting
# levels in this study's units, not values read from the original Channel Box.
BEFORE = ((-0.667,-1.60), (0.0,-1.29), (-0.718,-0.19), (-0.718,-0.19),
          (-0.299,0.24), (-0.299,0.17), (0.0,0.0))


def example_values(index, after):
    values = list(ZERO)
    values[:2] = [x*v for x,v in zip(BEFORE[index],VIDEO)]
    for channel in PAIRS[index][1]:
        values[channel] = 0.0
    if after:
        for channel in PAIRS[index][1]:
            values[channel] = VIDEO[channel]
    return values


def joint_quaternion(bend, twist, direction):
    h, t, d = radians(bend)*0.5, radians(twist)*0.5, radians(direction)
    sy, sz = -sin(d)*sin(h), cos(d)*sin(h)
    return (cos(h)*sin(t), sy*cos(t)+sz*sin(t),
            sz*cos(t)-sy*sin(t), cos(h)*cos(t))


def read_rotation(quat, old_bend, old_direction):
    """Resolve the driver's swing and axial twist, retaining a nearby signed pose."""
    x, y, z, w = quat
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    axial, swing = hypot(w, x), hypot(y, z)
    twist = degrees(2.0*atan2(x, w))
    if swing < 1e-9 or axial < 1e-9:
        return (0.0 if swing < 1e-9 else 150.0), twist, old_direction
    bend = min(150.0, degrees(2.0*atan2(swing, axial)))
    direction = degrees(atan2(z*x-y*w, z*w+y*x))
    delta = (direction-old_direction+180.0) % 360.0 - 180.0
    if abs(delta) > 90.0 or (abs(abs(delta)-90.0) < 1e-8 and old_bend < 0):
        return -bend, twist, (direction % 360.0)-180.0
    return bend, twist, direction


def rig_weights(w, three_bone):
    if not three_bone:
        return (1-w, w)
    helper = 1.2 * min(w, 1-w)
    return (1-w-helper/2, w-helper/2, helper)


def rotation_matrix(q):
    import numpy as np
    x, y, z, w = np.moveaxis(np.asarray(q), -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(x.shape+(3,3))


def center_response(weights, rotations):
    """Derivative of posed points with respect to bind centers in object space."""
    import numpy as np
    q = np.asarray(rotations, dtype=float)
    blend = np.zeros((len(weights), 4))
    for w, rotation in zip(weights.T, q):
        sign = np.where((blend*rotation).sum(axis=1)<0, -1, 1)
        blend += (w*sign)[:,None]*rotation
    blend /= np.linalg.norm(blend,axis=1)[:,None]
    return np.einsum("nk,kij->nij",weights,rotation_matrix(q))-rotation_matrix(blend)


def solve_centers(response, desired, center_limit=None):
    """Damped minimum-norm center displacement in bind-object space."""
    import numpy as np
    u, singular, vt = np.linalg.svd(response, full_matrices=False)
    components = u.swapaxes(1, 2) @ desired[:, :, None]
    gain = singular / (singular**2 + 0.015**2)
    delta = (vt.swapaxes(1, 2) @ (gain[:, :, None] * components))[:, :, 0]
    if center_limit is not None:
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        delta *= np.minimum(1.0, center_limit / np.maximum(length, 1e-12))
    return delta


def show(three_bone=True, center_limit=None):
    """Open a private preview window with native Maya skinCluster references."""
    import maya.cmds as mc
    from maya.api import OpenMaya as om
    if center_limit is not None and (not isfinite(center_limit) or center_limit <= 0):
        raise ValueError("center_limit must be finite and positive")
    win = "OCDDenseCenterWindow" + ("Three" if three_bone else "Two")
    if mc.window(win, exists=True):
        mc.showWindow(win)
        return win
    if "optimizedSkinCluster" not in (mc.allNodeTypes() or []):
        try:
            mc.loadPlugin("optimizedSkinningPlugin",quiet=True)
        except RuntimeError as exc:
            raise RuntimeError("请先在 Maya 插件管理器加载 optimizedSkinningPlugin.mll，再打开六模型对比。") from exc
    selection = mc.ls(selection=True,long=True) or []
    win = mc.window(win,title="OCD 圆柱验证  |  六模型正视对比",widthHeight=(1200,690))
    group = mc.createNode("transform",name="OCD_referencePreview#")
    history = []
    group_id = mc.ls(group,uuid=True)[0]

    def linear(cm):
        return om.MDistance(cm,om.MDistance.kCentimeters).asUnits(om.MDistance.uiUnit())

    def clean(*_):
        for name,uid in [(group,group_id)] + history:
            if mc.objExists(name) and mc.ls(name,uuid=True) == [uid]:
                mc.delete(name)

    mc.scriptJob(uiDeleted=[win,clean],runOnce=True)
    try:
        bind, faces = cylinder()
        counts = [len(f) for f in faces]
        indices = [j for f in faces for j in f]
        points = [om.MPoint(*p) for p in bind]
        import numpy as np
        joints, rigs, helpers = [], [], []
        weights = np.array([rig_weights(skin_weight(p[0]), three_bone) for p in bind])
        layout = (("Classic_LBS",-7,16),("DQ_Skin",7,16),
                  ("CoR",-7,0),("DQS_Com",7,0),("CoR_CurveGuided",7,-16),("OCD",-7,-16))
        for i,(label,offset,row_y) in enumerate(layout):
            base = mc.createNode("transform",name="ref_"+label+"#",parent=group)
            rigs.append(base)
            mc.setAttr(base+".translateX",linear(offset))
            mc.setAttr(base+".translateY",linear(row_y))
            # Evaluate repository deformers at the origin. The visible shape receives
            # their object-space output, independent of the grid cell translation.
            compute = base
            if label in ("CoR","DQS_Com","CoR_CurveGuided"):
                compute = mc.createNode("transform",name=label+"_compute#",parent=group)
                mc.setAttr(compute+".visibility",False)
            geo = mc.createNode("transform",name=label+"_geo#",parent=compute)
            sel = om.MSelectionList()
            sel.add(geo)
            obj = om.MFnMesh().create(points,counts,indices,parent=sel.getDependNode(0))
            shape = mc.rename(om.MFnDagNode(obj).fullPathName(),label+"Shape#")
            mc.sets(geo,edit=True,forceElement="initialShadingGroup")
            prev = mc.createNode("joint",name=label+"_prev#",parent=compute)
            elbow = mc.createNode("joint",name=label+"_elbow#",parent=prev)
            tip = mc.createNode("joint",name=label+"_tip#",parent=elbow)
            for j,dx in ((prev,-HALF),(elbow,HALF),(tip,HALF)):
                mc.setAttr(j+".translateX",linear(dx))
                mc.setAttr(j+".radius",0.09)
            joints.append(elbow)
            bones = [prev, elbow]
            if three_bone:
                helper = mc.createNode("joint",name=label+"_helper#",parent=prev)
                mc.setAttr(helper+".translateX",linear(HALF))
                bones.append(helper)
                helper_sel = om.MSelectionList()
                helper_sel.add(helper)
                helpers.append(om.MFnTransform(helper_sel.getDagPath(0)))
            if i < 2:
                existing_poses = set(mc.ls(type="dagPose") or [])
                skin = mc.skinCluster(*(bones+[geo]),toSelectedBones=True,skinMethod=i,
                                      maximumInfluences=len(bones),normalizeWeights=1,name=label+"_skin#")[0]
                new_poses = set(mc.ls(type="dagPose") or [])-existing_poses
                for node in [skin]+sorted(new_poses):
                    history.append((node,mc.ls(node,uuid=True)[0]))
                for row in range(2*HALF_RINGS+1):
                    first = row*SIDES
                    w = skin_weight(bind[first][0])
                    mc.skinPercent(skin,"%s.vtx[%d:%d]" % (geo,first,first+SIDES-1),
                                   transformValue=list(zip(bones,rig_weights(w,three_bone))))
                for ix,w in ((len(bind)-2,0.0),(len(bind)-1,1.0)):
                    mc.skinPercent(skin,"%s.vtx[%d]" % (geo,ix),transformValue=list(zip(bones,rig_weights(w,three_bone))))
            elif label in ("CoR","DQS_Com","CoR_CurveGuided"):
                skin = mc.deformer(geo,type="optimizedSkinCluster",name=label+"_skin#")[0]
                history.append((skin,mc.ls(skin,uuid=True)[0]))
                for k,joint in enumerate(bones):
                    mc.connectAttr(joint+".worldMatrix[0]",skin+".matrix[%d]" % k)
                    inverse = mc.getAttr(joint+".worldInverseMatrix[0]")
                    mc.setAttr(skin+".bindPreMatrix[%d]" % k,*inverse,type="matrix")
                for vertex,(station,_,_) in enumerate(bind):
                    w = skin_weight(station)
                    mc.setAttr("%s.weightList[%d].weights[0:%d]" % (skin,vertex,len(bones)-1),
                               *weights[vertex],size=len(bones))
                if label=="CoR_CurveGuided" and not mc.attributeQuery("corFitOffsets",node=skin,exists=True):
                    raise RuntimeError("请加载本测试分支新编译的插件。")
                mc.setAttr(skin+".algorithmType",{"CoR":0,"DQS_Com":2,"CoR_CurveGuided":8}[label])
                if label=="DQS_Com":
                    mc.setAttr(skin+".dqsCompStrength",1.0)
                result = mc.listRelatives(geo,shapes=True,noIntermediate=True,fullPath=True)[0]
                if label=="CoR":
                    ref_sel = om.MSelectionList()
                    ref_sel.add(result)
                    reference = om.MFnMesh(ref_sel.getDagPath(0))
                if label=="CoR_CurveGuided":
                    node_sel = om.MSelectionList()
                    node_sel.add(skin)
                    plug = om.MFnDependencyNode(node_sel.getDependNode(0)).findPlug("corFitOffsets",False)
                    result_sel = om.MSelectionList()
                    result_sel.add(result)
                    guided_mesh = om.MFnMesh(result_sel.getDagPath(0))
                display = mc.createNode("transform",name=label+"_view#",parent=base)
                proxy = mc.createNode("mesh",name=label+"_viewShape#",parent=display)
                mc.connectAttr(result+".outMesh",proxy+".inMesh")
                mc.sets(proxy,edit=True,forceElement="initialShadingGroup")
            else:
                sel = om.MSelectionList()
                sel.add(shape)
                mesh = om.MFnMesh(sel.getDagPath(0))

        for j in joints[:-1]:
            mc.connectAttr(joints[-1]+".rotate",j+".rotate")
        sel = om.MSelectionList()
        sel.add(joints[-1])
        driver = om.MFnTransform(sel.getDagPath(0))
        cam_xf, cam = mc.camera(name="OCD_referenceCamera#",orthographic=True)
        cam_xf = mc.parent(cam_xf,group)[0]
        panes = mc.paneLayout(parent=win,configuration="vertical2",paneSize=[(1,66,100),(2,34,100)])
        panel = mc.modelPanel(parent=panes,label="OCD reference",camera=cam_xf,menuBarVisible=False)
        editor = mc.modelPanel(panel,query=True,modelEditor=True)
        mc.modelEditor(editor,e=True,displayAppearance="smoothShaded",grid=False,
                       displayTextures=False,wireframeOnShaded=True,displayLights="default")
        mc.select(clear=True)
        mc.isolateSelect(panel,state=True)
        for rig in rigs:
            mc.isolateSelect(panel,addDagObject=rig)
        scroll = mc.scrollLayout(parent=panes,childResizable=True)
        col = mc.columnLayout(parent=scroll,adjustableColumn=True,rowSpacing=3)
        mc.text(parent=col,label="正视图　左列 / 右列\n上排　LBS / DQS\n中排　CoR / DQS_Com\n下排　OCD / CoR 动态曲线中心")
        mc.text(parent=col,label="动态中心拟合 · 全截面 OCD 目标")
        handles = []

        def slider(label,low,high,value):
            return mc.floatSliderGrp(parent=col,label=label,field=True,minValue=low,maxValue=high,
                                     fieldMinValue=low,fieldMaxValue=high,value=value,precision=2,
                                     columnWidth3=(127,52,105),adjustableColumn=3,
                                     dragCommand=lambda *_:redraw(),changeCommand=lambda *_:redraw())

        def set_values(items,values):
            for item,v in zip(items,values):
                mc.floatSliderGrp(item,e=True,value=v)

        def get_values(items):
            return [mc.floatSliderGrp(item,q=True,value=True) for item in items]

        last_rotation = [None]

        def redraw(*_):
            bend,twist,direction = get_values(motion)
            pose = Elbow(bend,twist,direction,get_values(handles),mc.checkBox(avoid,q=True,value=True))
            q = joint_quaternion(bend,twist,direction)
            if q != last_rotation[0]:
                last_rotation[0] = q
                driver.setRotation(om.MQuaternion(*q),om.MSpace.kTransform)
                rotations = [(0,0,0,1), q]
                if helpers:
                    half = joint_quaternion(bend*0.5,twist*0.35,direction)
                    for helper in helpers:
                        helper.setRotation(om.MQuaternion(*half),om.MSpace.kTransform)
                    rotations.append(half)
                original = np.asarray([(p.x,p.y,p.z) for p in reference.getPoints(om.MSpace.kObject)])
                base_pose = Elbow(bend,twist,direction)
                desired = np.asarray(base_pose.mesh(bind)) - original
                delta = solve_centers(center_response(weights,rotations),desired,center_limit)
                plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray(delta.ravel().tolist())))
            mesh.setPoints([om.MPoint(*p) for p in pose.mesh(bind)],om.MSpace.kObject)

        motion = [slider("弯曲",-150,150,118), slider("扭转",-180,180,0), slider("弯曲方向",-180,180,0)]

        def driver_changed(*_):
            r = driver.rotation(om.MSpace.kTransform,True)
            q = (r.x,r.y,r.z,r.w)
            if last_rotation[0] and abs(sum(a*b for a,b in zip(q,last_rotation[0]))) > 1.0-1e-12:
                return
            bend,_,direction = get_values(motion)
            set_values(motion,read_rotation(q,bend,direction))
            last_rotation[0] = None
            redraw()

        avoid = mc.checkBox(parent=col,label="旧版内侧避让",value=False,changeCommand=redraw)

        def set_pose(b,t=0,d=0):
            set_values(motion,(b,t,d))
            redraw()
        row = mc.rowLayout(parent=col,numberOfColumns=5)
        for name,args in (("伸直",(0,)),("120°",(120,)),("片头",(118,)),("深弯",(145,)),("弯＋扭",(105,70,35))):
            mc.button(parent=row,label=name,command=lambda *_,a=args:set_pose(*a))

        def camera_pose(yaw,elev,center,depth,width,perspective=False):
            # Maya's local camera looks down -Z. Its Y axis is screen up.
            rot = om.MEulerRotation(-radians(elev),radians(yaw),0.0)
            back = om.MVector(0,0,depth).rotateBy(rot.asQuaternion())
            mc.xform(cam_xf,objectSpace=True,translation=[linear(v) for v in
                     (center[0]+back.x,center[1]+back.y,center[2]+back.z)])
            cam_sel = om.MSelectionList()
            cam_sel.add(cam_xf)
            om.MFnTransform(cam_sel.getDagPath(0)).setRotation(rot,om.MSpace.kTransform)
            mc.setAttr(cam+".orthographic",not perspective)
            if perspective:
                aperture = mc.getAttr(cam+".horizontalFilmAperture")*25.4
                mc.setAttr(cam+".focalLength",aperture*depth/width)
                mc.setAttr(cam+".filmFit",1)   # horizontal fit
            else:
                mc.setAttr(cam+".orthographicWidth",linear(width))
            mc.setAttr(cam+".centerOfInterest",linear(depth))

        def frame_view(*_):
            focus = mc.checkBox(closeup,q=True,value=True)
            oblique = mc.checkBox(sideview,q=True,value=True)
            yaw,elev = (-22,12) if oblique else (0,0)
            if focus:
                camera_pose(yaw,elev,(-7.8,-13.9,0),40,16)
            else:
                width = mc.control(editor,q=True,width=True) if mc.control(editor,exists=True) else 0
                height = mc.control(editor,q=True,height=True) if mc.control(editor,exists=True) else 0
                aspect = width/height if min(width,height)>10 else 1.35
                camera_pose(yaw,elev,(0,0,0),70,max(31,48*aspect))
        row = mc.rowLayout(parent=col,numberOfColumns=3)
        closeup = mc.checkBox(parent=row,label="OCD 特写",changeCommand=frame_view)
        sideview = mc.checkBox(parent=row,label="斜视",changeCommand=frame_view)
        mc.checkBox(parent=row,label="线框",value=True,
                    changeCommand=lambda value:mc.modelEditor(editor,e=True,wireframeOnShaded=value))
        mc.separator(parent=col,height=8,style="in")
        choice = mc.optionMenu(parent=col,label="参考帧")
        for label,_ in PAIRS:
            mc.menuItem(parent=choice,label=label)

        def shot(*_):
            i = mc.optionMenu(choice,q=True,select=True)-1
            yaw,elev,scale,depth,bend = SHOTS[i]
            mc.checkBox(closeup,e=True,value=True)
            mc.checkBox(sideview,e=True,value=bool(yaw or elev))
            # Recenter the cropped movie on the elbow, preserving shot projection.
            rot = om.MEulerRotation(-radians(elev),radians(yaw),0.0).asQuaternion()
            offset = om.MVector(18/scale,(37 if i < 2 else 36)/scale,0).rotateBy(rot)
            camera_pose(yaw,elev,(-7+offset.x,-16+offset.y,offset.z),depth,854/scale,True)
            set_values(motion,(bend,0,0))
            example(False)
        def example(after):
            i = mc.optionMenu(choice,q=True,select=True)-1
            set_values(handles,example_values(i,after))
            redraw()
        row = mc.rowLayout(parent=col,numberOfColumns=4)
        mc.button(parent=row,label="参考姿态／机位",command=shot)
        mc.button(parent=row,label="Before",command=lambda *_:example(False))
        mc.button(parent=row,label="After",command=lambda *_:example(True))
        def neutral(*_):
            set_values(handles,ZERO)
            redraw()
        mc.button(parent=row,label="全部归零",command=neutral)
        def report(*_):
            bend,twist,direction = get_values(motion)
            target = np.asarray(Elbow(bend,twist,direction).mesh(bind))
            active = (weights>0).sum(axis=1)>1
            before,after = [np.linalg.norm(np.asarray([(p.x,p.y,p.z) for p in fn.getPoints(
                om.MSpace.kObject)])-target,axis=1) for fn in (reference,guided_mesh)]
            print("CoR -> rail CoR | active RMS %.5f -> %.5f" %
                  (np.sqrt(np.mean(before[active]**2)),np.sqrt(np.mean(after[active]**2))))
            print("All vertices RMS %.5f -> %.5f | single-influence max error %.5f" %
                  (np.sqrt(np.mean(before**2)),np.sqrt(np.mean(after**2)),after[~active].max()))
            # Find the actual inner/outer meridians in the signed bend plane.
            azimuth = radians(direction + (180 if bend<0 else 0))
            for ring in range(-2,3):
                start = (HALF_RINGS+ring)*SIDES
                phase = azimuth - radians(twist)*skin_weight(bind[start][0])
                inner = round(phase*SIDES/(2*pi))%SIDES
                i,o = start+inner,start+(inner+SIDES//2)%SIDES
                print("ring %+d | inner %.5f -> %.5f | outer %.5f -> %.5f" %
                      (ring,before[i],after[i],before[o],after[o]))
        mc.button(parent=col,label="打印当前姿态误差（默认 OCD）",command=report)
        handles.extend(slider(c,-1.0,1.0,0) for c in CHANNELS)
        def load_video(*_):
            set_values(handles,VIDEO)
            redraw()
        mc.button(parent=col,label="视频参数（补偿）",command=load_video)
        mc.text(parent=col,label="Before / After 保持当前骨姿与机位。关闭窗口清理本次预览。")
        frame_view()
        redraw()
        mc.scriptJob(parent=win,attributeChange=[joints[-1]+".rotate",driver_changed],allChildren=True)
        mc.showWindow(win)
        frame_view()
    except Exception:
        if mc.window(win,exists=True):
            mc.deleteUI(win,window=True)
        raise
    finally:
        mc.select(clear=True)
        keep = [o for o in selection if mc.objExists(o)]
        if keep:
            mc.select(keep,replace=True)
    return win


def check_shots(bind, faces):
    """Sample the fitted wireframe in original screenshot pixels (870 x 585)."""
    edges = {tuple(sorted((a,b))) for f in faces for a,b in zip(f,f[1:]+f[:1])}

    def project(index, after):
        yaw,elev,scale,depth,bend = SHOTS[index]
        cy,sy = cos(radians(yaw)),sin(radians(yaw))
        ce,se = cos(radians(elev)),sin(radians(elev))
        pivot_y = 309 if index < 2 else 308
        projected = []
        for x,y,z in Elbow(bend,values=example_values(index,after),avoid_overlap=True).mesh(bind):
            across,back = cy*x-sy*z,sy*x+cy*z
            up = ce*y-se*back
            perspective = 1.0-(se*y+ce*back)/depth
            projected.append((435+(scale*across-18)/perspective,
                              272+(-scale*up+pivot_y-272)/perspective))
        return projected

    def crossings(image, segments, axis, scan):
        hits = []
        for a,b in segments:
            p,q = image[a],image[b]
            if (p[axis] <= scan) != (q[axis] <= scan):
                t = (scan-p[axis])/(q[axis]-p[axis])
                hits.append(mix(p[1-axis],q[1-axis],t))
        return hits

    measurements = []
    for index in range(6):
        before,after = project(index,False),project(index,True)
        if index == 0:
            # The visible fold changes shape. Local patch matches in its center
            # give a range of translations rather than an exact vertex pairing.
            vertex = HALF_RINGS*SIDES
            for axis,target,tolerance in ((0,-24,4),(1,-9,5)):
                delta = after[vertex][axis]-before[vertex][axis]
                if abs(delta-target)>tolerance:
                    raise AssertionError(("inner fold",target,delta))
                measurements.append((PAIRS[index][0],target,round(delta,2)))
            continue
        elif index == 2:
            samples = []
            for row,target in zip((4,5,6),(-3,-5,-7)):
                first = row*SIDES
                visible = [(first+j,first+j+1) for j in range(SIDES//2)]
                samples.append((visible,1,278,target,1.2))
        elif index == 4:
            inside = [(row*SIDES,(row+1)*SIDES) for row in range(HALF_RINGS)]
            samples = [(inside,0,x,dy,1.5) for x,dy in ((259,-4),(280,-6),(300,-7))]
        elif index == 5:
            samples = [(edges,0,x,y,1.7) for x,y in zip((260,295,320,344,367),(3,4,4,4,6))]
        else:
            samples = [(edges,1,312,18 if index==1 else 12,1.5)]
        for segments,axis,scan,reference,tolerance in samples:
            delta = max(crossings(after,segments,axis,scan))-max(crossings(before,segments,axis,scan))
            if abs(delta-reference) > tolerance:
                raise AssertionError((PAIRS[index][0],reference,delta))
            measurements.append((PAIRS[index][0],reference,round(delta,2)))
    return measurements


def self_test():
    """Pure geometry checks. Run with --check or call from Maya's Script Editor."""
    bind,faces = cylinder()
    full = (1.0,)*len(CHANNELS)
    def near(a,b,tol=1e-8):
        if dist(a,b) > tol:
            raise AssertionError((a,b))

    def rotate_x(point, angle):
        x,y,z = point
        c,s = cos(radians(angle)),sin(radians(angle))
        return x,c*y-s*z,s*y+c*z

    def rotate_z(point, angle):
        x,y,z = point
        c,s = cos(radians(angle)),sin(radians(angle))
        return c*x-s*y,s*x+c*y,z

    for p,q in zip(bind,Elbow(0,values=full).mesh(bind)):
        near(p,q)
    for angle in (-150,-90,0,90,118,145,150):
        for values in (ZERO, full, example_values(0,False)):
            pose = Elbow(angle,35,0,values)
            turned = Elbow(angle,35,37,values)
            for p,q in zip(bind,pose.mesh(bind)):
                if not all(isfinite(v) for v in q):
                    raise AssertionError((angle,p,q))
                if p[0] == -HALF:
                    near(p,q)
                if p[0] == HALF:
                    near(q,rotate_z(rotate_x(p,35),angle))
                near(turned.point(rotate_x(p,37)),rotate_x(q,37))
        # In an untwisted bend each section projects to a straight line in the bend plane.
        neutral = Elbow(angle,avoid_overlap=True)
        for row in range(2*HALF_RINGS+1):
            rr = neutral.mesh(bind[row*SIDES:(row+1)*SIDES])
            a,b = rr[0],rr[SIDES//2]
            for x,y,_ in rr:
                if abs((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])) > 1e-9:
                    raise AssertionError(("section",angle,row))

    step = RING_STEP
    for avoid in (True,False):
        deep = Elbow(150,avoid_overlap=avoid)
        for sample in range(601):
            angle = sample/4
            pose = Elbow(angle,avoid_overlap=avoid)
            for branch in (-1,1):
                pts = [pose.point((branch*i*step,1,0)) for i in range(HALF_RINGS+1)]
                gaps = [dist(a,b) for a,b in zip(pts,pts[1:])]
                if angle and any(a>=b for a,b in zip(gaps,gaps[1:])):
                    raise AssertionError(("inside gap order",angle,branch,gaps))
            for pole in (-1,1):
                previous = float("inf")
                for row in range(1,HALF_RINGS+1):
                    u = row*step
                    rest = (-u,pole,0)
                    target = deep.point(rest)
                    point = pose.point(rest)
                    if pole<0:
                        near((point[0],),(mix(rest[0],target[0],angle/150),))
                    magnitude = abs(point[0]-rest[0])
                    if magnitude>previous+1e-9:
                        raise AssertionError(("spatial decay",angle,row,pole))
                    previous = magnitude

    for bend in (-150,150):
        for avoid in (False,True):
            pose = Elbow(bend,avoid_overlap=avoid)
            x,y,z = pose.point((2*step,1 if bend>0 else -1,0))
            gap = pose.ch*x+pose.sh*y*(1 if bend>0 else -1)
            if (gap>0) != avoid:
                raise AssertionError(("deep inner overlap",bend,avoid,gap))
    base = Elbow(120)
    for i,name in enumerate(CHANNELS):
        vals = list(ZERO)
        vals[i] = 1.0
        trial = Elbow(120,values=vals)
        moves = [dist(p,q) for p,q in zip(base.mesh(bind),trial.mesh(bind))]
        if max(moves) < 0.04:
            raise AssertionError(name)
        for (s,y,z),delta in zip(bind,moves):
            opposite_sector = (i!=10 and (y<0 if "In" in name else y>0))
            opposite_limb = (("Prev" in name and s>1) or ("Next" in name and s< -1))
            if abs(s)>=REACH or opposite_sector or opposite_limb or (i==10 and abs(z)<1e-9):
                if delta > 1e-8:
                    raise AssertionError((name,s,y,delta))
    for bend,twist,direction in ((-145,75,-80),(125,-40,177),(0,70,33),(90,180,50)):
        q = joint_quaternion(bend,twist,direction)
        alt = joint_quaternion(*read_rotation(q,bend,direction))
        if abs(sum(x*y for x,y in zip(q,alt))) < 1.0-1e-10:
            raise AssertionError(("driver rotation",q,alt))
    for label,reference,result in check_shots(bind,faces):
        print("%-24s  reference %+3d px    model %+6.2f px" % (label,reference,result))
    print("Passed: rest pose, rigid ends, bend-plane rotation, ring sections, deep crease, 11 controls and screen samples.")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv or "--self-test" in sys.argv:
        self_test()
    else:
        show()

