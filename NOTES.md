# OCD + Corrected CoR 移植笔记

> 这份文件是给"新对话里的自己"看的。当前仓库状态、已确认的结论、
> 被推翻的判断、以及下一步该做什么都写在这里，不要重新走一遍已知的弯路。

- 仓库：`https://github.com/changweibin42/optimizedSkinningPlugin`
- 分支：`codex/ocd-hinge-core`
- 本文件所在提交：见 `git log -1`
- 平台：Windows，Maya 2024，DevKit `C:\Users\changweibin\devkitBase`

**一句话状态（2026-11-27）**：目标已经达到。真实 `test04.ma` 上插件的
alg 5 与参考预览 `CoR_CurveGuided` 逐环一致到 `0.0005`，alg 3 与预览的
OCD 格一致到 `0.0002`。上一版笔记里"插件只有 1.03R"的判断**已被推翻**，
原因见 §5。剩下的工作是确认用户场景里 deformer 接在哪、以及视口表现（§11）。

---

## 1. 目标

把 `cor-local-structure-d78d264/scripts/ocd_elbow_reference.py` 里
`show()` 的 **`CoR_CurveGuided`** 格子的造型，完整复现到 C++ 插件的
`algorithmType` 里。

用户认可的效果（458 点圆柱、135°、三骨）在这个文件里能稳定复现：

```python
import runpy
script = r"D:\optimizedSkinningPlugin\cor-local-structure-d78d264\scripts\ocd_elbow_reference.py"
runpy.run_path(script)["show"](three_bone=True)
```

**关键事实：`CoR_CurveGuided` 格子最终网格就是插件算出来的。** Python 只负责
求中心偏移 `delta` 并写进 `corFitOffsets`，网格由插件的 `cor_paper::vertexTransform`
写回。旁边那个 `OCD` 格子才是 `Elbow.mesh()` 直接写点的纯 Python 对照。

---

## 2. 场景与数据

### 用户的主场景

- `D:/maya2024/scenes/ndj/test04.ma`
- 网格 `pCylinderTarget`：**16080 顶点**，16000 四边形，201 环 × 80 边
- 半径 `R = 0.984706`，`x ∈ [-12, 12]`，环距 0.12
- 骨骼 3 根：`joint1`(-12)、`L_Ankle_jnt`(0)、`joint3`(+12)
- **`joint3` 权重恒为 0**，实际只有两骨参与
- 混合区（`0 < w1 < 1`）约 `|x| ≤ 2.5`
- 姿态：`L_Ankle_jnt` 绕 Z **−147.2124°**（负弯）
- `ocdCenterLimit = 16.0`，`ocdHingeReport` = `parent=joint1 child=L_Ankle_jnt helper=<none>`

### 夹具文件

`D:\tmp\ocd_case3.json`（16080 顶点，bend −147.2124°），含：
`mesh.bindPoints`、`mesh.faces`、`bones[]`、`weightsSparse[]`、
`points.cor` / `points.ocd` / `points.final`、`hinge`、`attributes`。

**注意**：`weightsSparse` 是 `{influenceIndex: weight}` 稀疏字典，
必须按 `multiIndices` 读，不能按位置读。

### 参考圆柱夹具

`tests/_cyl1.json`（458 点，HALF=6，bend 146.13）+ `tests/_cyl2.json`（delta）。

### numpy

mayapy 默认**没有** numpy。已装到工作区：

```
.pylibs/            （.gitignore 已忽略）
```

用的时候设 `$env:PYTHONPATH = <repo>\.pylibs`。

---

## 3. 代码结构（当前分支）

### 算法分派 `src/optimizedSkinCluster.cpp` `deform()`

```
0 = CoR                 deformCoR(..., CenteredMode::CoR)
1 = DLB
2 = DQS Compensation
3 = OCD                 deformOCD(..., correctCenters=false)   ← 用 Elbow::point
4 = MOS
5 = OCD + Corrected CoR deformOCD(..., correctCenters=true)    ← 用 hingeDisplacement
6 = DDM v4
8 = CoR Fixed Center Fit deformCoR(..., CenteredMode::FixedFit) ← 用 corFitOffsets
```

**算法 3 和 5 共用 `deformOCD`，区别是 `correctCenters`。**

### `src/ocd_hinge.cpp` 里有两套几何

1. **`Elbow` 类**（完整移植）——`rail()` / `section()` / `point()`
   参考实现的 `Elbow` 逐行对应，含 `OUTER` / `INNER` / `INNER_CLEAR`
   三个偏移表（各 10 项，按 `HALF=6`、`RING_STEP=2/3` 拟合）。
   **算法 3 用它，输出绝对位置。**

2. **`hinge_section` / `hinge_displacement`**——内外两条轮廓 + 厚度梯度插值。
   **算法 5 用它，输出相对位移 `guide`。**

**这两者不是同一个几何。** `hinge_section` 是 `Elbow` 的简化替代。

### alg 5 的数据流

```
bind 网格
 → toLocalPoint：相对 childPivot，除以 R，投到 (axis, basisY, basisZ)
 → bindLocal（归一化局部坐标）
 → 纯 CoR 求值 → originalLocal（零偏移时的输出，同一空间）
 → guide = hingeDisplacement(...)  或  Elbow.point(bindLocal) - originalLocal
 → structuralCenters(edges, originalLocal, chainWeights, settings, guide, outer)
 → offsetsLocal（中心偏移，归一化空间）
 → centers[i] = baseCenters[i] + localOffsetToWorld(offsetsLocal[i])
 → cor_paper::vertexTransform(centers[i], ...) → 顶点
```

### `vertexTransform` 的恒等式（已验证精确成立）

```
p(c)    = Σ w_k·T_k(c)   + R_blend·(rest − c)
p(base) = Σ w_k·T_k(base) + R_blend·(rest − base)
⟹ p(c) − p(base) = (Σ w_k R_k − R_blend)·(c − base) = A_i · Δc
```

`A_i` 就是这个"中心响应矩阵"。

---

## 4. 已在 test04 上复现成功（重要）

`tests/show_test04_preview.py` 把参考预览搬到 test04 上（网格按 `1/R` 缩放，
`prev` 在 `−HALF`、`elbow` 在 `+HALF`、`helper` 与 `elbow` 共点）。

**截面半径结果（相对纯 CoR = 1.0）：**

```
   x/R      CoR    CurveGuided    OCD     OCD_CoR
 -1.219   1.0000     1.2508      1.1602   1.2508
 -0.975   1.0000     1.2550      1.1906   1.2550
 -0.731   1.0000     1.2508      1.2137   1.2508
 -0.487   1.0000     1.2406      1.2223   1.2406
  0.000   1.0000     1.1991      1.2195   1.1991
```

**两个关键结论：**

1. **`CurveGuided` 与 `OCD_CoR`（alg 5）逐位相同** —— 说明 alg 8 注入 Python delta
   与 alg 5 内部求解是同一个计算。
2. **造型确实能复现**：外扩到 1.25R，而纯 CoR 精确为 1.0。

**所以"test04 上做不出参考造型"这个判断是错的。** 造型能出来。

**⚠ 这个脚本有一个容易误读的地方**：它是在**新建的空场景**里用夹具重建了一套
`prev → elbow → helper` 的 rig，把网格按 `1/R` 缩放到单位半径，**不是**在
`test04.ma` 上跑。所以它证明的是「参考数学在这份网格上给出 1.25R」，
不是「插件在真实场景里给出 1.25R」。后者由 §5 的
`tests/real_target_curveguided.py` 证明。

---

## 5. 卡住的位置

**上一版这里写的是：同样的输入，C++ alg 5 在外扩量上只有 1.03R，而参考预览是 1.25R。**

**这个判断已经被推翻（见 §5.3）。** 在真实 `test04.ma` 上把插件建在 rest 网格
上游、摆成 −147.2124°，alg 5 与预览 `CoR_CurveGuided` 逐环一致到 **0.0005**，
alg 3（OCD）与预览的 OCD 格一致到 0.0002。**求解链没有任何残留偏差。**

复现命令（退出码 0 = 通过）：

```
set "MAYA_PLUG_IN_PATH=D:\...\optimizedSkinningPlugin\build\Release"
set "PYTHONPATH=D:\...\optimizedSkinningPlugin\.pylibs"
mayapy tests\real_target_curveguided.py
```

实测（真实场景，不是重建夹具；数据来自上面那条命令的输出）：

```
      x/R      cor/R    fig/cor     预览fig |        差    ocd/cor      预览OCD
   -1.219     1.0000     1.2508     1.2508    +0.0000     1.1602      1.1602
   -0.975     1.0000     1.2550     1.2550    +0.0000     1.1906      1.1906
   -0.731     1.0000     1.2508     1.2508    +0.0000     1.2138      1.2137
    0.000     1.0000     1.1991     1.1991    +0.0000     1.2195      1.2195
  24 个环的最大差 0.0005
```

### 5.1 「1.03R」最可能的来源是**接法**

`mc.deformer()` 会插在网格已有 deformer 的**下游**。test04 里
`pCylinderShape2` 上挂着原生 `skinCluster1`，所以

```python
mc.deformer("pCylinderShape2", type="optimizedSkinCluster")   # ← 错
```

拿到的 `inputGeom` 是 `skinCluster1.outputGeometry[0]`，也就是**已经摆好姿势
的网格**，而不是 rest 网格。插件于是把「姿势后的形状」当成 bind 网格去求
中心，输出又被再变形一次。实测同一姿态下：

| 接法 | deformer 输入 | x=−1.219 的 fig/cor |
|---|---|---|
| 上游（正确） | `pCylinderTargetShapeOrig.worldMesh` | **1.2508** |
| 下游（默认插入位置） | `skinCluster1.outputGeometry[0]` | 1.0952 |

下游接法在整条肢体上给出 0.83–1.13 的比值，量级正好落在「1.03R」附近。

**判据（一行）**：看 deformer 的输入是不是 rest 网格。

```python
mc.listConnections(skin + ".input[0].inputGeometry", s=True, d=False, plugs=True)
```

**正确的接法**：先断开网格的 `inMesh`，把 rest 形状的 `worldMesh` 直接接上去，
再 `mc.deformer()`；顺序反了就会插到原生蒙皮后面。

### 5.2 「1.03R」也可能是**度量口径**

畸变只集中在 `|x/R| < 3` 的混合区；`|x| > 6` 时 `Elbow`/`hinge_section` 的
rail 表取到 `OUTER[9] = 0`，是纯刚体，比值恰好 1.0。整条肢体平均会把 1.25
稀释成 1.03。夹具实测（`final/cor` 的逐环比值，按顶点数加权）：

```
全部 201 环    16080 点  平均 1.0447   ← 和一个"1.03R"的量级吻合
|x/R| <= 6      7920 点  平均 1.0908
|x/R| <= 3.2    4240 点  平均 1.1698
|x/R| <= 2.0    2640 点  平均 1.2294
```

**所以量外扩必须按环报，不要报整条肢体的一个均值**（§6.5 的同类坑）。

### 5.3 被推翻的判断

上一版 §5 的整张「已逐项排除」表方向是对的，但结论错了：它把
`originalLocal` / `chainWeights` / `settings.rotations` 当成仅剩嫌疑，
实际上这三者都在预期内，差异根本不在求解链上。**§11 的三步优先级因此作废。**

### 5.4 仍然值得记的一处口径差

`ocdCenterLimit` 属性按**世界单位**暴露，求解时除以 `radius` 变成
「肢体半径的倍数」；预览的 `center_limit=16` 本来就是半径倍数。所以

```
属性值 16.0            → 归一化 16.2485   （R = 0.984706）
预览等价                → 属性值 15.7553
```

实测两者对结果的最大影响只有 0.0005（`limit=4R` 时才把 x=−1.219 从 1.2508
拉到 1.2438，仍然远高于 1.03）。**但半径差别大的模型上这个语义差会放大**，
换模型时先算一遍 `limit / R`。

### 5.5 没验证的部分

我对**用户具体场景里 deformer 到底接在哪**没有直接观测。上面 A/B 两种接法
是我在真实 `test04.ma` 里自己搭的。要确认实际用法，跑 §5.1 那一行
`listConnections` 即可。

---

## 6. 必须记住的几个坑

### 6.1 测试 rig 层级必须与 `show()` 一致

**踩过这个坑，导致多个结论作废。**

```
正确：prev → elbow → tip，helper 是 prev 的另一个子节点
错误：elbow 挂在 holder 下、helper 挂在 prev 下
     → C++ 识别成 parent=prev child=helper helper=elbow
     → child 和 helper 互换，A_i 全错，delta 解在错误空间
```

**每次测试都用 `ocdHingeReport` 确认，应当是：**
`parent=prev child=elbow helper=helper`

### 6.2 `Elbow` 只覆盖 `|station| ≤ 6`

`_rail()` 里 `row = min(int(u/RING_STEP), HALF_RINGS-1)`，`u > 6` 时
`row = 8, t = 1` → 取 `OUTER[9] = 0`、`INNER[9] = 0` → **零偏移（刚体）**。

**test04 的 `|station|` 到 12，16080 个顶点里有 8080 个超过 6。**

混合区在 `|x| ≤ 2.5`，完全在有效范围内，所以关节附近的造型没问题。

### 6.3 负弯时 `plane` 翻转 180°

`hingeDisplacement` 里：

```cpp
const double plane = (params.direction + (params.bend < 0.0 ? 180.0 : 0.0)) * degToRad;
const double depth = depths[i] * cosPhase - laterals[i] * sinPhase;
```

`bend = −147°` 时 `plane = 180°`，`cos(0−180°) = −1`，
所以 `bind.y = −1`（视觉外侧）的顶点得到 `depth = +1`，
走 `inner` 分支，`h ≡ inner.h`，**与 `outer`（`gather` 作用处）无关**。

**这是 `ocdOuterGather` 对红圈一侧无效的原因，也是"视觉外轮廓由 inner 决定"的原因。**

### 6.4 `ocdElbowGuide` 默认是 `false`，这是有意的

**用 `Elbow::point` 驱动 alg 5 的 `guide` 是错的**，因为：

- `Elbow::point` 在"关节在原点、肢体沿 −X"的帧里建形状
- `bindLocal` 是相对 `childPivot` 的轴向坐标，跨越正负
- `originalLocal` 已是求值后的 CoR 输出

**两者相减是跨帧比较，135° 下正侧 `guide` 达 3.88，而解析形状实际只移动约 0.05。**
结果 `delta` 撞限幅 16，顶点位移 4.42，截面半径崩到 0.84。

**不要重新打开这个开关，除非先把 `Elbow` 的输出变换到与 `originalLocal` 同一帧。**

### 6.5 测量必须用坐标系无关的量

**踩过多次：**

- 用 `hypot(y, z)` 当截面半径 → 子骨旋转后完全失效
- 用全局 `PCA` 法向 → 圆环上特征值相等，法向不唯一
- 拿 `Elbow.mesh()`（肢体局部系）和 deformer 输出（对象空间）直接求最大差
  → 17.7 的假差异，实际两形状很接近

**正确做法：逐环取该环 80 个点，算质心，用 `‖p − 质心‖` 的 `std/mean`，
或用环内 PCA 的最佳拟合平面。**

### 6.6 `setStorable(false)` 的 dump 属性读不到

`deform()` 里用 `MPlug.setMObject` 写入的属性，若 `setStorable(false)`，
Maya 不保留，`getAttr` 永远是空数组。**必须 `setStorable(true)`。**（已在 `05e24c7` 修）

### 6.7 沙箱

旧版这里写"不要往 `D:\` 写文件，沙箱只允许写工作区"。**这条已作废**：
本会话的文件策略是 `danger-full-access`，工作区外也能读写，`D:\maya2024\scenes\ndj\test04.ma`
可以直接 `file -open`。

**但 `D:\optimizedSkinningPlugin\*` 是用户自己的拉取目录**，仍然不要在那里改仓库
（§12）。在 `D:\maya2024\scenes\ndj\` 下**只读不写**即可。

### 6.9 打开 test04.ma 之前要先 `loadPlugin`

```python
mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
mc.file(SCENE, open=True, force=True)
```

顺序反了会得到 `警告: 未知节点类型: optimizedSkinCluster`，场景里的节点变成
unknown，`mc.ls(type="optimizedSkinCluster")` 返回空。

### 6.10 场景打开时是零姿势

`test04.ma` 里的 `joint1 / L_Ankle_jnt / joint3` 打开后 `rotate` 全是
`(0,0,0)`，没有动画曲线驱动。夹具 `ocd_case3.json` 那一帧的
`−147.2124°` 是导出时手摆的。**测之前必须自己摆姿势**：

```python
mc.setAttr("L_Ankle_jnt.rotateZ", -147.2124)
```

不摆的话 alg 0 / alg 3 / alg 5 三者输出完全相同（都是 rest），
会误判成"插件没生效"。

### 6.8 `cmake` 不在 PATH

```
& "C:\Program Files\CMake\bin\cmake.exe" --build build --config Release
```

编译前设 `$env:DEVKIT_LOCATION = "C:\Users\changweibin\devkitBase"`。

---

## 7. 已实现但需要用户确认的改动

### `ocdEnableCenterLimit`（`05e24c7`，默认 `true`）

关闭后跳过 `structuralCenters` 末尾的逐顶点截断。

16080 夹具实测：
```
关/16 vs 开/32: 0.000e+00   （逐位相同，32 超过无限制解最大 28.07）
关闭时改限额 8/64: 0.000e+00 （完全不敏感）
重开/16 vs 原开/16: 0.000e+00
```

**但 458 点小圆柱上关闭限幅会退化**（顶点位移 4.5–5.0，圆度 0.11→0.22），
所以默认 `true` 是对的。

**用户反馈：这个开关在 test04 上"确实比之前好一点"。**

### `corFitOffsets` + alg 8（`91691d4`）

`CoR Fixed Center Fit`：`center += offset` → `vertexTransform`，插件不求解。

```
alg8 空偏移 vs alg0: 0.000e+00
alg8 全零偏移 vs alg0: 0.000e+00
```

长度校验为 `0` 或 `3 × 顶点数`，否则 `kInvalidParameter`。

---

## 8. 被推翻的判断（不要重复）

| 曾以为是 | 实际是 |
|---|---|
| Python 验证的是 `structuralCenters` | 用户认可的是 `show()` 的 `CoR_CurveGuided` 格子，`structuralCenters` 只是其中求 `delta` 的一步 |
| C++ 移植的 `structuralCenters` 有偏差 | 与 Python 一致到 `2.46e-07`（圆柱）/ `1e-6`（夹具） |
| `delta` 是顶点位移 | **`delta` 是中心偏移**，`centers = base + delta`，再走 `vertexTransform` |
| 红圈凹陷来自低权重区（`w1≈0.03`） | 负弯时 `plane` 翻转，视觉外轮廓由 `inner` 决定，`w1≈0.5` 才是关键 |
| `gather` 系数影响外轮廓 | 只影响 `bind.y>0` 一侧（负弯下），红圈一侧管不到 |
| 归一化 `gradient` 能修圆度 | 修了圆度但破坏纵向轮廓（118° `x=±0.667` 转角 0.87°→9.99°） |
| `support=2.6` 能修折痕 | 把折痕从 `x=0` 搬到 `x=±0.667`，且破坏 118°/120° 已认可形态 |
| test04 上做不出参考造型 | **能做出**，`show_test04_preview.py` 已复现 1.25R |
| C++ 用 `Elbow` 驱动 guide 会更好 | 跨帧相减，`guide` 达 3.88，结果崩溃 |
| 插件单独跑同样输入只有 1.03R，差异在 `originalLocal` / `chainWeights` | **错**。真实 `test04.ma` 上插件 alg 5 = 预览到 `0.0005`（§5）。1.03R 来自**接法**（deformer 插在原生蒙皮下游）或**把整条肢体平均**（§5.1/§5.2） |

---

## 9. 关键数值备忘

### 截面圆度定义

```
逐环取 80 点，质心 c，r_i = ‖p_i − c‖
圆度 = std(r) / mean(r)      越小越圆
```

### 纯 CoR 圆度精确为 0

因为它就是刚体旋转，环保持完美圆形。

### `Elbow` 的厚度梯度 `k = ‖(inner − outer) / 2R‖`

```
bend    中线k    最大k     圆度
  0     1.000   1.000   0.0000
 60     1.032   1.115   0.0383
 90     1.090   1.213   0.0678
118     1.196   1.332   0.0999
146.13  1.372   1.470   0.1329
```

**`k → 圆度` 的解析关系与实测吻合。**

### `hinge_section` 在 `station=0` 处

- 位置**精确连续**（差 `0.000e+00`）
- `depth=+1`：左导数 `0.5143`，右导数 `−0.4817`，**夹角 167.74°**
- 三点转角（`s=−2/3, 0, +2/3`）= **168.05°**

**这是两段刚体段之间的夹角，是 146° 弯曲的几何必然，不是 bug。**

### `Elbow` 在 `station=0` 处（2026-11-27 补测）

内轨侧（`y=+1`）单侧导数（`eps=1e-7`）：

```
d-/ds = (+0.908694, -0.417463)
d+/ds = (-0.987139, +0.159863)
夹角 = 164.5244°   偏折 15.4756°
位置连续：|p(0)-p(-eps)| = 2.697e-08
```

外轨侧（`y=-1`）**没有**折角（0.00°，光滑）。

**同一个解析形状，换采样密度后三点转角天差地别：**

| 采样 | 环距 | 内轨三点转角 | 外轨三点转角 |
|---|---|---|---|
| 24 边 × 19 环（458 点） | 0.6667 | **169.00°** | **44.60°** |
| 80 边 × 201 环（16080 点） | 0.0600 | 164.88° | 7.08° |
| 160 边 × 401 环（64160 点） | 0.0300 | 164.70° | 3.83° |

**关键**：`24x19` 的 169° 并不比 `80x201` 的 164.88° 更"平"，
而是**低密度折线把尖角切掉了**——`80x201` 顶点到 `24x19` 折线的最大距离
`3.549e-03`，这就是被切掉的尖角高度。

**所以，同一个形状，458 点看着圆滑，16080 点看着是刀口。**
用户的两张截图正是这个现象（图1 = 458 点，图2 = 16080 点），
不是插件移植错了，也不是 alg 5 求解错了。

---

## 10. 现有脚本清单

| 脚本 | 用途 |
|---|---|
| `tests/real_target_curveguided.py` | **★ 关键验收**：真实 `test04.ma` 上插件 vs 参考预览，含上游/下游两种接法 |
| `tests/show_test04_preview.py` | 把参考预览搬到 test04（七格数值对比，从夹具重建） |
| `tests/ocd_elbow_reference_latest.py` | 与 `cor-local-structure-d78d264` 版本 SHA256 相同 |
| `tests/ocd_center_limit_toggle.py` | `ocdEnableCenterLimit` 验收 |
| `tests/ocd_alg8_fixedfit.py` | alg 8 语义验收 |
| `tests/verify_A_matrix.py` | `A·delta` 与顶点位移一致性 |
| `tests/guide_vs_elbow.py` | `guide` 与 `Elbow` 期望对比 |
| `tests/elbow_guide_compare.py` | 两种 guide 来源对比 |
| `tests/cpp_py_rig_fixed.py` | 正确 rig 层级下的 alg5 半径（458 点） |
| `tests/preview_vs_cpp_radius.py` | 458 点圆柱上 preview / CoR / alg5 逐环对比 |
| `tests/ring_roundness.py` | 坐标系无关的逐环圆度 |
| `tests/support_sweep.py` | `support` 扫描 |
| `tests/verify_hinge_continuity.py` | `hinge_section` 连续性与导数 |
| `tests/replay_exported_case.py` | 从 `D:\tmp\ocd_case3.json` 重建场景并重放（复现到 `1e-22`） |

跑测试的通用环境：

```powershell
$env:MAYA_PLUG_IN_PATH = "<repo>\build\Release"
$env:PYTHONPATH = "<repo>\.pylibs"
```

**注意**：本机 `PYTHONPATH` 全局指向一个 Python 3.11 共享目录，会污染 mayapy
的 3.10 解释器（`SRE module mismatch`）。跑 mayapy 前必须把它覆盖成 `.pylibs`。

---

## 11. 下一步建议

§11 原来那三步（查 `originalLocal` / 查 `chainWeights` 第三列 / 参数化
`Elbow` 的 `HALF`）**已作废**：§5 已证明求解链在真实场景上与预览一致到
0.0005，那三步是在追一个不存在的偏差。

**现在的优先级：**

1. **确认用户场景里 deformer 接在哪**（§5.1 那一行 `listConnections`）。
   如果接在原生蒙皮下游，就是全部解释；如果不是，再说。

2. **把「已复现」这件事在视觉上确认一次**。数值上 alg 5 = 预览到 0.0005，
   但我没有亲眼看过视口。用户看的是视口，若视口仍不对，问题在 GPU/VP2 路径
   或另一个网格（test04 里 `pCylinderTarget` 和 `pCylinderShape2` 是两份
   16808/16080 点的网格，容易看错对象）。alg 3/5 都走 CPU（
   `optimizedSkinClusterGPUDeformer.cpp:344` 对 alg 3 显式返回失败），
   所以 GPU 路径理论上不参与，但没有验证过 alg 5 的视口结果。

3. **`Elbow` 的 `HALF=6` 限制**仍然是真的（§6.2 那条成立）：`|station| > 6`
   取到 `OUTER[9]=0` 是纯刚体。test04 混合区在 `|x| ≤ 2.5`，够用；
   换到更长的肢体才需要参数化。**这条不是当前卡点。**

---

## 11b. 2026-11-27：两张截图的差别（折角 / 刀口）

用户给了两张图：图1（698×488，看着圆滑）是**想要的**，
图2（749×556，肘部有刀口）是**现在 test04 的样子**。

结论：**这不是插件的问题，是同一个解析形状在两个网格密度下的不同表现。**

| | 图1 | 图2 |
|---|---|---|
| 网格 | 24 边 × 19 环 = 458 点 | 80 边 × 201 环 = 16080 点 |
| 环距（半径倍数） | 0.667 | 0.060 |
| 内轨 `station=0` 三点转角 | 169.0° | 164.9° |

`Elbow` 的内轨在 `station=0` 处**本来就有 15.48° 的折角**
（左右单侧导数夹角 164.52°，位置连续）。这是 `_rail()` 里两支
在 `station > 0` 时被 `rotate`、`station <= 0` 时不旋转的直接后果，
是解析模型的几何必然，不是移植误差——纯 Python `Elbow` 在同一位置
给出同一个角。

458 点之所以看着圆滑：环距 0.667，相邻两环跨过折角很远，
折线把尖角切成了圆弧（80×201 顶点到 24×19 折线的最大距离 `3.55e-03`）。
16080 点环距 0.060，把折角如实采出来，所以是刀口。

**复现**（只用纯 Python，不加载插件）：

```
mayapy tests/crease_shape_identity.py    # 折角单侧导数 + 形状同一性 + 各密度的转角
mayapy tests/crease_vs_density.py        # 把 24x19 与 80x201 都送进插件，结果与密度无关
```

**要改成图1 的样子，只有一条路：把折角磨圆。**代价：

| 圆角半径 R（肢体半径倍数） | station 跨度 | 最大横向偏差 |
|---|---|---|
| 1.0 | 0.269 | 0.009 |
| 2.0 | 0.539 | 0.018 |
| 3.0 | 0.808 | 0.027 |
| 5.0 | 1.346 | 0.046 |

要让 15.5° 在 16080 点上"看不出来"，R 至少 ~2.5，station 跨度 ~0.67
——**正好等于 458 点圆柱的环距**。换句话说，458 网格根本没有分辨这个
折角的采样力，用户认可的形态里从来没有包含"圆角"这个信息。

**这是需要用户决定的设计选择，不是 bug 修复**：
- 磨圆会让 458 点那套已认可的造型也发生变化（偏差 0.009–0.046）
- 不磨圆，16080 点就永远会看到折角
- 参考实现 `ocd_elbow_reference.py` 本身也没做圆角，所以"复现参考预览"
  和"看起来像图1"是**两个不同的目标**

### 11b.1 已实现：`ocdCreaseFillet`

属性 `ocdCreaseFillet`（短名 `occf`，默认 **0 = 关闭**），单位是肢体半径。
在 `|station| < W` 内用三次 Hermite 把折角接起来，窗口外**完全**沿用原实现。

两处都改了：
- `Elbow::rail()`（alg3 与 `ocdElbowGuide=true` 的 alg5 走这条）
- `hingeSection()`（alg5 默认的 `hingeDisplacement` 走这条）

实测（80×201 圆柱，alg3，station=0 的三点转角）：

```
fillet   turning
  0.00    164.88
  0.10     26.19
  0.20     12.12
  0.30      7.95
  0.40      5.92
  0.60      3.91
  0.80      2.91
```

`fillet=0` 时 alg3 与参考 `Elbow` 的最大顶点差 `4.677e-07`，
反复开关后逐位偏差 `0.000e+00`。验收：`tests/crease_fillet_check.py`。

### 11b.2 折角有三个独立来源（重要，别混为一谈）

| 来源 | 位置 | 性质 | 状态 |
|---|---|---|---|
| 形状折角 | `station=0` | `Elbow`/`hinge_section` 的左右两支旋转切换 | **已修**，`ocdCreaseFillet` |
| 求解器病态 | `|s|≈2.2` | `A_i` 秩 2 且弱区增益只有 0.0025 | **未修完**，见 §11c |
| 限幅撕裂 | `2.2<|s|<2.6` | 714 个顶点被硬截到 16.0 | 与病态同源，修病态后归零 |

**只修第一条不够**：alg5 在 ±2.2 仍有 112° 的折角，它来自求解器，不在圆角
窗口里。位置与量级：

```
alg3 (纯形状)  164.88° @ s=0     ← fillet=0.8 后 2.91°
alg5 (默认guide) 168.22° @ s=0   ← fillet 后 112.52° @ s=-2.1
```

---

## 11c. 求解器病态：`A_i` 秩恒为 2

`tests/solver_conditioning.py` 与 `tests/nullspace_probe.py` 的实测：

```
A_i 的奇异值（弯曲轴方向恒为 0）
  station -2.50  w1=0.0011   [0.00249, 0.00249, 0]
  station  0.00  w1=0.5000   [0.70868, 0.70868, 0]
  station  2.50  w1=0.9989   [0.00249, 0.00249, 0]
```

- **零空间恒为弯曲轴**（与 +Z 的 |cos| = 1.000000）。沿该轴的中心偏移对
  顶点位置**精确无影响**：移 1e3 单位，位移 `0.000e+00`（`tests/project_center.py`）。
- 弱区（`w1≈0` 或 `1`）仅剩的可观测奇异值只有 `0.00249`，比中位弱 285 倍。
  于是目标函数只要 `|guide| = 0.8`，反解就要求 `|delta| = 70`（放大 86 倍）。
- **PCG 是收敛的**：41 次迭代，`residualRatio = 6.4e-13`（`tests/pcg_diagnostics.py`）。
  所以那 70 不是数值噪声，是这套最小二乘的**真实解**。
- 后果：714 个顶点被中心限幅截到 16.0，相邻截断量不同 → `2.2<|s|<2.6`
  出现 158.98° 的阶梯撕裂；关掉限幅后降到 109.78°，说明撕裂之外还有病态。

**已试过但不够的**：
- 把解投影回 `A_i` 行空间（去零空间）：几何精确不变，但没降低 `|delta|`
  —— 因为 70 全在可观测方向。
- 按 `A_i` 可观测增益做自适应 ridge：`|delta|` 从 70 降到 5.2、被截顶点
  714→0，但 test04 的复现精度从 `0.0005` 掉到 `0.1121`，**不可接受**，
  已撤回。

**下一步要做的**（需要动求解器结构，不是调参）：
`structuralCenters` 目前是逐顶点独立的正则，缺少沿经线的耦合。
`delta` 沿经线应该是平滑函数，但相邻顶点各自解一个病态问题就会震荡
（实测 `|guide|` 每步平滑变化 0.029，而 `|delta|` 每步跳 0.41–1.89）。
要真正解决，得让中心偏移在经线方向上也受光滑性约束，
或者改用能直接约束 `A_i` 零空间的分解（SVD 截断而非 ridge）。

---

## 11d. 109.78° 折角的逐层定位（2026-11-27 晚）

这一节是上面 §11c 的继续：不再扫参数，而是把折角拆到「哪条边、由哪一项造成」。

### 三层折角互相独立（已用实验分开）

| 层 | 位置 | 成因 | 状态 |
|---|---|---|---|
| 1 形状折角 | `station=0` | `hinge_section`/`Elbow` 内轨左右两支的旋转切换 | `ocdCreaseFillet` 可磨到 2.91° |
| 2 求解器病态 | `|s|≈2.2` | `A_i` 秩 2，弱区可观测增益 0.00249 | 见下 |
| 3 限幅撕裂 | `2.2<|s|<2.6` | 714 顶点被硬截到 16.0 | 与层 2 同源 |

**圆角只治层 1**：`fillet=0.4` 把 `s=0` 从 170.24° 降到 13.01°，
但 `|s|≈2.2` 的 109.78° **一动不动**。

### 3 折角在 `s=-2.28` 的精确结构（顶点 ID 对齐）

```
     ID    station     引导 T_guide              最终 final
i0  4880   -2.340   (-2.97259, +0.87217)   (-2.88534, +0.88777)
i1  4960   -2.280   (-2.93108, +0.86843)   (-2.89752, +0.87376)
i2  5040   -2.220   (-2.89010, +0.86459)   (-2.87770, +0.86607)

边   引导边 XY            最终边 XY            投影比    最终边长
b-  (+0.04151,-0.00374)  (-0.01217,-0.01401)  -0.2608   0.01856   ← 反向
b+  (+0.04098,-0.00384)  (+0.01982,-0.00769)  +0.4970   0.02127   ← 正向
```

复算 `atan2(0.000371, -0.000134) = +109.7834°`，与 `signed_turning` 一致。

**签名验证**：`引导夹角 -0.214° + b+偏转(-15.854°) - b-偏转(-125.851°) = +109.783°`
→ **入边 `4880→4960` 反向折返 125.85°，出边只偏 15.85°。是边折返，不是两边横向对冲。**

**三次踩过的错（务必按 ID 定位，不要按 station）**：
1. 曾把 `+0.4970` 记成 `b−` —— 环索引与转角索引错开一格（`turn[j]` 对应 `stations[j+1]`）。
2. 曾把 `|guide|max`（顶点 8880, s=+0.66, 1.2336）与 `|delta|max`
   （顶点 11320, s=+2.46, 70.092）当成同一个点算放大倍数。
3. 曾用中点矩阵代表三点 —— 实际 `σ(A)` 在 4880/4960/5040 上是
   `0.016479 / 0.024735 / 0.034554`，各不相同。

### 结构项消融（原版 `fillet=0`，λ=0，无限幅）

| 边（起点→终点 ID） | 原 compliance | compliance=0 |
|---|---|---|
| 入边 `4880→4960` | **-0.2608** | **+0.1367** |
| 出边 `4960→5040` | +0.4970 | +0.6196 |
| 邻边1 `4720→4800` | -2.7817 | **-3.1238**（更负） |
| 邻边2 `4800→4880` | -1.5017 | -1.0189 |
| 顶点 `4960` 转角 | +109.78° | **+50.76°** |
| 整段最大转角 | 109.78° @ 4960 | **124.29° @ 4640** |

**凸侧牵引项完全无关**：关掉它，所有数字逐位不变，可排除。

**`compliance` 同时做两件相反的事**：在入边上**制造**折返，在邻边1 上**缓解**过度反向。
所以全局关闭会修好入边、恶化邻边、并把尖峰推到 `4640`。

### 弱响应区局部衰减（`floor` 保留非零下限）

用 `g_i = σ_max(A_i)` 归一化后 smoothstep，对 `compliance` 整体乘系数：

```
配置                      入边4880→  出边→5040  邻1 4720→  邻2 4800→  最大转角  中点ID
原版（scale=1）           -0.2608   0.4970   -2.7817   -1.5017   109.78   4960
全关（scale=0）           +0.1367   0.6196   -3.1238   -1.0189   124.29   4640
floor=0.4 lo=.05 hi=.60   -0.0470   0.6031   -2.9599   -1.3410   103.19   4640
```

**`floor=0.4` 是本轮最好的单档**（留作对照，**未采纳为默认**）：
- 整段峰值 109.78° → **103.19°**（降 6.59°）
- 入边 -0.2608 → -0.0470（**仍反向，只是接近零**）
- 邻边1 -2.9599（优于全关的 -3.1238）
- 极点位移变化 **0.0000**
- **但最短边长从 0.01856 缩到 0.01416（-24%）**，峰值仍在 `4640`

三项改善 + 一项恶化，属**有限度改善**，不是尖峰转移，也远不是解决。

### 归因收窄（用户纠正，记下来）

`|delta|` 的陡变只描述**中心解**。真正让边翻转的是相邻顶点的
**实际位移差** `A_j·Δc_j − A_i·Δc_i` 吃掉了基础边的正向分量。
单点切向投影能跟上目标，**不足以证明相邻两点能同时形成一条正向边**。
所以验收一律用**边投影比 + 最终经线**，`|delta|max` 只作辅助记录。

### 圆角的作用范围比预期大（待解释）

`hinge_section_filleted` 用整批权重时，`Δinner(s)` 在 `|s|<1.6` 内**都不为零**：

```
 station    |Δinner|   |Δouter|   |ΔT|
  -1.020    1.159e-01  5.940e-02  1.159e-01
   0.000    9.890e-02  1.922e-04  9.890e-02
   0.960    1.166e-01  5.343e-02  1.166e-01   ← 峰值在窗口外
   1.500    2.023e-02  2.082e-02  2.023e-02
   1.620    0.000e+00  0.000e+00  0.000e+00
```

用户给出的关系式**实测完全成立**：`ΔT = (1+d/R)/2·ΔI + (1-d/R)/2·ΔO`
（预测 `1.166e-01` vs 实际 `1.166e-01`）。所以问题在 `ΔI(s)` 本身不局部，
不在数据流。**外轨也被移动了 0.05343**，以「局部圆角」为目标还算不合格。

注意：`ref.hinge_section` 会用**整批权重和站距**推导共享的 `support`，
所以逐点传 `W[i:i+1]` 即使行索引正确，也可能与整批调用给出不同轮廓。
`fillet=0` 已验收逐位一致（`0.000e+00`），可作基线；
量非零圆角时必须先固定原版整批算出的 `support`。

### 下一步（2026-11-27 决定）

**直接约束最终边的有符号投影**（用户方案），不再细扫 `taper`：

```
q_ij = [(p_j0 - p_i0) + A_j·Δc_j - A_i·Δc_i] · t_hat_ij / l_ij
E_edge = μ Σ_ij k_ij l_ij² [max(0, ρ - q_ij)]²        ρ = 0.1
```

用每顶点的 CoR 中心偏移做未知量，复用现有 PCG：
先解一遍找出 `q < ρ` 的边，作为二次项加入，再解并更新活跃边集。
**第一轮排除跨中线的边**（那里的原引导自带深折，会污染判断）。
软惩罚允许残余违约，**最终必须实测投影比**，不能从"开了惩罚"推"边已正向"。

### 11e. 纯中心边约束的结论：这条路线在预算内走不通

照上面的方案做了三件事，全部有实测：

**（1）正确算子。** 稀疏 `B` 与显式矩阵对拍 `1.735e-18`（双向）。
踩过三个坑，都记下来别再犯：

| 坑 | 症状 | 修正 |
|---|---|---|
| 梯度少 `l` 因子 | 与有限差分差 2 倍 | `∂E/∂Δc_b = -2μklv·A_bᵀt̂`（是 `l` 不是 `l²`）；`tests/edge_grad_check.py` 误差 `3.9e-08` |
| `q` 在 `l→0` 时发散 | `q_base` 到 `±38` | 按 `l > 0.15·bind_len` 过滤（过滤 700 条） |
| `Bᵀy` 用了 `A` 不是 `Aᵀ` | 与显式差 `5e-3`，只在顶点同时属 `ea`/`eb` 时暴露 | `np.einsum("nji,...")` |

另外：`B` 的每行**不带 `1/l`**，长度因子全部由 `d = √k·l·(ρ − q_base)` 承担；
`E = μ k l²(q−ρ)²` 与 `‖Bx − d‖²` 才严格对应。

**（2）软惩罚实测（`fillet=0`、`λ=0`、无限幅、`ρ=0.1`）：**

```
配置      入边4880→  邻1 4720→  最大转角  峰值ID   |dc|max  最短边  违约边
原版      -0.2608   -2.7817   109.78   4960   70.09  0.01856  5964
w=1       -0.3035   -2.7095   112.16   4960   69.75  0.01959  5452
w=10      -0.5118   -2.2369   117.06  11440   66.58  0.01474  5394
w=100     -0.5384   -1.0480   178.79  10880   59.90  0.00153  5276
```

违约数和邻边1 改善，但**入边越加大 `w` 越负**，最大转角全程变差。
（注：上表的 `-0.2608` 是「最终网格」的投影比；`q_base`（Δc=0）是 **`-2.7941`**，
两者不是同一个量，早期把二者混用过。）

**（3）可行性核验（决定性）：**
夹具 `radius = 1.000000`，故归一化预算 `C = 16.0`（与世界单位一致）。
取违约量最大的 2220 条边做子集：

```
单约束可达性：2220 / 2220 条单独都可达
在 ||dc_i|| <= 16 下做投影梯度：
  C 只用到 1.0092（远未用满 16）
  仍违约 1576 / 2220 条，最小 q = -1.0644
```

**单约束全可达、联合不可达，且中心偏移只用到 1.0092 就停滞**——
阻碍来自**约束间的方向冲突，不是预算不足**。加大预算不会改善。

**限定（必须保留）：** 这是投影梯度，不是凸锥求解器（mayapy 无 scipy），
所以严格说只证明「一阶方法在 C=16 内找不到可行解」，未证严格不可行。
但「用到 1.0092 就停滞」与预算无关，足以支持下面的工程结论。

**工程结论**（限定条件要一起说）：
> 对这张网格、这个姿态和当前候选边集合，纯 CoR 中心偏移无法在生产预算内
> 同时满足边投影要求。

### 11f. 转向方向 A：曲线引导邻域权重

原方案（`OCD曲线引导CoR旋转中心优化方案.docx`）不是「先定形状再反解中心」，
而是改 CoR 的**邻域权重**：

```
s*_ij = s^W_ij · g^curve_ij
```

这条路线绕开了 `A_i` 反解，因此不受 11c/11e 那个病态的约束。

**已实测的性质**（458 点圆柱，`tests/curve_gate_feasibility.py`）：

```
环内点对：曲线参数差恒为 0  =>  g_curve = 1.000000
跨环点对（σ_s=0.1）：1 环 0.857、2 环 0.539、3 环 0.249、6 环 0.0039
```

即**曲线因子只作用于跨环、不影响环内**，所以天然不会破坏已认可的截面造型。
量级（0.54~0.86）与现有 CoR 积分核（`σ=0.1` 时 `Δw=0.1` → 0.368）相当，是温和调制。

**第一版要验证的**：同一批顶点 ID 下，原 CoR vs 曲线引导邻域权重，
比最终经线转角、反向边、最短边、内外极点。

---

## 12. 与用户协作的注意事项

- **用户要诚实**：没验证过的不要说"验证过了"；被推翻的判断要明确说明
- **用户不接受"这是算法局限"这种说法**，除非有数据证明
- **用户明确要求**：不要每次简单任务都问，直接做
- **用户的 Maya 有 numpy**，但 mayapy 需要装到 `.pylibs`
- **不要在没确认的情况下改 `D:\` 下的仓库**，那是用户的拉取目标
- **用户会自己编译**：给他干净的拉取命令 + 编译命令，不要让他猜
- **每次提交都要写清实测数据**，因为用户会看 commit message
