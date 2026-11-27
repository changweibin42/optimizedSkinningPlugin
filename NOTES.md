# OCD + Corrected CoR 移植笔记

> 这份文件是给"新对话里的自己"看的。当前仓库状态、已确认的结论、
> 被推翻的判断、以及卡住的确切位置都写在这里，不要重新走一遍已知的弯路。

- 仓库：`https://github.com/changweibin42/optimizedSkinningPlugin`
- 分支：`codex/ocd-hinge-core`
- 本文件所在提交：见 `git log -1`
- 平台：Windows，Maya 2024，DevKit `C:\Users\changweibin\devkitBase`

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

---

## 5. 卡住的确切位置

**同样的输入，C++ alg 5 在外扩量上只有 1.03R，而参考预览是 1.25R。**

已逐项排除：

| 验证项 | 结果 | 说明 |
|---|---|---|
| 纯父骨顶点 `guide` 符合公式 | `0.00000` | 73 个 `w1=0` 顶点精确相等 |
| C++ delta 经 alg 8 求值 vs alg 5 | `1e-6` | 求值路径一致 |
| `A·delta` 复现 C++ 顶点位移 | `0.00000` | 按 `w1` 分段全部精确 |
| `structuralCenters` 移植 | 与 Python 一致 | HALF=6 圆柱上 `2.46e-07` |

**所以求解器、`A_i`、`delta`、求值路径都没有错。**

**剩下的差异只能在 `original`（`originalLocal` vs 预览的 `original`），
或者 `chainWeights` / `settings.rotations` 的构造。**

### 未完成的下一步

**导出 C++ 的 `originalLocal`，与预览的 `original` 逐点对比。**

`ocdGuideDump` 给出 `guide`，所以可以反推：

```
originalLocal = Elbow.point(bindLocal) − guide      （当 ocdElbowGuide=true 时）
```

或者更直接：在 `deformOCD` 里临时把 `originalLocal` 写进一个新 dump 属性。

**这是唯一还没排除的环节。**

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

### 6.7 不要往 `D:\` 写文件

沙箱只允许写工作区 `C:\Users\changweibin\Documents\deepseek-harness\default-workspace`。
往 `D:\` 复制会 `PermissionDenied`。要给用户文件，走 git 推送。

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

---

## 10. 现有脚本清单

| 脚本 | 用途 |
|---|---|
| `tests/show_test04_preview.py` | **把参考预览搬到 test04**（七格数值对比） |
| `tests/ocd_elbow_reference_latest.py` | 与 `cor-local-structure-d78d264` 版本 SHA256 相同 |
| `tests/ocd_center_limit_toggle.py` | `ocdEnableCenterLimit` 验收 |
| `tests/ocd_alg8_fixedfit.py` | alg 8 语义验收 |
| `tests/verify_A_matrix.py` | `A·delta` 与顶点位移一致性 |
| `tests/guide_vs_elbow.py` | `guide` 与 `Elbow` 期望对比 |
| `tests/elbow_guide_compare.py` | 两种 guide 来源对比 |
| `tests/cpp_py_rig_fixed.py` | 正确 rig 层级下的 alg5 半径 |
| `tests/ring_roundness.py` | 坐标系无关的逐环圆度 |
| `tests/support_sweep.py` | `support` 扫描 |
| `tests/verify_hinge_continuity.py` | `hinge_section` 连续性与导数 |

---

## 11. 下一步建议

**优先级 1：查 `originalLocal` 与预览 `original` 的数值差异。**

方法：在 `deformOCD` 里加一个 dump 属性写 `originalLocal`（或者临时用
`Elbow.point(bindLocal) − guide` 反推，前提是 `ocdElbowGuide=true`），
与 `tests/show_test04_preview.py` 里的 `original`（alg 8 零偏移输出）逐点比较。

若两者一致 → 差异在 `chainWeights` 或 `settings.rotations`。
若不一致 → 就是它。

**优先级 2：确认 `chainWeights` 的第三列（helper）。**

test04 里 `helperBone = -1`（无 helper），但 `chainWeights[i][2]` 是否恒为 0？
若不为 0 而 `settings.rotations[2]` 是单位四元数，`A_i` 就会偏。

**优先级 3：如果前两项都排除，考虑让 `Elbow` 的 `HALF` 参数化。**

`Elbow` 的偏移表按 `HALF=6` 拟合。test04 的半长是 12，超出 2 倍。
若要让 `Elbow` 覆盖全肢体，需要把 `RING_STEP` 按实际 `|station|max` 缩放。
**但这会改变已认可的 458 点圆柱造型，需谨慎。**

---

## 12. 与用户协作的注意事项

- **用户要诚实**：没验证过的不要说"验证过了"；被推翻的判断要明确说明
- **用户不接受"这是算法局限"这种说法**，除非有数据证明
- **用户明确要求**：不要每次简单任务都问，直接做
- **用户的 Maya 有 numpy**，但 mayapy 需要装到 `.pylibs`
- **不要在没确认的情况下改 `D:\` 下的仓库**，那是用户的拉取目标
- **用户会自己编译**：给他干净的拉取命令 + 编译命令，不要让他猜
- **每次提交都要写清实测数据**，因为用户会看 commit message
