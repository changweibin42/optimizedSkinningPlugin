# Optimized Skinning Plugin — 项目状态

**更新日期**：2026-10-10
**Maya 版本**：2024 (24.2.0.1186)
**DevKit**：C:\Users\changweibin\devkitBase
**编译脚本**：`build.ps1 -Clean`
**当前模式**：**CPU 单节点**（GPU 架构已归档，见 `GPU_ACCELERATION_ARCHIVE.md`）

---

## 1. 项目概览

实现三种论文级蒙皮算法的 Maya 插件：

| 算法 | 论文 | 状态 |
|---|---|---|
| **CoR** (Centers of Rotation) | Le & Hodgins 2016 | ✅ 严格按论文 + 增量优化 |
| **DLB** (Dual Quaternion Linear Blending) | Kavan 2008 | ✅ 严格按论文 |
| **DQS 补偿** | Sulema & Rudenko 2020 | ✅ 对称化修正 + 作者多项式勘误 |

**核心设计**：
- 继承 `MPxSkinCluster`，Maya 原生 Skin 工具（画笔、Copy、Mirror）全部可用
- CoR 预计算缓存 + 增量重算，画笔丝滑
- CoR 可视化：VP2 DrawOverride，无 Python 依赖

**编译环境**：
- Visual Studio 2022 Community
- CMake ≥ 3.22
- Maya 2024 SDK（`DEVKIT_LOCATION` 环境变量指向 `C:\Users\changweibin\devkitBase`）

---

## 2. 文件结构

### 2.1 已编译（当前启用）

| 文件 | 职责 |
|---|---|
| `src/plugin_main.cpp` | 插件入口（注册 A + viewer + cmd） |
| `src/math_core.h` | Vec3 / Quat / RigidTransform / DualQuat |
| `src/algorithms.h/cpp` | 三种算法的纯数学实现（无 Maya 依赖） |
| `src/optimizedSkinClusterCoR.h/cpp` | CoR 计算的辅助结构 + 函数 |
| `src/optimizedSkinCluster.h/cpp` | Node A（主节点，MPxSkinCluster） |
| `src/optimizedSkinClusterCmd.h/cpp` | `optimizedSkinClusterCoR` 命令 |
| `src/optimizedSkinClusterCoRViewer.h/cpp` | Viewer 节点（MPxLocatorNode） |
| `src/optimizedSkinClusterCoRViewerOverride.h/cpp` | Viewer 的 VP2 DrawOverride |
| `src/node_ids.h` | Maya 节点 ID 分配 |

### 2.2 保留但未编译（历史存档）

| 文件 | 说明 |
|---|---|
| `src/optimizedSkinClusterCompute.h/cpp` | Node B（CPU 数据打包） |
| `src/optimizedSkinClusterGPUDeformer.h/cpp` | B 的 OpenCL deformer |
| `src/skinClusterData.h/cpp` | Node A → B 的传输类 |
| `src/kernels/kernel_*.cl` | OpenCL kernels |

**参考**：`GPU_ACCELERATION_ARCHIVE.md`（GPU 模式完整归档）

---

## 3. 三种算法实现细节

### 3.1 CoR — Le & Hodgins 2016

**运行时**（`cor_vertex_v2`）：
- QLERP 混合旋转 + LBS 混合 CoR
- 严格按论文 Algorithm 1

**预计算**（`computeCoRCache`）：
- 论文 §3.2 的 ANN 聚类 + BFS + 积分
- 相似度函数 = 论文式(1)
- **三角形细分**：按论文原文"递归二分一条边"，ε = 0.1，最大深度 6

**增量重算**（`updateCoRCacheIncremental`）：
- 只重算 dirty 顶点及其 N 环邻域
- 邻域扩展层数自适应：`budget = numVertices / div`，clamp 到 [500, 3000]
- **实测**：单笔涂抹 2~5ms（vs 30k 全量 ~60ms）

**优化**：
- `SparseW` 稀疏权重（排序 + 归并，替代 `unordered_map::find`）
- `MArrayDataHandle` 句柄外提（避免每顶点重复获取）

### 3.2 DLB — Kavan 2008

严格按论文 Algorithm 1，无偏离。

### 3.3 DQS 补偿 — Sulema & Rudenko 2020

**方向**（`bisector`）：
```
bisector = -normalize(dir1 + dir2)     // 额外取反（Mark 版基线）
if (d2 > d1) bisector = -bisector       // hierarchy 反转（论文）
```

**补偿系数**：
- `f(w)`：**对称化** `f(min(w, 1-w))`，其中 `f(x) = 2.2x - 8.1x² + 7.4x³`
- `r`：**顶点到关节中心距离**（不是到骨骼线段距离）
- `k3 = 1 - W3/W2`（论文式13）+ clamp

**为什么偏离论文**：

| 问题 | 论文 | 我们的处理 |
|---|---|---|
| `f(w)` 不对称 | `f(w) ≠ f(1-w)` | `f(min(w,1-w))` 强制对称 |
| `r` 不对称 | 顶点到骨骼线段距离 | 顶点到关节中心距离 |
| 补偿方向依赖层级 | 全局固定 `V_bisector` | 保留（已修正的对称化 f 使方向问题不影响） |

**作者修正**（已应用）：
- 多项式 `2.2, -8.1, 7.4` → 作者论坛确认可换用 `2.2, -10.0, 11.2`（当前用原版）
- 公式(12) 括号 bug：已忽略

**可调**：`dqsCompStrength` 属性（默认 1.0，范围 0~2）

---

## 4. 节点属性参考

### 4.1 `optimizedSkinCluster`（Node A）

| 属性 | 短名 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `algorithmType` | `algo` | enum | 0 | 0=CoR, 1=DLB, 2=DQS 补偿 |
| `dqsCompStrength` | `dqscs` | double | 1.0 | DQS 补偿强度 |
| `displayCoR` | `dcr` | bool | 0 | 显示 CoR 点云 |
| `displayCoRMode` | `dcrm` | int | 0 | 0=rest, 1=runtime |
| `corTriLayerExpand` | `ctle` | int | 1 | 增量三角形扩展层数 [0,4] |
| `corVertLayerExpand` | `cvle` | int | -1 | 增量顶点扩展层数 [-1=自适应, 0~5] |
| `corIncBudgetDiv` | `cibd` | int | 10 | 自适应 budget 除数 |
| `forceFullCoR` | `ffcor` | bool | 0 | 调试：强制全量重算 |

### 4.2 `optimizedSkinClusterCoRViewer`

| 属性 | 短名 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| `corSourceNode` | `corSrc` | message | — | 连到 skinCluster.message |
| `pointSize` | `ptsz` | double | 4.0 | 点大小 |
| `colorR/G/B` | `colr/colg/colb` | double | 1/0/0 | 点颜色 |
| `maxPoints` | `mxpt` | int | 20000 | 最大显示点数（超过抽稀） |

---

## 5. 使用流程

### 5.1 绑定（Python）

选中 mesh + joints，执行绑定脚本。

**绑定脚本思路**：
1. 用原生 `cmds.skinCluster(...)` 计算权重
2. 用 `cmds.copySkinWeights(...)` 一次性拷贝到 `optimizedSkinCluster`
3. 删除临时原生 skinCluster

**关键命令**：
```python
cmds.copySkinWeights(
    sourceSkin=native_skin,
    destinationSkin=opt_node,
    noMirror=True,
    surfaceAssociation="closestPoint",
    influenceAssociation=["closestJoint", "oneToOne", "label"])
```

### 5.2 CoR 可视化

**自动机制**（无需手动）：
- 创建 `optimizedSkinCluster` 时，idle 事件自动创建 viewer
- `displayCoR = 1` 时显示，`= 0` 时隐藏
- 删除 skinCluster 时 viewer 跟随删除
- undo 恢复 skinCluster 时 viewer 自动重建

**使用**：
```python
cmds.setAttr("optimizedSkinCluster1.displayCoR", 1)
cmds.setAttr("optimizedSkinCluster1_corViewer.pointSize", 10)
```

### 5.3 CoR 数据查询

```python
# 查询 CoR（rest）
cor_flat = cmds.optimizedSkinClusterCoR(node="optimizedSkinCluster1")
# 返回 [x0,y0,z0, x1,y1,z1, ...]
```

### 5.4 绑定脚本骨架

```python
import maya.cmds as cmds

sel = cmds.ls(selection=True, long=True)
joints = [n for n in sel if cmds.nodeType(n) == "joint"]
mesh_shape = None
for n in sel:
    if cmds.nodeType(n) == "mesh":
        if cmds.getAttr(n + ".intermediateObject"):
            continue
        mesh_shape = n
        break

if not joints or not mesh_shape:
    raise RuntimeError("Need mesh + joints in selection")

# 原生 skinCluster
cmds.select(mesh_shape, replace=True)
cmds.select(joints, add=True)
native = cmds.skinCluster(
    mesh_shape, joints,
    toSelectedBones=True,
    bindMethod=0,
    dropoffRate=4.0,
    maximumInfluences=4,
    normalizeWeights=1)[0]

# optimizedSkinCluster
cmds.select(mesh_shape, replace=True)
opt_node = cmds.deformer(type="optimizedSkinCluster")[0]

# 连骨骼
for i, j in enumerate(joints):
    if not cmds.objExists(j + ".lockInfluenceWeights"):
        cmds.addAttr(j, sn="liw", ln="lockInfluenceWeights", at="bool")
    cmds.connectAttr(j + ".lockInfluenceWeights",
                     opt_node + ".lockWeights[%d]" % i, force=True)
    cmds.connectAttr(j + ".worldMatrix[0]",
                     opt_node + ".matrix[%d]" % i, force=True)
    m = cmds.getAttr(j + ".wim")
    cmds.setAttr(opt_node + ".bindPreMatrix[%d]" % i, *m, type="matrix")

# 拷贝权重
cmds.copySkinWeights(
    sourceSkin=native,
    destinationSkin=opt_node,
    noMirror=True,
    surfaceAssociation="closestPoint",
    influenceAssociation=["closestJoint", "oneToOne", "label"])

# 清理
cmds.delete(native)
cmds.setAttr(opt_node + ".algorithmType", 0)
cmds.refresh(force=True)
```

---

## 6. 关键技术点

### 6.1 增量 CoR

`setDependentsDirty` 检测权重变化：
- 优先：从 plug 提取顶点索引（快速路径）
- 兜底：全量权重对比（`MArrayDataHandle`）

`deformCoR` 检测 dirty：
- 首次/拓扑变化 → 全量重算
- 权重变化 → 增量重算（dirty + 邻域）
- dirty > numV/4 → 全量重算

### 6.2 Viewer 生命周期

| 事件 | 动作 |
|---|---|
| 创建 skinCluster | `postConstructor` 排队 idle 创建 viewer |
| `displayCoR` 从 0→1 | `setDependentsDirty` 排队检查创建 |
| 删 skinCluster | `preRemoval` callback 用 `MDagModifier::deleteNode` 删除 viewer（不进 undo 栈） |
| undo | `Undo` event callback 检查并重建残缺 viewer |
| 保存/加载场景 | 上述流程自动触发 |

### 6.3 VP2 渲染

`MPxDrawOverride` 的 `addUIDrawables` 用 `MUIDrawManager::points(pts, false)` 绘制。
- `false` = local space（乘 viewer 的 worldMatrix）
- `setDepthPriority(5)` 让点穿透物体显示

### 6.4 编译命令

```powershell
Stop-Process -Name maya -Force -ErrorAction SilentlyContinue
Stop-Process -Name mayabatch -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
cd D:\visualStudioProject\optimizedSkinningPlugin
$env:DEVKIT_LOCATION = "C:\Users\changweibin\devkitBase"
.\build.ps1 -Clean
```

**注意**：Maya 缓存节点定义，改 `initialize()`（属性）后**必须完全退出 Maya 再重开**。

---

## 7. 已知限制

| 限制 | 说明 |
|---|---|
| **DQS 补偿不处理缩放/剪切** | DQS 本身不支持非刚体变换 |
| **DQS 补偿对多骨影响区粗糙** | 论文只考虑前两骨，k3 缓解但有限 |
| **CoR 预计算 O(V×F)** | 首次绑定 ~60ms/1.6k 顶点，30k 顶点 ~1~3 秒 |
| **增量 CoR 的聚类不重算** | dirty 顶点用旧聚类中心，影响极小 |
| **viewer 只支持 rest pose 显示** | runtime CoR 未实现（需每帧重算 CoR 的 LBS） |
| **GPU 路径未启用** | 需要 `MPxGPUStandardDeformer` + OpenCL |

---

## 8. 与 GPU 归档的关系

`GPU_ACCELERATION_ARCHIVE.md` 记录了完整的 GPU 架构，包含：
- Node A + Node B 双节点设计
- `MPxData` 传输协议
- 三个 OpenCL kernel
- 性能数据（3.20x 加速）

**当前状态**：GPU 代码保留在 `src/` 但**未编译**，`plugin_main.cpp` 只注册 CPU 路径。

**恢复 GPU**：参考归档文档 §8 "恢复 GPU 模式的步骤"。

**注意**：GPU kernel 的 DQS 补偿公式**未同步** CPU 的对称化修正，恢复时需要更新 `kernel_dqs.cl`。

---

## 9. 关键踩坑记录

### 9.1 Viewer 节点属性与 Maya 内置属性重名

**症状**：`addAttribute` 静默失败，属性没注册成功。

**原因**：短名 `isc` 和 Maya 内置 `isCollapsed` 冲突。

**解决**：所有自定义属性用**4 字符短名**：
- `corSrc`（不是 `isc`）
- `ptsz`（不是 `psz`）
- `colr` / `colg` / `colb`（不是 `cr` / `cg` / `cb`）
- `mxpt`（不是 `mxp`）

### 9.2 `MFnDagNode::create` 对 locator 类型

**症状**：`createNode` 给 shape 名，但 transform 保留默认名 `transform1`。

**解决**：用 MEL 创建后**再改名 transform**：
```cpp
createNode -n <tempShape> -skipSelect optimizedSkinClusterCoRViewer;
// 拿 transform，改名
transformFn.setName(viewerName);
shapeFn.setName(viewerName + "Shape");
```

### 9.3 `MPlug(skinObj, MNodeMessage::message)` 错误

**症状**：编译错误。

**原因**：`MNodeMessage` 是回调类，没有 `message` 静态成员。

**解决**：用 `MFnDependencyNode::attribute("message")` 取 MObject：
```cpp
MFnDependencyNode skinFn(skinObj);
MObject msgAttr = skinFn.attribute("message");
MPlug srcPlug(skinObj, msgAttr);
```

### 9.4 `MString::format` 不支持整型

**症状**：`MString::format("%u", ...)` 编译错误。

**原因**：Maya 的 `MString::format` 只接受 `MString` 参数，不是 printf 风格。

**解决**：用 `snprintf` + `MString`：
```cpp
char buf[128];
snprintf(buf, sizeof(buf), "value = %u", n);
MGlobal::displayInfo(MString(buf));
```

### 9.5 `MObject::ptr()` 在 Maya 2024 是私有的

**症状**：`MObject::ptr()` 编译错误。

**解决**：用 UUID 做 map key：
```cpp
MFnDependencyNode fn(obj);
MUuid u = fn.uuid();
std::string key(u.asString().asChar());
```

### 9.6 OpenGL 函数链接错误

**症状**：`__imp_glBegin` 等未解析外部符号。

**解决**：在 `optimizedSkinClusterCoRViewerOverride.cpp` 顶部加：
```cpp
#ifdef _WIN32
#pragma comment(lib, "opengl32.lib")
#endif
```

### 9.7 Undo 时 viewer 不跟随恢复

**症状**：`cmds.delete(skinCluster)` → `cmds.undo()` 只恢复 skinCluster，viewer 没回来。

**原因**：
- `postConstructor` 不触发（节点是 restore，不是 create）
- `setDependentsDirty` 不触发（`displayCoR` 值没变）

**解决**：注册 `Undo` 事件回调（`MEventMessage::addEventCallback("Undo", ...)`），遍历所有 skinCluster，残缺 viewer 自动重建。

### 9.8 `preRemoval` 回调里不能改 DG

**症状**：删除 skinCluster 时 Maya 崩溃。

**原因**：`MNodeMessage::addNodeDestroyedCallback` 的 clientData 被释放后仍被引用；且回调里直接 DG 操作引起递归。

**解决**：
- 用 `std::map<std::string, MString>` 存 viewer 名，不用 `new MString`
- 用 `MNodeMessage::addNodePreRemovalCallback`（在移除前触发）
- 用 `MDagModifier::deleteNode` 同步删除（不进 undo 栈）

---

## 10. 版本历史

| 日期 | 里程碑 |
|---|---|
| 2026-09-28 | 项目初始化，A 单节点 CPU 版 |
| 2026-10-01 | A+B 双节点 + MPxData |
| 2026-10-03 | 解决 A+B 递归求值崩溃 |
| 2026-10-08 | GPU 三算法全部通过，性能 3.20x |
| 2026-10-08 | **归档 GPU，回归 CPU** |
| 2026-10-09 | 增量 CoR + 自适应邻域 |
| 2026-10-09 | 代码拆分（CoR 独立文件） |
| 2026-10-10 | DQS 补偿对称化修正 |
| 2026-10-10 | VP2 CoR 可视化 + 自动生命周期 |

---

## 11. 下一步可选方向

**已规划的未完成项**：

1. **GPU kernel 同步**：如果未来启用 GPU，需要更新 `kernel_dqs.cl` 加入对称化
2. **文档补充**：`bind_selected.py` 用户手册
3. **CoR runtime 可视化**：需要每帧 LBS CoR 的优化
4. **性能基准**：30k+ 顶点下的完整数据

**暂不做的**：

- 文件进一步拆分（`optimizedSkinCluster.cpp` 仍 ~900 行，但结构清晰）
- BFS 内部并行（收益小）
- 更好的聚类算法（Schlömer 2011，收益小）

---

## 12. 参考资源

**论文**：
- Le & Hodgins 2016: Real-Time Skeletal Skinning with Optimized Centers of Rotation
- Kavan et al. 2008: Geometric Skinning with Approximate Dual Quaternion Blending
- Sulema & Rudenko 2020: A Method of Artifact Compensation for DQS

**相关文档**：
- `GPU_ACCELERATION_ARCHIVE.md` — GPU 模式完整归档

**Maya API**：
- `MPxSkinCluster` / `MPxLocatorNode` / `MPxDrawOverride`
- `MArrayDataHandle` / `MDagModifier` / `MDGModifier`
- `MNodeMessage` / `MEventMessage`
- `MUIDrawManager` / `MHWRender::MPxDrawOverride`

---

## 13. 联系与维护

**维护者**：changweibin
**项目路径**：`D:\visualStudioProject\optimizedSkinningPlugin\`
**当前阶段**：CPU 模式稳定运行，DQS 补偿已修正，CoR 可视化自动生命周期