# GPU 加速架构归档

**归档日期**：2026-10-08
**归档原因**：CPU 模式优先（保留 Maya Skin 全部工具），GPU 架构暂时停用但完整保留供未来参考
**Maya 版本**：Maya 2024 (24.2.0.1186)
**DevKit**：C:\Users\changweibin\devkitBase

---

## 1. 成果概述

成功实现 **3 种论文算法**的 OpenCL GPU 加速：

| 算法 | 论文 | CPU vs GPU Max Diff | 状态 |
|---|---|---|---|
| **CoR** | Le & Hodgins 2016 | `0.000000` | ✅ 逐位一致 |
| **DLB** | Kavan 2008 | `0.000000` | ✅ 逐位一致 |
| **DQS 补偿** | Sulema & Rudenko 2020 | `0.000000` | ✅ 逐位一致 |

**性能**：30,060 顶点圆柱（`sx=60, sy=500`）场景下，GPU 相对 CPU **加速 3.20x**。

**验证环境**：
- Maya 2024 + Viewport 2.0
- GPU Override 开启（`optionVar -iv gpuOverride 1`）
- deformerEvaluator + GPUBuiltInDeformer 插件已加载

---

## 2. 架构

### 2.1 双节点分离

```
┌──────────────────────────────────────────────────────────────┐
│  原始网格 (origMesh)                                          │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌─────────────────────────────┐
│  Node A           │    │  Node B                     │
│  optimizedSkin-   │    │  optimizedSkinCluster-      │
│  Cluster          │    │  Compute                    │
│  (MPxSkinCluster) │    │  (MPxDeformerNode)          │
│                   │    │                             │
│  职责：           │    │  职责：                     │
│  · 提供 weightList│    │  · GPU/CPU 真变形           │
│  · 打包数据到     │    │  · 输出到 mesh              │
│    outData        │    │                             │
│  · 不参与显示     │    │                             │
└────────┬──────────┘    └──────────┬──────────────────┘
         │                          │
         │  outData → inData        │
         │  useGPU  → inUseGPU      │
         │  (MPxData 单标量传输)    │
         │                          │
         └────────► ┌───────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  meshShape.inMesh    │
        │  (最终显示网格)       │
        └──────────────────────┘
        
        旁路：A.outputGeometry → dummy mesh（隐藏）
```

### 2.2 关键设计决策

**为什么 A 走旁路（输出到 dummy mesh）？**

Maya 的 deformer evaluator 从 `meshShape.inMesh` 沿几何连接向上扫描：

```
meshShape.inMesh
    ↑
    B.outputGeometry       ← kDeformerNode，evaluator 认识
    ↑
    B.inputGeometry
    ↑
    A.outputGeometry       ← kSkinCluster 类型，evaluator 报错：
                              "Node type not registered"
```

**A 一旦出现在显示链上（上游或下游），整条链的 GPU 化被阻断。**

解决方案：**把 A 输出重定向到隐藏的 dummy mesh，让 B 直接从 origMesh 拿输入**。

```
显示链：origMesh → B.input → B.output → meshShape.inMesh     ← B 是唯一的 GPU 变形节点
旁路链：origMesh → A.input → A.output → dummy.inMesh         ← A 只打包数据
数据链：A.outData → B.inData
```

**这就是 GPU 能生效的关键。**

---

## 3. 数据流详解

### 3.1 A 侧每帧做的事

1. 读 `matrix[]` / `bindPreMatrix[]` → 骨骼姿态
2. 读 `weightList[i].weights[j]` → 每顶点权重
3. 计算 CoR（论文式 1 + 4，**缓存**，权重变化才重算）
4. 打包到 `SkinClusterData`：
   - `algorithmType`, `dqsStrength`, `useGPU`
   - `weightVersion`（缓存版本号）
   - `restList`, `poseList`（骨骼变换）
   - `vertices`（每顶点 ≤4 骨权重）
   - `cors`（CoR，仅 algo=0 时）
   - `boneDirs`, `jointPositions`（供 DQS 用）
5. 通过 `MFnPluginData` 打包成 `MObject`，`setMObject(aOutData)`

### 3.2 B 侧每帧做的事

1. 从 `inData` 读 `SkinClusterData*`
2. 判断 `weightVersion` 是否变化：
   - **相同** → 跳过顶点级 buffer 上传（省 ~2 MB / 帧）
   - **不同** → 重新上传 `boneIds` / `boneWts` / `boneCounts` / `cors`
3. 骨骼级 buffer 每帧上传（数据量小）
4. 选择 kernel（CoR / DLB / DQS 三个之一）
5. `enqueueKernel` → GPU 计算
6. `finishEvaluation` → 输出到 `outputGeometry`

### 3.3 数据结构（`skinClusterData.h`）

```cpp
struct RigidData {              // 12 bytes
    float qx, qy, qz, qw;       // 四元数
    float tx, ty, tz;           // 平移
};

struct Vec3Data {               // 12 bytes
    float x, y, z;
};

struct SkinVertex {             // 20 bytes
    int   boneIds[4];           // 前 4 大骨骼索引
    float weights[4];           // 对应权重
    int   boneCount;            // 实际骨骼数（1~4）
};

class SkinClusterData : public MPxData {
    static const MString typeName;   // "skinClusterData"
    static MTypeId       id;         // 0x00142A10
    
    int   algorithmType;             // 0=CoR, 1=DLB, 2=DQS
    float dqsStrength;
    bool  useGPU;
    int   weightVersion;             // 缓存版本号
    
    std::vector<RigidData>  restList;
    std::vector<RigidData>  poseList;
    std::vector<SkinVertex> vertices;
    std::vector<Vec3Data>   cors;              // CoR
    std::vector<Vec3Data>   boneDirs;          // 骨骼世界方向
    std::vector<Vec3Data>   jointPositions;    // 关节位置
};
```

---

## 4. 关键文件清单

### 4.1 已编译（CPU 模式）

| 文件 | 作用 |
|---|---|
| `src/plugin_main.cpp` | 插件入口（当前只注册 A） |
| `src/algorithms.cpp` | 三种算法的纯数学实现（CoR/DLB/DQS） |
| `src/algorithms.h` | 算法声明 |
| `src/math_core.h` | Vec3/Quat/RigidTransform 数学库 |
| `src/optimizedSkinCluster.cpp` | Node A（CPU 变形版，当前模式） |
| `src/optimizedSkinCluster.h` | Node A 声明 |
| `src/node_ids.h` | Maya 节点 ID 分配 |

### 4.2 保留但未编译（GPU 架构参考）

| 文件 | 作用 |
|---|---|
| `src/optimizedSkinClusterCompute.h` | Node B 声明 |
| `src/optimizedSkinClusterCompute.cpp` | Node B 的 CPU 实现（fallback） |
| `src/optimizedSkinClusterGPUDeformer.h` | B 的 GPU deformer 声明 |
| `src/optimizedSkinClusterGPUDeformer.cpp` | GPU deformer 实现 |
| `src/skinClusterData.h` | 数据传输类 |
| `src/skinClusterData.cpp` | 数据传输实现 |
| `src/kernels/kernel_cor.cl` | CoR OpenCL 内核 |
| `src/kernels/kernel_dlb.cl` | DLB OpenCL 内核 |
| `src/kernels/kernel_dqs.cl` | DQS 补偿 OpenCL 内核 |

### 4.3 辅助脚本

| 文件 | 作用 |
|---|---|
| `scripts/optimizedSkinBind.mel` | 一键建 A+B 数据链 |
| `scripts/optimizedSkinAutoWeight.mel` | 按 Y 高度自动分配权重 |
| `scripts/copy_skin_weights.py` | 从原生 skinCluster 拷贝权重到 A |
| `scripts/copy_skin_weights.mel` | MEL 封装（可选） |

---

## 5. 核心实现细节

### 5.1 Node A 的数据打包

```cpp
void OptimizedSkinCluster::fillOutData(
    MDataBlock& block, MItGeometry& iter, int algo) {

    // 读骨骼变换
    std::vector<mathcore::RigidTransform> rest_t, Cs_t;
    std::vector<mathcore::Quat> poseQ_t;
    readBoneTransforms(block, rest_t, Cs_t, poseQ_t);
    
    SkinClusterData* sd = new SkinClusterData();
    sd->algorithmType = algo;
    sd->dqsStrength = ...;
    sd->weightVersion = m_currentWeightVersion;
    
    // 打包骨骼数据
    // 打包顶点权重（rebuildVertices，带缓存）
    // 打包 CoR（仅 algo=0）
    // 打包 boneDirs / jointPositions（供 DQS）
    
    // 通过 MFnPluginData 打包成 MObject
    MFnPluginData fnData;
    MObject dataObj = fnData.create(SkinClusterData::id, &status);
    SkinClusterData* dst = dynamic_cast<SkinClusterData*>(fnData.data(&status));
    // copy 数据...
    
    MDataHandle outH = block.outputValue(aOutData, &status);
    outH.setMObject(dataObj);
    outH.setClean();
}
```

### 5.2 Node A 的缓存机制

**两个独立的缓存**：

1. **`m_vtxCache`** —— 顶点权重映射缓存
   - `weightVersion` 匹配 → 直接复用
   - 权重变化 → `setDependentsDirty` 里 `++m_currentWeightVersion` → 下一帧重建

2. **`m_corCache`** —— CoR 缓存
   - 权重变化 → `m_corCache.valid = false` → 下一帧重算
   - **重算代价 O(V × F)** → 30k 顶点约 3~10 秒 → **这是主要性能瓶颈**

**优化提示**：CoR 邻接图平滑 + 硬分组形心可大幅加速（见 §6.4）。

### 5.3 Node B 的缓存机制

```cpp
struct {
    int               m_lastWeightVersion = -1;   // 上次上传的版本号
    unsigned int      m_lastFullCount = 0;        // 上次上传的顶点数
    bool              m_vtxBuffersCached = false; // 顶点级 buffer 是否有效
};
```

**判断是否需要重传顶点级 buffer**：

```cpp
bool needVtxUpload = false;
if (!m_vtxBuffersCached)                                   needVtxUpload = true;
else if (sdCheck->weightVersion != m_lastWeightVersion)    needVtxUpload = true;
else if (full != m_lastFullCount)                          needVtxUpload = true;
else if (sdCheck->algorithmType != m_lastAlgo)             needVtxUpload = true;
```

### 5.4 三个 OpenCL kernel 结构

所有 kernel 接收相同的参数模板：

```c
__kernel void xxx_deform(
    __global const float* inPos,       // N*3  输入顶点位置
    __global float* outPos,            // N*3  输出顶点位置
    __global const int* boneIds,       // N*4  每顶点最多 4 骨索引
    __global const float* boneWts,     // N*4  对应权重
    __global const int* boneCounts,    // N    实际骨骼数
    __global const float* restQ,       // B*4  骨骼 rest 四元数
    __global const float* restT,       // B*3  骨骼 rest 平移
    __global const float* poseQ,       // B*4  骨骼 pose 四元数
    __global const float* poseT,       // B*3  骨骼 pose 平移
    __global const int* affectMap,     // A    受影响的顶点索引（可为 NULL）
    const int numAffect,               // kernel 执行次数
    const int hasAffectMap,            // 0/1
    const float envelope               // 混合系数
);
```

**CoR 特有的额外参数**：`cors`（N*3，每顶点 CoR）
**DQS 特有的额外参数**：`boneDirs`（B*3）、`jointPos`（B*3）、`numBones`、`dqsStrength`

### 5.5 Maya 2024 的 API 注意事项

**1. `MFnSkinCluster.getWeights()` 返回单数组（不是 tuple）**

```python
# ❌ 错
weights, num_inf = fn_skin.getWeights(dag, comp, indices)

# ✅ 对（参考 SkinWeightPainter.paint_pair）
weights = fn_skin.getWeights(dag, comp, indices)   # 单 MDoubleArray
num_inf = len(fn_skin.influenceObjects())          # 单独取
```

**2. `MFnSkinCluster.setWeights()` 参数顺序**

```python
# ❌ 错
fn_skin.setWeights(dag, weights, comp, normalize)

# ✅ 对
fn_skin.setWeights(dag, comp, indices, weights, normalize=False)
```

**3. `MFnTypedAttribute::create()` 需要 5 个参数**

```cpp
// ✅ 对（Maya 2024）
MObject attr = tAttr.create(
    "outData", "outd",
    SkinClusterData::id,
    MObject::kNullObj,
    &status);
```

**4. `MObject` 与 `MDagPath` 区分**

```python
# MFnMesh 构造需要 MDagPath
sel = om.MSelectionList()
sel.add(shape_name)
dag = sel.getDagPath(0)          # 返回 MDagPath
fn_mesh = om.MFnMesh(dag)
```

---

## 6. 踩过的坑与解决方案

### 6.1 崩溃问题 1：A+B 双节点导致 Maya 崩溃

**症状**：A 做 CPU 变形 + B 存在时，Maya 崩在 idle 时。崩溃栈顶是 `TidleAction::preDoIdleAction`。

**原因**：B 通过 `MArrayDataHandle::next()` 读 A 的属性时，**会触发对 A 的递归求值**。Maya 求值器状态被破坏。

**解决方案**：
- 改用 `jumpToArrayElement()` 按逻辑索引遍历，**不触发求值**
- 加索引保护 `if (idx == (unsigned)-1) continue;`（`elementIndex()` 可能返回无效值）

**更根本的方案**：改用 **`MPxData` 单标量传输**，所有数据打包成一个 MObject 通过 `setMObject` 传递 → **完全避免数组属性访问**。

### 6.2 崩溃问题 2：`MArrayDataHandle::next()` 触发递归

**代码片段**：

```cpp
// ❌ 危险
for (unsigned int i = 0; i < nMat; ++i) {
    unsigned int idx = matH.elementIndex();
    ...
    matH.next();   // ← 在 DG 求值中触发上游求值
}

// ✅ 安全
for (unsigned int i = 0; i < nMat; ++i) {
    if (!matH.jumpToArrayElement(i)) continue;
    unsigned int idx = matH.elementIndex();
    if (idx == static_cast<unsigned int>(-1)) continue;
    ...
}
```

### 6.3 崩溃问题 3：`MPxDeformerNode::deform` 必须遍历 `iter`

**症状**：直接 `return MS::kSuccess;` 会导致下游读到"未初始化几何" → 延迟崩溃。

**解决方案**：

```cpp
// 直通时必须遍历 iter 一遍
MPointArray allPos;
iter.allPositions(allPos);
iter.setAllPositions(allPos);
return MS::kSuccess;
```

### 6.4 性能问题：CoR 重算卡死

**症状**：改权重后第一次 `cmds.refresh()` 卡 3~10 秒。

**原因**：`computeCoRCache` 是 **O(V × F)** —— 30k 顶点 × 60k 三角形 ≈ 18 亿次相似度计算。

**解决方案**：
- 硬分组形心替代精确积分（快 100 倍）：
  ```cpp
  // 按权重分组 → 每组一个形心 → 组内顶点共享 CoR
  // 30k 顶点 CoR 重算 < 0.5 秒
  ```
- 或保留精确版本 + ANN / BFS / 边细分加速（论文 §3.2 的方案）

### 6.5 GPU 不生效问题

**症状**：`registerGPUDeformerCreator status = 0` 但 GPU deformer 从不被调用。

**排查清单**：

| 检查项 | 命令 | 期望 |
|---|---|---|
| deformerEvaluator 已加载 | `cmds.pluginInfo("deformerEvaluator", q=True, loaded=True)` | True |
| GPU Override 已开启 | `cmds.optionVar(q="gpuOverride")` | 1 |
| Evaluation Manager 是并行 | `cmds.evaluationManager(q=True, mode=True)` | `['parallel', ...]` |
| Viewport 是 VP2 | `cmds.modelEditor(p, q=True, rendererName=True)` | `vp2Renderer` |
| 节点有动画驱动 | 加 keyframe | 关键！ |

**关键发现**：
- **GPU Override 需要重启 Maya 才生效**（`turnOnOpenCLEvaluatorActive()` 是临时切换）
- **节点必须有 time-dependent 属性驱动**，否则 evaluator 判定 "Node not animated" 跳过
- **`deformerEvaluator -deformers`** 列出所有已注册的 GPU deformer
- **`deformerEvaluator -dch -v true`** 列出选中 mesh 的变形链及各节点状态

### 6.6 `Node type not registered` 问题

**症状**：A（自定义 `kSkinCluster` 类型）出现在显示链上 → deformer evaluator 报此错 → 整条链的 GPU 化被阻断。

**关键发现**：Maya 的 deformer evaluator **只认内置 skinCluster 类型 ID**，不认继承自它的自定义类型。

**解决方案**：A 必须走旁路（输出到 dummy mesh），B 直接从 origMesh 拿输入。

**这意味着 "Paint Skin Weights 工具 + GPU 加速" 在 Maya 2024 架构下不可兼得。**

---

## 7. 性能数据

### 7.1 CPU vs GPU 时间（30,060 顶点，50 帧）

| 场景 | CPU 时间 | GPU 时间 | 加速比 |
|---|---|---|---|
| CoR（无缓存优化） | 15.25 ms/frame | 6.62 ms/frame | 2.30x |
| CoR（缓存优化后） | 11.91 ms/frame | 6.32 ms/frame | 1.89x |
| **CoR（完整优化）** | **12.33 ms/frame** | **3.85 ms/frame** | **3.20x** |

### 7.2 顶点数对比（CoR 算法）

| 顶点数 | CPU 时间 | GPU 时间 | 加速比 |
|---|---|---|---|
| 6,012 | 1.55 ms/frame | 1.55 ms/frame | 1.00x（固定开销主导） |
| 30,060 | 12.33 ms/frame | 3.85 ms/frame | **3.20x** |
| 100,000+（预估） | ~40 ms/frame | ~5 ms/frame | **5~8x** |

### 7.3 缓存优化的效果

| 优化项 | 效果 |
|---|---|
| 顶点级 buffer 缓存（B 侧） | 每帧省 ~2 MB 上传 |
| 顶点权重映射缓存（A 侧） | 省 30k 次权重读取+排序 |
| CoR 缓存（A 侧） | 只在权重变化时重算 |
| **总计** | 30k 顶点下 ~40% 加速 |

---

## 8. 恢复 GPU 模式的步骤

### 步骤 1：恢复 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(optimizedSkinningPlugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include($ENV{DEVKIT_LOCATION}/cmake/pluginEntry.cmake)
add_compile_options(/utf-8)
link_directories($ENV{DEVKIT_LOCATION}/lib)

set(PROJECT_NAME optimizedSkinningPlugin)

set(SOURCE_FILES
    src/plugin_main.cpp
    src/skinClusterData.cpp
    src/algorithms.cpp
    src/optimizedSkinCluster.cpp
    src/optimizedSkinClusterCompute.cpp
    src/optimizedSkinClusterGPUDeformer.cpp
)

set(LIBRARIES
    OpenMaya
    Foundation
    OpenMayaAnim
    OpenMayaUI
    OpenMayaRender
    clew
)

build_plugin()

# 拷贝 OpenCL kernels
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMENT "Copying OpenCL kernels to output directory"
)

# 拷贝 MEL 脚本
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/scripts"
    COMMENT "Copying MEL scripts to output directory"
)
```

### 步骤 2：恢复 `plugin_main.cpp`

注册 A + B + Data + GPU deformer（详见历史版本）。

### 步骤 3：恢复 `optimizedSkinCluster.h/cpp` 为"只打包"模式

关键点：
- `deform()` 只调用 `fillOutData()`
- 添加 `aOutData`（`SkinClusterData::id` 类型）
- 添加 `aUseGPU`
- 添加 `aWeightVersion`
- 添加 `rebuildVertices` / `updateDynamic` / `computeCoRCache`

### 步骤 4：确保 `src/optimizedSkinClusterCompute.*` 和 `src/optimizedSkinClusterGPUDeformer.*` 存在

这些文件不应删除，从 Git 历史或备份恢复。

### 步骤 5：重启 Maya 并开启 GPU Override

```python
import maya.cmds as cmds
cmds.optionVar(iv=("gpuOverride", 1))
```

**必须重启 Maya** 才能让 GPU Override 生效。

### 步骤 6：加载插件并验证

```python
import maya.cmds as cmds
cmds.loadPlugin("D:/.../optimizedSkinningPlugin.mll")
# 期望：[optimizedSkinningPlugin] registerGPU(B) status = 0
#       [optimizedSkinningPlugin] A + B + Data + GPU(B) registered
```

### 步骤 7：搭建测试场景

见 §10 完整测试脚本。

---

## 9. Maya deformer evaluator 参考

### 9.1 常用调试命令

```mel
// 列出所有已注册的 GPU deformer 类型
deformerEvaluator -deformers;

// 列出当前在 GPU 上的节点
deformerEvaluator -ls;
deformerEvaluator -ls -m;    // 只列 mesh

// 查看选中 mesh 的变形链及各节点状态
select -r pCylinder1;
deformerEvaluator -dch -v true;

// 查看节点状态
deformerEvaluator -ns -txt;

// 查看所有活动节点
deformerEvaluator -act -dch;
```

### 9.2 常见状态码含义

| 状态 | 含义 |
|---|---|
| `"GPU Active"` | 节点在 GPU 上运行 |
| `"Node type not registered"` | 节点类型不在 evaluator 白名单 |
| `"Node not animated"` | 节点没有 time-dependent 属性驱动 |
| `"Mesh not animated"` | mesh 没有动画驱动 |
| `"Mesh has been assigned to a material..."` | 其他信息 |

### 9.3 GPU Override 系统设置

| 位置 | 命令 | 说明 |
|---|---|---|
| **optionVar** | `optionVar -iv gpuOverride 1` | 持久化，重启 Maya 生效 |
| **GUI** | Preferences → Display → Viewport 2.0 → GPU Override | 界面选项 |
| **临时切换** | `turnOnOpenCLEvaluatorActive()` | 当前会话有效 |
| **临时关闭** | `turnOffOpenCLEvaluatorActive()` | 当前会话有效 |

### 9.4 官方 simpleDeformer 参照

Maya devkit 里的官方 `simpleDeformerNode.cpp` 是验证环境是否支持自定义 GPU deformer 的**最小可运行例子**。如果官方例子都不能 GPU 化，说明环境有问题，与自定义代码无关。

---

## 10. 完整测试脚本

### 10.1 GPU 模式测试

```python
import maya.cmds as cmds
import maya.mel as mel
import time

PLUGIN = "D:/visualStudioProject/optimizedSkinningPlugin/build/Release/optimizedSkinningPlugin.mll"
cmds.loadPlugin(PLUGIN)

# 建场景
cmds.file(new=True, f=True)
cyl = cmds.polyCylinder(r=0.5, h=10, sx=60, sy=500, ch=1)[0]
meshShape = cmds.listRelatives(cyl, s=True)[0]
vtxCount = cmds.polyEvaluate(meshShape, v=True)

cmds.select(cl=True)
cmds.joint(p=(0,  5, 0), n="j_root")
cmds.joint(p=(0,  0, 0), n="j_mid")
cmds.joint(p=(0, -5, 0), n="j_tip")
joints = ["j_root", "j_mid", "j_tip"]

# 建 A
cmds.select(meshShape)
nodeA = cmds.deformer(type="optimizedSkinCluster")[0]
conns = cmds.listConnections(f"{nodeA}.input[0].inputGeometry", s=True, d=False, p=True)
origOutPlug = conns[0]

# A.output → dummy
cmds.disconnectAttr(f"{nodeA}.outputGeometry[0]", f"{meshShape}.inMesh")
dummyMesh = cmds.createNode("mesh", n="_optimizedSkin_dummy")
cmds.connectAttr(f"{nodeA}.outputGeometry[0]", f"{dummyMesh}.inMesh")
cmds.setAttr(f"{dummyMesh}.visibility", 0)

# 绑骨骼 + 权重
for i, j in enumerate(joints):
    if not cmds.objExists(f"{j}.lockInfluenceWeights"):
        cmds.addAttr(j, sn="liw", ln="lockInfluenceWeights", at="bool")
    cmds.connectAttr(f"{j}.liw", f"{nodeA}.lockWeights[{i}]", f=True)
    cmds.connectAttr(f"{j}.worldMatrix[0]", f"{nodeA}.matrix[{i}]", f=True)
    m = cmds.getAttr(f"{j}.wim")
    cmds.setAttr(f"{nodeA}.bindPreMatrix[{i}]", *m, type="matrix")

vertexY = []
for v in range(vtxCount):
    p = cmds.xform(f"{meshShape}.vtx[{v}]", q=True, os=True, t=True)
    vertexY.append(p[1])
yMin, yMax = min(vertexY), max(vertexY)
yRange = yMax - yMin
for v in range(vtxCount):
    for k in range(3):
        cmds.setAttr(f"{nodeA}.weightList[{v}].weights[{k}]", 0.0)
    t = (vertexY[v] - yMin) / yRange
    b = 0 if t > 0.66 else (2 if t < 0.33 else 1)
    cmds.setAttr(f"{nodeA}.weightList[{v}].weights[{b}]", 1.0)

# 建 B
nodeB = cmds.createNode("optimizedSkinClusterCompute")
cmds.connectAttr(origOutPlug, f"{nodeB}.input[0].inputGeometry")
cmds.connectAttr(f"{nodeB}.outputGeometry[0]", f"{meshShape}.inMesh")
cmds.connectAttr(f"{nodeA}.outData", f"{nodeB}.inData")
cmds.connectAttr(f"{nodeA}.useGPU",  f"{nodeB}.inUseGPU")
cmds.setAttr(f"{nodeA}.envelope", 1.0)
cmds.setAttr(f"{nodeB}.envelope", 1.0)

# 加 keyframe（GPU evaluator 前置条件）
cmds.setKeyframe("j_mid", at="rotateZ", t=1, v=0)
cmds.setKeyframe("j_mid", at="rotateZ", t=100, v=90)
cmds.playbackOptions(minTime=1, maxTime=100)

# 测 GPU
cmds.setAttr(f"{nodeA}.useGPU", 1)
cmds.refresh(f=True)

# 性能测试
t0 = time.time()
for i in range(50):
    cmds.setAttr("j_mid.rotateZ", (i % 90))
    cmds.refresh(f=True)
print(f"GPU 50 frames = {time.time()-t0:.3f}s")
```

### 10.2 deformerEvaluator 检查

```mel
// MEL
select -r pCylinderShape1;
deformerEvaluator -dch -v true;
```

**期望输出**：
```
Deformation chain of pCylinderShape1:
    polyCylinder1 : "Node type not registered"
    pCylinderShape1Orig : "Mesh not animated"
    optimizedSkinClusterCompute1 : "GPU Active"       ← ★
    pCylinderShape1 : "GPU Active"                    ← ★
```

---

## 11. 已知限制

| 限制 | 说明 | 影响 |
|---|---|---|
| **无 Paint Skin Weights 工具** | A 走旁路，Maya Skin 工具链找不到它 | 权重编辑需脚本化 |
| **无 Copy/Mirror Skin Weights** | 同上 | 跨网格权重迁移需脚本 |
| **CoR 重算慢** | O(V × F) 精确积分 | 改权重后第一帧卡 3~10 秒 |
| **需要 Maya 2024 + VP2** | 依赖新 API | 低版本不支持 |
| **需要 GPU Override 开启** | 系统设置 | 未开启时自动回退 CPU |
| **需要节点有动画驱动** | deformer evaluator 前置条件 | 静态场景可能不上 GPU |
| **需要重启 Maya 生效** | GPU Override 的选项 | 开发调试麻烦 |
| **A 类型注册为 `kSkinCluster`** | evaluator 不认 | 无法同时满足 Paint + GPU |

---

## 12. 未来改进方向

### 12.1 短期（1~2 天）

- **CoR 快速化**：硬分组形心替代精确积分（快 100 倍，质量损失小）
- **`copy_skin_weights.py` 完善**：支持 mesh 顶点重排、命名空间、部分影响拷贝
- **B 的 CPU/GPU 无缝切换**：`validateNodeValues` 中 `inUseGPU` 变化时立即重建缓存

### 12.2 中期（1 周）

- **DQS 补偿的 GPU 加速改进**：目前 3.20x 加速低于 CoR，因为 DQS 需要更多骨骼遍历
- **支持 affect map 的更细粒度缓存**：只上传变化的顶点子集
- **支持多 mesh 共享 A**：目前 A/B 一对一

### 12.3 长期（1 个月+）

- **A 改为 `MPxDeformerNode`**（放弃 `kSkinCluster` 类型）：
  - 手动实现 `weightList` / `weights` / `matrix` / `bindPreMatrix`
  - 让 A 和 B 都在显示链上，都注册 GPU deformer
  - **代价**：完全放弃 Maya 原生 Skin 工具链
- **实现自定义 Paint 工具**：用 `MPxContext` 做权重画笔
- **实现自定义权重工具集**：Copy / Mirror / Prune / Hammer / Normalize

### 12.4 替代方案

- **方案 A：双网格工作流**（推荐）
  - 编辑网格用原生 skinCluster（全部 Skin 工具可用）
  - 显示网格用 GPU 加速版
  - 通过 `copy_skin_weights.py` 桥接权重
  - **优点**：兼顾两者
  - **缺点**：需维护两个网格

- **方案 B：Maya 未来版本**
  - 等待 Maya 提供 `MPxGPUDeformer` 的更好 API
  - 或等 Maya 开放 Skin 工具的第三方扩展点

---

## 13. 版本历史

| 日期 | 里程碑 |
|---|---|
| 2026-09-28 | 项目初始化，A 单节点 CPU 版 |
| 2026-10-01 | 加入 A+B 双节点 + MPxData 传输 |
| 2026-10-03 | 解决 A+B 递归求值崩溃 |
| 2026-10-05 | GPU deformer 注册成功 |
| 2026-10-07 | GPU Override 开启，deformerEvaluator 识别 |
| 2026-10-08 | 三算法 GPU 全部通过，性能 3.20x |
| **2026-10-08** | **归档 GPU 模式，回归 CPU 模式** |

---

## 14. 参考资源

- **论文**
  - Le & Hodgins 2016: Real-Time Skeletal Skinning with Optimized Centers of Rotation
  - Kavan et al. 2008: Geometric Skinning with Approximate Dual Quaternion Blending
  - Sulema & Rudenko 2020: A Method of Artifact Compensation for DQS

- **Maya API 文档**
  - `MPxGPUStandardDeformer`
  - `MGPUDeformerRegistry`
  - `MFnSkinCluster`
  - `MPxData` / `MFnPluginData`
  - `MOpenCLBuffer` / `MOpenCLKernelInfo`

- **Maya devkit 示例**
  - `simpleDeformerNode.cpp`（GPU 加速入门）
  - `basicSkinCluster.cpp`（skinCluster 自定义节点）
  - `identityGeomFilter.cpp`（最简 deformer）

- **官方文档**
  - Maya 2024 API Reference
  - Deformer Evaluator 使用指南
  - OpenCL 集成指南

---

## 15. 联系与维护

**归档者**：changweibin
**归档背景**：CPU 模式优先（保留 Maya 全部 Skin 工具），GPU 架构保留供未来参考
**恢复时机**：
- 未来需要 GPU 加速且不需要 Maya Skin 工具时
- 或 Maya 新版本放宽了 `kSkinCluster` 类型限制时
- 或需要做 GPU/CPU 混合架构时

**联系方式**：项目仓库 `D:\visualStudioProject\optimizedSkinningPlugin\`