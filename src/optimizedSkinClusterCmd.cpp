#include "optimizedSkinClusterCmd.h"
#include "optimizedSkinCluster.h"
#include "optimizedSkinClusterCoRViewer.h"
#include "math_core.h"
#include "plugin_debug.h"

#include <maya/MArgList.h>
#include <maya/MArgParser.h>
#include <maya/MSyntax.h>
#include <maya/MSelectionList.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnDagNode.h>
#include <maya/MDagPath.h>
#include <maya/MDagModifier.h>
#include <maya/MDGModifier.h>
#include <maya/MNodeMessage.h>
#include <maya/MMessage.h>
#include <maya/MEventMessage.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MUuid.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MGlobal.h>
#include <maya/MDoubleArray.h>
#include <maya/MString.h>
#include <maya/MObject.h>

#include <vector>
#include <map>
#include <string>

// ================================================================
// static 状态：UUID 做 key
// ================================================================
namespace {
    std::map<std::string, MString>     g_skinUuidToViewerName;
    std::map<std::string, MCallbackId> g_skinUuidToCallbackId;
    MCallbackId g_undoCallbackId = 0;
}

static std::string getNodeUuidStr(const MObject& obj) {
    MFnDependencyNode fn(obj);
    MUuid u = fn.uuid();
    MString s = u.asString();
    return std::string(s.asChar());
}

// ================================================================
// 用 MDagModifier 删 viewer（不进 undo 栈，删整个 DAG 子树）
// ================================================================
static void deleteViewerByDagModifier(const MString& viewerName) {
    MSelectionList sl;
    if (!sl.add(viewerName)) return;

    MObject viewerObj;
    sl.getDependNode(0, viewerObj);
    if (viewerObj.isNull()) return;

    MDagModifier dagMod;
    dagMod.deleteNode(viewerObj);
    MStatus st = dagMod.doIt();
    if (st) {
        MGlobal::displayInfo(
            MString("CoR viewer deleted: ") + viewerName);
    }
    else {
        MGlobal::displayWarning(
            MString("Failed to delete CoR viewer: ") + viewerName);
    }
}

// ================================================================
// preRemoval 回调
// ================================================================
static void onSkinNodePreRemoval(MObject& node, void* /*clientData*/) {
    std::string key = getNodeUuidStr(node);

    auto it = g_skinUuidToViewerName.find(key);
    if (it == g_skinUuidToViewerName.end()) return;

    MString viewerName = it->second;

    // ★ 用 MDagModifier 直删，不进 undo 栈
    deleteViewerByDagModifier(viewerName);

    g_skinUuidToViewerName.erase(it);
    auto cbIt = g_skinUuidToCallbackId.find(key);
    if (cbIt != g_skinUuidToCallbackId.end()) {
        g_skinUuidToCallbackId.erase(cbIt);
    }
}

// ================================================================
// undo 事件回调：检查所有 skinCluster，残缺 viewer 自动重建
// ================================================================
static void onAfterUndo(void* /*clientData*/) {
    MItDependencyNodes it(MFn::kSkinClusterFilter);
    for (; !it.isDone(); it.next()) {
        MObject obj = it.thisNode();
        MFnDependencyNode fn(obj);
        OptimizedSkinCluster* skin =
            dynamic_cast<OptimizedSkinCluster*>(fn.userNode());
        if (!skin) continue;

        MString nodeName = fn.name();

        MPlug dp(obj, OptimizedSkinCluster::aDisplayCoR);
        if (dp.isNull() || !dp.asBool()) continue;

        MString viewerName = nodeName + "_corViewer";
        MString shapeName = viewerName + "Shape";

        // C++ 原生检查（不用 MEL，避免 format 和语法问题）
        MSelectionList slT;
        bool tExists = slT.add(viewerName);
        MSelectionList slS;
        bool sExists = slS.add(shapeName);

        if (tExists && sExists) continue;   // 完好

        // 残缺：用 MDagModifier 清残留（同时删 transform 和 shape）
        MSelectionList sl;
        if (sl.add(viewerName)) {
            MObject vObj;
            sl.getDependNode(0, vObj);
            if (!vObj.isNull()) {
                MDagModifier dagMod;
                dagMod.deleteNode(vObj);
                dagMod.doIt();
            }
        }
        // shape 也可能残留（孤儿）
        MSelectionList sl2;
        if (sl2.add(shapeName)) {
            MObject sObj;
            sl2.getDependNode(0, sObj);
            if (!sObj.isNull()) {
                MDagModifier dagMod;
                dagMod.deleteNode(sObj);
                dagMod.doIt();
            }
        }

        // ★ 直接创建，不走 idle
        MString createCmd;
        createCmd.format(
            "optimizedSkinClusterCoR -create -node ^1s;",
            nodeName);
        MGlobal::executeCommand(createCmd, false, false);

        MGlobal::displayInfo(
            MString("CoR viewer rebuilt for: ") + nodeName);
    }
}

void registerCoRViewerUndoCallback() {
    if (g_undoCallbackId == 0) {
        g_undoCallbackId = MEventMessage::addEventCallback(
            "Undo", onAfterUndo, nullptr);
        if (optSkinDebugLog()) {
            MGlobal::displayInfo(
                "[optimizedSkinningPlugin] CoR viewer undo callback registered");
        }
    }
}

void unregisterCoRViewerUndoCallback() {
    if (g_undoCallbackId != 0) {
        MMessage::removeCallback(g_undoCallbackId);
        g_undoCallbackId = 0;
    }
}

// ================================================================
// command
// ================================================================
MSyntax OptimizedSkinClusterCoRCmd::newSyntax() {
    MSyntax syntax;
    syntax.addFlag("-n", "-node", MSyntax::kString);
    syntax.addFlag("-c", "-create");
    syntax.addFlag("-d", "-delete");
    return syntax;
}

MStatus OptimizedSkinClusterCoRCmd::doIt(const MArgList& args) {
    MStatus status;
    MArgParser parser(newSyntax(), args, &status);
    if (!status) return status;

    if (!parser.isFlagSet("-node")) {
        MGlobal::displayError("optimizedSkinClusterCoR: -node required");
        return MS::kFailure;
    }

    MString nodeName;
    status = parser.getFlagArgument("-node", 0, nodeName);
    if (!status) return MS::kFailure;

    MString viewerName = nodeName + "_corViewer";

    // ============================================================
    // create
    // ============================================================
    if (parser.isFlagSet("-create")) {
        MSelectionList skinSel;
        if (!skinSel.add(nodeName)) {
            MGlobal::displayError(
                MString("optimizedSkinClusterCoR: node not found: ") + nodeName);
            return MS::kFailure;
        }
        MObject skinObj;
        skinSel.getDependNode(0, skinObj);
        std::string skinUuid = getNodeUuidStr(skinObj);

        // C++ 原生检查
        MSelectionList slT;
        bool tExists = slT.add(viewerName);
        MSelectionList slS;
        bool sExists = slS.add(viewerName + "Shape");

        if (tExists && sExists) {
            if (optSkinDebugLog()) {
                MGlobal::displayInfo(
                    MString("CoR viewer already exists: ") + viewerName);
            }
            if (g_skinUuidToCallbackId.find(skinUuid) ==
                g_skinUuidToCallbackId.end()) {
                MCallbackId cbId = MNodeMessage::addNodePreRemovalCallback(
                    skinObj, onSkinNodePreRemoval, nullptr);
                g_skinUuidToCallbackId[skinUuid] = cbId;
                g_skinUuidToViewerName[skinUuid] = viewerName;
            }
            return MS::kSuccess;
        }

        // 预先清理残留
        deleteViewerByDagModifier(viewerName);
        deleteViewerByDagModifier(viewerName + "Shape");

        MString tempShapeName = viewerName + "Shape";
        MString createMel;
        createMel.format(
            "createNode -n ^1s -skipSelect optimizedSkinClusterCoRViewer;",
            tempShapeName);
        MString createdShapeName = MGlobal::executeCommandStringResult(
            createMel, false, false);
        if (createdShapeName.length() == 0) {
            MGlobal::displayError("Failed to create viewer node");
            return MS::kFailure;
        }

        MSelectionList shapeSel;
        if (!shapeSel.add(createdShapeName)) {
            MGlobal::displayError(
                MString("Shape not found: ") + createdShapeName);
            return MS::kFailure;
        }

        MDagPath shapeDag;
        shapeSel.getDagPath(0, shapeDag);
        shapeDag.extendToShape();

        MFnDagNode shapeFn(shapeDag);
        MObject shapeObj = shapeDag.node();

        MObject transformObj = shapeFn.parent(0);
        if (transformObj.isNull()) {
            MGlobal::displayError("Viewer transform not found");
            return MS::kFailure;
        }

        MFnDagNode transformFn(transformObj);
        MStatus renameTStatus;
        transformFn.setName(viewerName, false, &renameTStatus);
        if (!renameTStatus) {
            MGlobal::displayWarning(
                MString("Failed to rename transform to ") + viewerName);
        }

        MStatus renameSStatus;
        shapeFn.setName(viewerName + "Shape", false, &renameSStatus);
        if (!renameSStatus) {
            MGlobal::displayWarning(MString("Failed to rename shape"));
        }

        MString actualTransformName = transformFn.name();

        MString reparentMel;
        reparentMel.format(
            "string $p[] = `listRelatives -p ^1s`;"
            "if (size($p) > 0 && $p[0] != \"world\") { parent -w ^1s; }",
            actualTransformName, actualTransformName);
        MGlobal::executeCommand(reparentMel, false, false);

        MFnDependencyNode skinFn(skinObj, &status);
        if (!status) {
            MGlobal::displayError("Failed to get MFnDependencyNode");
            return MS::kFailure;
        }

        MObject msgAttr = skinFn.attribute("message");
        if (msgAttr.isNull()) {
            MGlobal::displayError("Cannot find 'message' attribute");
            return MS::kFailure;
        }

        MPlug srcPlug(skinObj, msgAttr);
        MPlug dstPlug(shapeObj,
            OptimizedSkinClusterCoRViewer::aCorSourceNode);

        if (srcPlug.isNull() || dstPlug.isNull()) {
            MGlobal::displayError("Failed to get plugs");
            return MS::kFailure;
        }

        MDGModifier dgMod;
        status = dgMod.connect(srcPlug, dstPlug);
        if (!status) {
            MGlobal::displayError("Failed to connect");
            return MS::kFailure;
        }
        dgMod.doIt();

        MPlugArray checkConns;
        dstPlug.connectedTo(checkConns, true, false);
        if (checkConns.length() == 0) {
            MGlobal::displayError("Connection verification failed");
            return MS::kFailure;
        }

        auto oldIt = g_skinUuidToCallbackId.find(skinUuid);
        if (oldIt != g_skinUuidToCallbackId.end()) {
            MMessage::removeCallback(oldIt->second);
            g_skinUuidToCallbackId.erase(oldIt);
        }

        MCallbackId cbId = MNodeMessage::addNodePreRemovalCallback(
            skinObj, onSkinNodePreRemoval, nullptr);
        g_skinUuidToCallbackId[skinUuid] = cbId;
        g_skinUuidToViewerName[skinUuid] = actualTransformName;

        MGlobal::displayInfo(
            MString("CoR viewer created: ") + actualTransformName);
        return MS::kSuccess;
    }

    // ============================================================
    // delete
    // ============================================================
    if (parser.isFlagSet("-delete")) {
        deleteViewerByDagModifier(viewerName);
        deleteViewerByDagModifier(viewerName + "Shape");
        return MS::kSuccess;
    }

    // ============================================================
    // query
    // ============================================================
    MSelectionList sel;
    status = sel.add(nodeName);
    if (!status) {
        MGlobal::displayError(
            MString("optimizedSkinClusterCoR: node not found: ") + nodeName);
        return MS::kFailure;
    }

    MObject nodeObj;
    status = sel.getDependNode(0, nodeObj);
    if (!status) return MS::kFailure;

    MFnDependencyNode fn(nodeObj, &status);
    if (!status) return MS::kFailure;

    OptimizedSkinCluster* skin =
        dynamic_cast<OptimizedSkinCluster*>(fn.userNode());
    if (!skin) {
        MGlobal::displayError(
            "optimizedSkinClusterCoR: not an optimizedSkinCluster");
        return MS::kFailure;
    }

    std::vector<mathcore::Vec3> cors;
    if (!skin->getCoRData(cors)) {
        MGlobal::displayWarning(
            "optimizedSkinClusterCoR: CoR cache not valid");
        return MS::kFailure;
    }

    MDoubleArray result;
    result.setLength((unsigned int)cors.size() * 3);
    for (size_t i = 0; i < cors.size(); ++i) {
        result[i * 3 + 0] = cors[i].x;
        result[i * 3 + 1] = cors[i].y;
        result[i * 3 + 2] = cors[i].z;
    }

    setResult(result);
    return MS::kSuccess;
}