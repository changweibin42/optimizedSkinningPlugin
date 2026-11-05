#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include <maya/MStatus.h>
#include <maya/MString.h>
#include <maya/MDrawRegistry.h>

#include "optimizedSkinCluster.h"
#include "optimizedSkinClusterCmd.h"
#include "optimizedSkinClusterCoRViewer.h"
#include "optimizedSkinClusterCoRViewerOverride.h"
#include "node_ids.h"

MStatus initializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj, "changweibin", "1.0", "Any");

    status = plugin.registerNode(
        OptimizedSkinCluster::typeName,
        OptimizedSkinCluster::id,
        OptimizedSkinCluster::creator,
        OptimizedSkinCluster::initialize,
        MPxNode::kSkinCluster);
    if (!status) {
        status.perror("registerNode optimizedSkinCluster failed");
        return status;
    }

    status = plugin.registerNode(
        "optimizedSkinClusterCoRViewer",
        OptimizedSkinClusterCoRViewer::id,
        OptimizedSkinClusterCoRViewer::creator,
        OptimizedSkinClusterCoRViewer::initialize,
        MPxNode::kLocatorNode,
        &OptimizedSkinClusterCoRViewer::drawDbClassification);
    if (!status) {
        status.perror("registerNode CoRViewer failed");
        return status;
    }

    status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
        OptimizedSkinClusterCoRViewer::drawDbClassification,
        OptimizedSkinClusterCoRViewer::drawRegistrantId,
        OptimizedSkinClusterCoRViewerOverride::creator);
    if (!status) {
        status.perror("registerDrawOverrideCreator failed");
        return status;
    }

    status = plugin.registerCommand(
        "optimizedSkinClusterCoR",
        OptimizedSkinClusterCoRCmd::creator,
        OptimizedSkinClusterCoRCmd::newSyntax);
    if (!status) {
        status.perror("registerCommand failed");
        return status;
    }

    // ★ 注册 undo 回调
    registerCoRViewerUndoCallback();

    // 说明：MPxSkinCluster 派生节点的 weights 已由 Maya 自动注册为 paintable，
    // 显式调用 makePaintable 反而会因属性类型定义不匹配产生警告，
    // 故此处不再调用。

    MGlobal::displayInfo(
        "[optimizedSkinningPlugin] registered (skinCluster + CoR viewer + cmd + VP2 override)");
    return MS::kSuccess;
}

MStatus uninitializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj);

    // ★ 反注册 undo 回调
    unregisterCoRViewerUndoCallback();

    plugin.deregisterCommand("optimizedSkinClusterCoR");

    MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
        OptimizedSkinClusterCoRViewer::drawDbClassification,
        OptimizedSkinClusterCoRViewer::drawRegistrantId);

    plugin.deregisterNode(OptimizedSkinClusterCoRViewer::id);
    plugin.deregisterNode(OptimizedSkinCluster::id);

    return MS::kSuccess;
}