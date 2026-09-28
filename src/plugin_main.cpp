#include <maya/MFnPlugin.h>
#include "optimizedSkinCluster.h"
#include "node_ids.h"

MStatus initializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj, "changweibin", "1.0", "Any");

    status = plugin.registerNode(
        "optimizedSkinCluster",
        OptimizedSkinCluster::id,
        OptimizedSkinCluster::creator,
        OptimizedSkinCluster::initialize,
        MPxNode::kSkinCluster   // 注册为 skinCluster 类型
    );
    if (!status) {
        status.perror("registerNode failed");
        return status;
    }
    return MS::kSuccess;
}

MStatus uninitializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj);

    status = plugin.deregisterNode(OptimizedSkinCluster::id);
    if (!status) {
        status.perror("deregisterNode failed");
        return status;
    }
    return MS::kSuccess;
}