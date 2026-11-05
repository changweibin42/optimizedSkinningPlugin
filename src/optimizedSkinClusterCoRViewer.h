#pragma once

#include <maya/MPxLocatorNode.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include <maya/MStatus.h>
#include <maya/MString.h>
#include <maya/M3dView.h>
#include <maya/MBoundingBox.h>
#include <maya/MDagPath.h>
#include <maya/MFnDependencyNode.h>

class OptimizedSkinClusterCoRViewer : public MPxLocatorNode {
public:
    OptimizedSkinClusterCoRViewer() {}
    ~OptimizedSkinClusterCoRViewer() override {}

    static void* creator() { return new OptimizedSkinClusterCoRViewer(); }
    static MStatus initialize();

    // legacy VP 用的 draw（保留，但 VP2 走 override）
    void draw(M3dView& view, const MDagPath& path,
        M3dView::DisplayStyle style, M3dView::DisplayStatus status) override;

    bool isBounded() const override { return false; }
    MBoundingBox boundingBox() const override;

    static MTypeId id;

    // ★ DrawOverride 分类
    static MString drawDbClassification;
    static MString drawRegistrantId;

    static MObject aCorSourceNode;
    static MObject aPointSize;
    static MObject aColorR;
    static MObject aColorG;
    static MObject aColorB;
    static MObject aMaxPoints;
};