#ifdef _WIN32
#pragma comment(lib, "opengl32.lib")
#endif

#include "optimizedSkinClusterCoRViewer.h"
#include "optimizedSkinCluster.h"
#include "math_core.h"

#include <maya/MFnMessageAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MGlobal.h>

#include <vector>

MTypeId OptimizedSkinClusterCoRViewer::id(0x00142A11);

// ★ DrawOverride 分类
MString OptimizedSkinClusterCoRViewer::drawDbClassification(
    "drawdb/geometry/optimizedSkinClusterCoRViewer");
MString OptimizedSkinClusterCoRViewer::drawRegistrantId(
    "optimizedSkinClusterCoRViewer");

MObject OptimizedSkinClusterCoRViewer::aCorSourceNode;
MObject OptimizedSkinClusterCoRViewer::aPointSize;
MObject OptimizedSkinClusterCoRViewer::aColorR;
MObject OptimizedSkinClusterCoRViewer::aColorG;
MObject OptimizedSkinClusterCoRViewer::aColorB;
MObject OptimizedSkinClusterCoRViewer::aMaxPoints;

MStatus OptimizedSkinClusterCoRViewer::initialize() {
    MStatus status;
    MFnMessageAttribute mAttr;
    MFnNumericAttribute nAttr;

    aCorSourceNode = mAttr.create("corSourceNode", "corSrc", &status);
    if (!status) return status;
    mAttr.setStorable(true);
    mAttr.setReadable(true);
    mAttr.setWritable(true);
    status = addAttribute(aCorSourceNode);
    if (!status) return status;

    aPointSize = nAttr.create("pointSize", "ptsz",
        MFnNumericData::kDouble, 4.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(1.0);
    nAttr.setMax(20.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    addAttribute(aPointSize);

    aColorR = nAttr.create("colorR", "colr",
        MFnNumericData::kDouble, 1.0, &status);
    nAttr.setMin(0.0); nAttr.setMax(1.0);
    nAttr.setKeyable(true); nAttr.setStorable(true);
    addAttribute(aColorR);

    aColorG = nAttr.create("colorG", "colg",
        MFnNumericData::kDouble, 0.0, &status);
    nAttr.setMin(0.0); nAttr.setMax(1.0);
    nAttr.setKeyable(true); nAttr.setStorable(true);
    addAttribute(aColorG);

    aColorB = nAttr.create("colorB", "colb",
        MFnNumericData::kDouble, 0.0, &status);
    nAttr.setMin(0.0); nAttr.setMax(1.0);
    nAttr.setKeyable(true); nAttr.setStorable(true);
    addAttribute(aColorB);

    aMaxPoints = nAttr.create("maxPoints", "mxpt",
        MFnNumericData::kInt, 20000, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(100);
    nAttr.setMax(1000000);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    addAttribute(aMaxPoints);

    return MS::kSuccess;
}

MBoundingBox OptimizedSkinClusterCoRViewer::boundingBox() const {
    return MBoundingBox(MPoint(-1e5, -1e5, -1e5),
        MPoint(1e5, 1e5, 1e5));
}

// ================================================================
// legacy viewport draw（保留兼容）
// ================================================================
void OptimizedSkinClusterCoRViewer::draw(
    M3dView& view,
    const MDagPath& /*path*/,
    M3dView::DisplayStyle /*style*/,
    M3dView::DisplayStatus /*status*/) {

    MObject thisNode = thisMObject();

    MPlug inPlug(thisNode, aCorSourceNode);
    MPlugArray conns;
    if (!inPlug.connectedTo(conns, true, false) || conns.length() == 0)
        return;

    MObject skinObj = conns[0].node();
    MFnDependencyNode skinFn(skinObj);
    OptimizedSkinCluster* skin =
        dynamic_cast<OptimizedSkinCluster*>(skinFn.userNode());
    if (!skin) return;

    MPlug displayPlug(skinObj, OptimizedSkinCluster::aDisplayCoR);
    if (displayPlug.isNull() || !displayPlug.asBool())
        return;

    std::vector<mathcore::Vec3> cors;
    if (!skin->getCoRData(cors)) return;
    if (cors.empty()) return;

    double ps = MPlug(thisNode, aPointSize).asDouble();
    double cr = MPlug(thisNode, aColorR).asDouble();
    double cg = MPlug(thisNode, aColorG).asDouble();
    double cb = MPlug(thisNode, aColorB).asDouble();
    int maxPts = MPlug(thisNode, aMaxPoints).asInt();

    size_t n = cors.size();
    size_t step = 1;
    if (maxPts > 0 && (int)n > maxPts) {
        step = (n + maxPts - 1) / maxPts;
    }

    view.beginGL();
    glPushAttrib(GL_CURRENT_BIT | GL_POINT_BIT | GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glPointSize((float)ps);
    glColor3f((float)cr, (float)cg, (float)cb);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < n; i += step) {
        glVertex3d(cors[i].x, cors[i].y, cors[i].z);
    }
    glEnd();
    glPopAttrib();
    view.endGL();
}