#include "optimizedSkinClusterCoRViewerOverride.h"
#include "optimizedSkinClusterCoRViewer.h"
#include "optimizedSkinCluster.h"
#include "math_core.h"

#include <maya/MFnDependencyNode.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MPoint.h>
#include <maya/MString.h>

#include <vector>


namespace {

    class CoRViewerDrawData : public MUserData {
    public:
        CoRViewerDrawData() : MUserData(false) {}
        ~CoRViewerDrawData() override {}

        MPointArray points;
        float pointSize = 4.0f;
        MColor color = MColor(1.0f, 0.0f, 0.0f);
    };

} // anonymous namespace

MHWRender::MPxDrawOverride* OptimizedSkinClusterCoRViewerOverride::creator(
    const MObject& obj) {
    return new OptimizedSkinClusterCoRViewerOverride(obj);
}

OptimizedSkinClusterCoRViewerOverride::OptimizedSkinClusterCoRViewerOverride(
    const MObject& obj)
    : MHWRender::MPxDrawOverride(obj, nullptr) {}

OptimizedSkinClusterCoRViewerOverride::~OptimizedSkinClusterCoRViewerOverride() {}

MHWRender::DrawAPI
OptimizedSkinClusterCoRViewerOverride::supportedDrawAPIs() const {
    return MHWRender::kOpenGL | MHWRender::kDirectX11 |
        MHWRender::kOpenGLCoreProfile;
}

bool OptimizedSkinClusterCoRViewerOverride::isBounded(
    const MDagPath& /*objPath*/,
    const MDagPath& /*cameraPath*/) const {
    return false;
}

MBoundingBox OptimizedSkinClusterCoRViewerOverride::boundingBox(
    const MDagPath& /*objPath*/,
    const MDagPath& /*cameraPath*/) const {
    return MBoundingBox(MPoint(-1e5, -1e5, -1e5),
        MPoint(1e5, 1e5, 1e5));
}

MUserData* OptimizedSkinClusterCoRViewerOverride::prepareForDraw(
    const MDagPath& objPath,
    const MDagPath& /*cameraPath*/,
    const MHWRender::MFrameContext& /*frameContext*/,
    MUserData* oldData) {

    CoRViewerDrawData* data = dynamic_cast<CoRViewerDrawData*>(oldData);
    if (!data) {
        data = new CoRViewerDrawData();
    }

    data->points.clear();
    data->pointSize = 4.0f;
    data->color = MColor(1.0f, 0.0f, 0.0f);

    MObject thisNode = objPath.node();
    if (thisNode.isNull()) return data;

    // 读 viewer 属性
    MPlug psPlug(thisNode, OptimizedSkinClusterCoRViewer::aPointSize);
    MPlug crPlug(thisNode, OptimizedSkinClusterCoRViewer::aColorR);
    MPlug cgPlug(thisNode, OptimizedSkinClusterCoRViewer::aColorG);
    MPlug cbPlug(thisNode, OptimizedSkinClusterCoRViewer::aColorB);
    MPlug mpPlug(thisNode, OptimizedSkinClusterCoRViewer::aMaxPoints);

    if (!psPlug.isNull()) data->pointSize = (float)psPlug.asDouble();
    float cr = crPlug.isNull() ? 1.0f : (float)crPlug.asDouble();
    float cg = cgPlug.isNull() ? 0.0f : (float)cgPlug.asDouble();
    float cb = cbPlug.isNull() ? 0.0f : (float)cbPlug.asDouble();
    data->color = MColor(cr, cg, cb);
    int maxPts = mpPlug.isNull() ? 20000 : mpPlug.asInt();

    // 拿 skinCluster 连接
    MPlug srcPlug(thisNode, OptimizedSkinClusterCoRViewer::aCorSourceNode);
    if (srcPlug.isNull()) return data;

    MPlugArray conns;
    srcPlug.connectedTo(conns, true, false);
    if (conns.length() == 0) return data;

    MObject skinObj = conns[0].node();

    // 检查 displayCoR
    MPlug dispPlug(skinObj, OptimizedSkinCluster::aDisplayCoR);
    if (dispPlug.isNull() || !dispPlug.asBool()) return data;

    MFnDependencyNode skinFn(skinObj);
    OptimizedSkinCluster* skin =
        dynamic_cast<OptimizedSkinCluster*>(skinFn.userNode());
    if (!skin) return data;

    std::vector<mathcore::Vec3> cors;
    if (!skin->getCoRData(cors) || cors.empty()) return data;

    size_t n = cors.size();
    size_t step = 1;
    if (maxPts > 0 && (int)n > maxPts) {
        step = (n + maxPts - 1) / maxPts;
    }

    size_t count = (n + step - 1) / step;
    data->points.setLength((unsigned int)count);

    for (size_t i = 0, k = 0; i < n; i += step, ++k) {
        data->points[k] = MPoint(cors[i].x, cors[i].y, cors[i].z);
    }

    return data;
}

void OptimizedSkinClusterCoRViewerOverride::addUIDrawables(
    const MDagPath& /*objPath*/,
    MHWRender::MUIDrawManager& drawManager,
    const MHWRender::MFrameContext& /*frameContext*/,
    const MUserData* data) {

    const CoRViewerDrawData* d = dynamic_cast<const CoRViewerDrawData*>(data);
    if (!d) return;
    if (d->points.length() == 0) return;

    drawManager.beginDrawable();
    drawManager.setColor(d->color);
    drawManager.setPointSize(d->pointSize);
    drawManager.setDepthPriority(5);
    drawManager.points(d->points, false);
    drawManager.endDrawable();
}