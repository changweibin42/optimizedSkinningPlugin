#pragma once

#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MUIDrawManager.h>
#include <maya/MFrameContext.h>
#include <maya/MPointArray.h>
#include <maya/MColor.h>
#include <maya/MDagPath.h>
#include <maya/MBoundingBox.h>

class OptimizedSkinClusterCoRViewerOverride
    : public MHWRender::MPxDrawOverride {
public:
    static MHWRender::MPxDrawOverride* creator(const MObject& obj);
    ~OptimizedSkinClusterCoRViewerOverride() override;

    MHWRender::DrawAPI supportedDrawAPIs() const override;

    bool isBounded(const MDagPath& objPath,
                   const MDagPath& cameraPath) const override;

    MBoundingBox boundingBox(const MDagPath& objPath,
                             const MDagPath& cameraPath) const override;

    bool hasUIDrawables() const override { return true; }

    MUserData* prepareForDraw(const MDagPath& objPath,
                              const MDagPath& cameraPath,
                              const MHWRender::MFrameContext& frameContext,
                              MUserData* oldData) override;

    void addUIDrawables(const MDagPath& objPath,
                        MHWRender::MUIDrawManager& drawManager,
                        const MHWRender::MFrameContext& frameContext,
                        const MUserData* data) override;

private:
    OptimizedSkinClusterCoRViewerOverride(const MObject& obj);
};