#pragma once

#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>

class OptimizedSkinClusterCoRCmd : public MPxCommand {
public:
    OptimizedSkinClusterCoRCmd() {}
    ~OptimizedSkinClusterCoRCmd() override {}

    static void* creator() { return new OptimizedSkinClusterCoRCmd(); }
    static MSyntax newSyntax();

    MStatus doIt(const MArgList& args) override;
    bool isUndoable() const override { return false; }
};

// 由 plugin_main.cpp 调用的注册 / 反注册函数
void registerCoRViewerUndoCallback();
void unregisterCoRViewerUndoCallback();