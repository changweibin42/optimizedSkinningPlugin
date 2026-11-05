#pragma once
#include <cstdlib>

// ================================================================
// plugin_debug.h
//
// 环境变量 OPT_SKIN_DEBUG 控制诊断日志的开关。
//
// 说明：不再使用 static 缓存，每次调用实时读环境变量，
//       避免「Maya 已启动后设置 OPT_SKIN_DEBUG 不生效」的问题。
// ================================================================

inline bool optSkinDebugLog() {
    return (std::getenv("OPT_SKIN_DEBUG") != nullptr);
}