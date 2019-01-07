#ifndef ALIGPUCADISPLAYCONFIG_H
#define ALIGPUCADISPLAYCONFIG_H

#if !defined(GPUCA_STANDALONE) || defined(GPUCA_BUILD_O2_LIB)
#define QCONFIG_CPP11_INIT
#endif
#include "cmodules/qconfig.h"

typedef structConfigGL AliGPUCADisplayConfig;

#endif
