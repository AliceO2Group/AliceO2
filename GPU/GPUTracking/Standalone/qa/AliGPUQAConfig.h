#ifndef ALIGPUQACONFIG_H
#define ALIGPUQACONFIG_H

#if !defined(GPUCA_STANDALONE)
#define QCONFIG_CPP11_INIT
#endif
#include "utils/qconfig.h"

typedef structConfigQA AliGPUQAConfig;

#endif
