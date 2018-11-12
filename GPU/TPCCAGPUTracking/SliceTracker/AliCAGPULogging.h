#ifndef ALICAGPULOGGING_H
#define ALICAGPULOGGING_H

#if defined(HLTCA_BUILD_ALIROOT_LIB) && defined(HLTCA_GPULIBRARY)
#warning ALIROOT LOGGING DISABLED FOR GPU TRACKING, CUDA incompatible to C++17 ROOT
#endif

#if defined(HLTCA_BUILD_ALIROOT_LIB) && !defined(HLTCA_GPULIBRARY) && !defined(HLTCA_LOGGING_PRINTF)
#include "AliHLTLogging.h"
#define AliCAGPULogging AliHLTLogging
#define CAGPUError(...) HLTError(__VA_ARGS__)
#define CAGPUWarning(...) HLTWarning(__VA_ARGS__)
#define CAGPUInfo(...) HLTInfo(__VA_ARGS__)
#define CAGPUImportant(...) HLTImportant(__VA_ARGS__)
#define CAGPUDebug(...) HLTDebug(__VA_ARGS__)
#define CAGPUFatal(...) HLTFatal(__VA_ARGS__)
#else

#define AliCAGPULogging AliCAGPULoggingFake

class AliCAGPULoggingFake
{
public:
	virtual ~AliCAGPULoggingFake() {};
};

#define CAGPUError(...) {printf(__VA_ARGS__);printf("\n");}
#define CAGPUWarning(...) {printf(__VA_ARGS__);printf("\n");}
#ifdef HLTCA_BUILD_O2_LIB
#define CAGPUInfo(...) {} //{printf(__VA_ARGS__);printf("\n");}
#else
#define CAGPUInfo(...) {printf(__VA_ARGS__);printf("\n");}
#endif
#define CAGPUImportant(...) {printf(__VA_ARGS__);printf("\n");}
#define CAGPUDebug(...) {} //{printf(__VA_ARGS__);printf("\n");}
#define CAGPUFatal(...) {printf(__VA_ARGS__);printf("\n");exit(1);}

#endif

#endif //ALICAGPULOGGING_H
