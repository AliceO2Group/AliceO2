#ifndef ALICAGPULOGGING_H
#define ALICAGPULOGGING_H

#ifdef HLTCA_BUILD_ALIROOT_LIB
#include "AliHLTLogging.h"
#define AliCAGPULogging AliHLTLogging
#else

class AliCAGPULogging
{
public:
	virtual ~AliCAGPULogging() {};
};

#define HLTError(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTWarning(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTInfo(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTImportant(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTDebug(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTFatal(...) {printf(__VA_ARGS__);printf("\n");exit(1);}

#endif
#endif //ALICAGPULOGGING_H
