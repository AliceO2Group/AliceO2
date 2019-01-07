#ifndef ALITPCCOMMONFAIRLOGGER_H
#define ALITPCCOMMONFAIRLOGGER_H

#if (defined(GPUCA_STANDALONE) && !defined(GPUCA_BUILD_O2_LIB)) || defined(__OPENCL__) || defined(GPUCA_GPULIBRARY)

#include <iostream>
#define LOG(type) std::cout
namespace FairLogger {
	const char* endl = "\n";
}

#else

#include <FairLogger.h>

#endif

#endif
