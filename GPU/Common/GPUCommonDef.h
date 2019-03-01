#ifndef GPUCOMMONDEF_H
#define GPUCOMMONDEF_H

#if defined(__CINT__) || defined(__ROOTCINT__)
#define CON_DELETE
#define CON_DEFAULT
#define CONSTEXPR const
#else
#define CON_DELETE = delete
#define CON_DEFAULT = default
#define CONSTEXPR constexpr
#endif

#include "GPUCommonDefGPU.h"

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && !defined(GPUCA_ALIROOT_LIB)
	#define GPUCA_BUILD_MERGER
	#if defined(HAVE_O2HEADERS) && !defined(__HIPCC__)
		#define GPUCA_BUILD_TRD
	#endif
	#if defined(HAVE_O2HEADERS) && !defined(__HIPCC__)
		#define GPUCA_BUILD_ITS
	#endif
#endif

#if defined(GPUCA_STANDALONE) || defined(GPUCA_O2_LIB) || defined(GPUCA_GPULIBRARY)
	#define GPUCA_ALIGPUCODE
#endif

#endif
