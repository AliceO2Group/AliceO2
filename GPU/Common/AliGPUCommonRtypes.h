#ifndef ALIGPUCOMMONRTYPES_H
#define ALIGPUCOMMONRTYPES_H

#if defined(GPUCA_STANDALONE) || defined(GPUCA_O2_LIB) || defined(GPUCA_GPULIBRARY)
	#if !defined(ROOT_Rtypes) && !defined(__CLING__)
		#define ClassDef(name,id)
		#define ClassDefNV(name, id)
		#define ClassDefOverride(name, id)
		#define ClassImp(name)
		#ifndef GPUCA_GPUCODE_DEVICE
			typedef unsigned long long int ULong64_t;
			typedef unsigned int UInt_t;
			#include <iostream>
		#endif
	#endif
#else
	#include "Rtypes.h"
#endif

#endif
