#ifndef ALITPCCOMMONRTYPES_H
#define ALITPCCOMMONRTYPES_H

#if defined(GPUCA_STANDALONE) || defined(__OPENCL__) || defined(GPUCA_GPULIBRARY)
	#if !defined(ROOT_Rtypes) && !defined(__CLING__)
		#define ClassDef(name,id)
		#define ClassDefNV(name, id)
		#define ClassDefOverride(name, id)
		#define ClassImp(name)
	#endif
#else
	#include "Rtypes.h"
#endif

#endif
