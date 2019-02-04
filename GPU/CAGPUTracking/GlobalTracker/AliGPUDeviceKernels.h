#ifndef ALIGPUDEVICEKERNELS_H
#define ALIGPUDEVICEKERNELS_H

#include "AliGPUTPCDef.h"

#include "AliGPUTPCTrackParam.cxx"
#include "AliGPUTPCTrack.cxx"

#include "AliGPUTPCHitArea.cxx"
#include "AliGPUTPCGrid.cxx"
#include "AliGPUTPCRow.cxx"
#include "AliGPUCAParam.cxx"
#include "AliGPUTPCTracker.cxx"

#include "AliGPUTPCTrackletSelector.cxx"
#include "AliGPUTPCNeighboursFinder.cxx"
#include "AliGPUTPCNeighboursCleaner.cxx"
#include "AliGPUTPCStartHitsFinder.cxx"
#include "AliGPUTPCStartHitsSorter.cxx"
#include "AliGPUTPCTrackletConstructor.cxx"

#include "AliGPUGeneralKernels.cxx"

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
	#ifdef GPUCA_GPU_MERGER
		#include "AliGPUTPCGMMerger.h"
		#include "AliGPUTPCGMTrackParam.cxx"
		#include "AliGPUTPCGMPhysicalTrackModel.cxx"
		#include "AliGPUTPCGMPropagator.cxx"

		#include "AliGPUTRDTrack.cxx"
		#include "AliGPUTRDTracker.cxx"
		#include "AliGPUTRDTrackletWord.cxx"
		#ifdef HAVE_O2HEADERS
			#include "TRDGeometryBase.cxx"
		#endif
	#endif

	#if defined(HAVE_O2HEADERS) && !defined(GPUCA_O2_LIB)
		#include "TrackerTraitsNV.cu"
		#include "Context.cu"
		#include "Stream.cu"
		#include "DeviceStoreNV.cu"
		#include "Utils.cu"
	#endif
#endif

#endif
