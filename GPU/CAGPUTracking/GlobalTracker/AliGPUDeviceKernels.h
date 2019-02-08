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
#include "AliGPUTPCGMMergerGPU.cxx"
#include "AliGPUTRDTrackerGPU.cxx"
#endif

#ifdef GPUCA_BUILD_MERGER
	#include "AliGPUTPCGMMerger.h"
	#include "AliGPUTPCGMTrackParam.cxx"
	#include "AliGPUTPCGMPhysicalTrackModel.cxx"
	#include "AliGPUTPCGMPropagator.cxx"
#endif

#ifdef GPUCA_BUILD_TRD
	#include "AliGPUTRDTrack.cxx"
	#include "AliGPUTRDTracker.cxx"
	#include "AliGPUTRDTrackletWord.cxx"
	#include "TRDGeometryBase.cxx"
#endif

#ifdef GPUCA_BUILD_ITS
	#include "TrackerTraitsNV.cu"
	#include "Context.cu"
	#include "Stream.cu"
	#include "DeviceStoreNV.cu"
	#include "Utils.cu"
#endif

#endif
