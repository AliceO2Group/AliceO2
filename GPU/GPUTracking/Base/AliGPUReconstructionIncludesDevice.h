#ifndef ALIGPURECONSTRUCTIONINCLUDESDEVICE_H
#define ALIGPURECONSTRUCTIONINCLUDESDEVICE_H

#include "AliGPUTPCDef.h"

#include "AliGPUTPCTrackParam.cxx"
#include "AliGPUTPCTrack.cxx"
#include "AliGPUTPCHitArea.cxx"
#include "AliGPUTPCGrid.cxx"
#include "AliGPUTPCRow.cxx"
#include "AliGPUParam.cxx"
#include "AliGPUTPCTracker.cxx"

#include "AliGPUGeneralKernels.cxx"

#include "AliGPUTPCTrackletSelector.cxx"
#include "AliGPUTPCNeighboursFinder.cxx"
#include "AliGPUTPCNeighboursCleaner.cxx"
#include "AliGPUTPCStartHitsFinder.cxx"
#include "AliGPUTPCStartHitsSorter.cxx"
#include "AliGPUTPCTrackletConstructor.cxx"

#ifdef GPUCA_BUILD_MERGER
	#include "AliGPUTPCGMMergerGPU.cxx"
	
	#include "AliGPUTPCGMMerger.h"
	#include "AliGPUTPCGMTrackParam.cxx"
	#include "AliGPUTPCGMPhysicalTrackModel.cxx"
	#include "AliGPUTPCGMPropagator.cxx"
#endif

#ifdef GPUCA_BUILD_TRD
	#include "AliGPUTRDTrackerGPU.cxx"

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
