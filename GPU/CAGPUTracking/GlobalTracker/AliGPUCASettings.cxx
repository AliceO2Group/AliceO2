#include "AliGPUCASettings.h"
#include "AliGPUTPCSettings.h"
#include "AliGPUReconstruction.h"
#include <string.h>

void AliGPUCASettingsRec::SetDefaults()
{
	HitPickUpFactor = 2.;
	NeighboursSearchArea = 3.f;
	ClusterError2CorrectionY = 1.f;
	ClusterError2CorrectionZ = 1.f;
	MinNTrackClusters = -1;
	MaxTrackQPt = 1.f / MIN_TRACK_PT_DEFAULT;
	NWays = 1;
	NWaysOuter = false;
	RejectMode = 5;
	GlobalTracking = true;
	SearchWindowDZDR = 0.f;
	TrackReferenceX = 1000.f;
	NonConsecutiveIDs = false;
	DisableRefitAttachment = 0;
}

void AliGPUCASettingsEvent::SetDefaults()
{
	solenoidBz = -5.00668;
	constBz = 0;
	homemadeEvents = 0;
	continuousMaxTimeBin = 0;
}

void AliGPUCASettingsProcessing::SetDefaults()
{
	deviceType = AliGPUReconstruction::DeviceType::CPU;
	forceDeviceType = true;
}

void AliGPUCASettingsDeviceProcessing::SetDefaults()
{
	nThreads = 1;
	deviceNum = -1;
	platformNum = -1;
	globalInitMutex = false;
	gpuDeviceOnly = false;
	nDeviceHelperThreads = GPUCA_GPU_DEFAULT_HELPER_THREADS;
	debugLevel = -1;
	debugMask = -1;
	comparableDebutOutput = true;
	resetTimers = 1;
	eventDisplay = nullptr;
	runQA = false;
	stuckProtection = 0;
	memoryAllocationStrategy = 0;
	keepAllMemory = false;
	nStreams = 8;
	trackletConstructorInPipeline = true;
	trackletSelectorInPipeline = false;
	runTPCSliceTracker = true;
	runTPCMerger = true;
	runTRDTracker = true;
}
