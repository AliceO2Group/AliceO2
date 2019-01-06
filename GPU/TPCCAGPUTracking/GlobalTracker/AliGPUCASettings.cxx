#include "AliGPUCASettings.h"
#include "AliHLTTPCCASettings.h"
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
	NWaysOuter = 0;
	RejectMode = 5;
	GlobalTracking = 1;
	SearchWindowDZDR = 0.f;
	TrackReferenceX = 1000.f;
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
	nDeviceHelperThreads = HLTCA_GPU_DEFAULT_HELPER_THREADS;
	debugLevel = 0;
	debugMask = -1;
	resetTimers = 1;
	eventDisplay = nullptr;
	runQA = false;
	stuckProtection = 0;
}
