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
	nThreads = 0;
	deviceNum = -1;
	platformNum = -1;
	nDeviceHelperThreads = 1;
	debugLevel = 0;
	runEventDisplay = 0;
	runQA = 0;
}
