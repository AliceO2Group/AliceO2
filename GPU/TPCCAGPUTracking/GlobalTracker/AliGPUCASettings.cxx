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
	nThreads = 0;
	deviceType = AliGPUReconstruction::DeviceType::CPU;
	forceDeviceType = true;
}
