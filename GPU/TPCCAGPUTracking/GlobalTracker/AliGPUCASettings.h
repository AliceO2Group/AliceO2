#ifndef ALIGPUCASETTINGS_H
#define ALIGPUCASETTINGS_H

#include "AliTPCCommonMath.h"

struct AliGPUCASettingsRec
{
#ifndef HLTCA_GPUCODE
	AliGPUCASettingsRec() {SetDefaults();}
	void SetDefaults();
	void SetMinTrackPt( float v ){ MaxTrackQPt = CAMath::Abs(v)>0.001 ?1./CAMath::Abs(v) :1./0.001; }
#endif

	float HitPickUpFactor;				// multiplier for the chi2 window for hit pick up procedure
	float NeighboursSearchArea;			// area in cm for the search of neighbours
	float ClusterError2CorrectionY;		// correction for the squared cluster error during tracking
	float ClusterError2CorrectionZ;		// correction for the squared cluster error during tracking
	int MinNTrackClusters;				//* required min number of clusters on the track
	float MaxTrackQPt;					//* required max Q/Pt (==min Pt) of tracks
	char NWays;							//Do N fit passes in final fit of merger
	char NWaysOuter;					//Store outer param
	char RejectMode;					//0: no limit on rejection or missed hits, >0: break after n rejected hits, <0: reject at max -n hits
	float SearchWindowDZDR;				//Use DZDR window for seeding instead of vertex window
	float TrackReferenceX;				//Transport all tracks to this X after tracking (disabled if > 500)
};

struct AliGPUCASettingsEvent
{
#ifndef HLTCA_GPUCODE
	AliGPUCASettingsEvent() {SetDefaults();}
	void SetDefaults();
#endif
	
	//All new members must be sizeof(int)/sizeof(float) for alignment reasons!
	float solenoidBz;
	int constBz;
	int homemadeEvents;
	int continuousMaxTimeBin; //0 for triggered events, -1 for default of 23ms
};

struct AliGPUCASettingsProcessing
{
#ifndef HLTCA_GPUCODE
	AliGPUCASettingsProcessing() {SetDefaults();}
	void SetDefaults();
#endif
	
	int nThreads; //0 = auto-detect
};

#endif
