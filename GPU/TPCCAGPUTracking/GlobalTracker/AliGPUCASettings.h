#ifndef ALIGPUCASETTINGS_H
#define ALIGPUCASETTINGS_H

#include "AliTPCCommonMath.h"

class AliGPUCASettings
{
public:
};

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
	char GlobalTracking;				//Enable Global Tracking (prolong tracks to adjacent sectors to find short segments)
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
	
	unsigned int deviceType;
	char forceDeviceType;
};

struct AliGPUCASettingsDeviceProcessing
{
	#ifndef HLTCA_GPUCODE
		AliGPUCASettingsDeviceProcessing() {SetDefaults();}
		void SetDefaults();
	#endif
		
	int nThreads;						//Numnber of threads on CPU, 0 = auto-detect
	int deviceNum;						//Device number to use, in case the backend provides multiple devices (-1 = auto-select)
	int platformNum;					//Platform to use, in case the backend provides multiple platforms (-1 = auto-select)
	bool globalInitMutex;				//Global mutex to synchronize initialization over multiple instances
	int nDeviceHelperThreads;			//Additional CPU helper-threads for CPU parts of processing with accelerator
	int debugLevel;						//Level of debug output
	int debugMask;						//Mask for debug output dumps to file
	int resetTimers;					//Reset timers every event
	bool runEventDisplay;				//Run event display after processing
	bool runQA;							//Run QA after processing
	int stuckProtection;				//Timeout in us, When AMD GPU is stuck, just continue processing and skip tracking, do not crash or stall the chain
};

#endif
