#ifndef ALIGPUCASETTINGS_H
#define ALIGPUCASETTINGS_H

#include "AliTPCCommonMath.h"
class AliGPUCADisplayBackend;

class AliGPUCASettings
{
public:
};

//Settings concerning the reconstruction
struct AliGPUCASettingsRec
{
#ifndef GPUCA_GPUCODE
	AliGPUCASettingsRec() {SetDefaults();}
	void SetDefaults();
	void SetMinTrackPt( float v ){ MaxTrackQPt = CAMath::Abs(v)>0.001 ?1./CAMath::Abs(v) :1./0.001; }
#endif

	//There must be no bool in here, use char, as sizeof(bool) is compiler dependent and fails on GPUs!!!!!!
	float HitPickUpFactor;						// multiplier for the chi2 window for hit pick up procedure
	float NeighboursSearchArea;					// area in cm for the search of neighbours
	float ClusterError2CorrectionY;				// correction for the squared cluster error during tracking
	float ClusterError2CorrectionZ;				// correction for the squared cluster error during tracking
	int MinNTrackClusters;						//* required min number of clusters on the track
	float MaxTrackQPt;							//* required max Q/Pt (==min Pt) of tracks
	char NWays;									//Do N fit passes in final fit of merger
	char NWaysOuter;							//Store outer param
	char RejectMode;							//0: no limit on rejection or missed hits, >0: break after n rejected hits, <0: reject at max -n hits
	char GlobalTracking;						//Enable Global Tracking (prolong tracks to adjacent sectors to find short segments)
	float SearchWindowDZDR;						//Use DZDR window for seeding instead of vertex window
	float TrackReferenceX;						//Transport all tracks to this X after tracking (disabled if > 500)
	char NonConsecutiveIDs;						//Non-consecutive cluster IDs as in HLT, disables features that need access to slice data in TPC merger
	unsigned char DisableRefitAttachment;		//Bitmask to disable cluster attachment steps in refit: 1: attachment, 2: propagation, 4: loop following, 8: mirroring
};

//Settings describing the events / time frames
struct AliGPUCASettingsEvent
{
#ifndef GPUCA_GPUCODE
	AliGPUCASettingsEvent() {SetDefaults();}
	void SetDefaults();
#endif
	
	//All new members must be sizeof(int)/sizeof(float) for alignment reasons!
	float solenoidBz;							//solenoid field strength
	int constBz;								//for test-MC events with constant Bz
	int homemadeEvents;							//Toy-MC events
	int continuousMaxTimeBin;					//0 for triggered events, -1 for default of 23ms
};

//Settings defining the setup of the AliGPUReconstruction processing (basically selecting the device / class instance)
struct AliGPUCASettingsProcessing
{
#ifndef GPUCA_GPUCODE
	AliGPUCASettingsProcessing() {SetDefaults();}
	void SetDefaults();
#endif
	
	unsigned int deviceType;					//Device type, shall use AliGPUReconstructions::DEVICE_TYPE constants, e.g. CPU / CUDA
	char forceDeviceType;						//Fail if device initialization fails, otherwise falls back to CPU
};

//Settings steering the processing once the device was selected
struct AliGPUCASettingsDeviceProcessing
{
	#ifndef GPUCA_GPUCODE
		AliGPUCASettingsDeviceProcessing() {SetDefaults();}
		void SetDefaults();
	#endif
		
	int nThreads;								//Numnber of threads on CPU, 0 = auto-detect
	int deviceNum;								//Device number to use, in case the backend provides multiple devices (-1 = auto-select)
	int platformNum;							//Platform to use, in case the backend provides multiple platforms (-1 = auto-select)
	bool globalInitMutex;						//Global mutex to synchronize initialization over multiple instances
	bool gpuDeviceOnly;							//Use only GPU as device (i.e. no CPU for OpenCL)
	int nDeviceHelperThreads;					//Additional CPU helper-threads for CPU parts of processing with accelerator
	int debugLevel;								//Level of debug output (-1 = silent)
	int debugMask;								//Mask for debug output dumps to file
	bool comparableDebutOutput;					//Make CPU and GPU debug output comparable (sort / skip concurrent parts)
	int resetTimers;							//Reset timers every event
	AliGPUCADisplayBackend* eventDisplay;		//Run event display after processing, ptr to backend
	bool runQA;									//Run QA after processing
	int stuckProtection;						//Timeout in us, When AMD GPU is stuck, just continue processing and skip tracking, do not crash or stall the chain
	int memoryAllocationStrategy;				//0 = auto, 1 = new/delete per resource (default for CPU), 2 = big chunk single allocation (default for device)
	bool keepAllMemory;							//Allocate all memory on both device and host, and do not reuse
	int nStreams;								//Number of parallel GPU streams
	bool trackletConstructorInPipeline;			//Run tracklet constructor in pileline like the preceeding tasks instead of as one big block
	bool trackletSelectorInPipeline;			//Run tracklet selector in pipeline, requres also tracklet constructor in pipeline
};

#endif
