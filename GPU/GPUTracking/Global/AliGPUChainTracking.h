#ifndef ALIGPUCHAINTRACKING_H
#define ALIGPUCHAINTRACKING_H

#include "AliGPUChain.h"
#include "AliGPUReconstructionHelpers.h"
#include <atomic>
#include <array>

class AliGPUTPCSliceOutput;
class AliGPUTPCSliceOutTrack;
class AliGPUTPCSliceOutCluster;
class AliGPUTPCGMMergedTrack;
struct AliGPUTPCGMMergedTrackHit;
class AliGPUTRDTrackletWord;
class AliHLTTPCClusterMCLabel;
class AliGPUTPCMCInfo;
class AliGPUTRDTracker;
class AliGPUTPCGPUTracker;
struct AliGPUTPCClusterData;
struct AliHLTTPCRawCluster;
struct ClusterNativeAccessExt;
struct AliGPUTRDTrackletLabels;
class AliGPUDisplay;
class AliGPUQA;
class AliGPUTRDGeometry;

namespace o2 { namespace trd { class TRDGeometryFlat; }}
namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative; }}
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform; }}
using TPCFastTransform = ali_tpc_common::tpc_fast_transformation::TPCFastTransform;

class AliGPUChainTracking : public AliGPUChain, AliGPUReconstructionHelpers::helperDelegateBase
{
	friend class AliGPUReconstruction;
public:
	virtual ~AliGPUChainTracking();
	virtual void RegisterPermanentMemoryAndProcessors() override;
	virtual void RegisterGPUProcessors() override;
	virtual int Init() override;
	virtual int Finalize() override;
	virtual int RunStandalone() override;

	//Structures for input and output data
	struct InOutPointers
	{
		InOutPointers() : mcLabelsTPC(nullptr), nMCLabelsTPC(0), mcInfosTPC(nullptr), nMCInfosTPC(0),
			mergedTracks(nullptr), nMergedTracks(0), mergedTrackHits(nullptr), nMergedTrackHits(0),
			trdTracks(nullptr), nTRDTracks(0), trdTracklets(nullptr), nTRDTracklets(0), trdTrackletsMC(nullptr),
			nTRDTrackletsMC(0)
		{}
		InOutPointers(const InOutPointers&) = default;
		
		const AliGPUTPCClusterData* clusterData[NSLICES];
		unsigned int nClusterData[NSLICES];
		const AliHLTTPCRawCluster* rawClusters[NSLICES];
		unsigned int nRawClusters[NSLICES];
		const o2::TPC::ClusterNativeAccessFullTPC* clustersNative;
		const AliGPUTPCSliceOutTrack* sliceOutTracks[NSLICES];
		unsigned int nSliceOutTracks[NSLICES];
		const AliGPUTPCSliceOutCluster* sliceOutClusters[NSLICES];
		unsigned int nSliceOutClusters[NSLICES];
		const AliHLTTPCClusterMCLabel* mcLabelsTPC;
		unsigned int nMCLabelsTPC;
		const AliGPUTPCMCInfo* mcInfosTPC;
		unsigned int nMCInfosTPC;
		const AliGPUTPCGMMergedTrack* mergedTracks;
		unsigned int nMergedTracks;
		const AliGPUTPCGMMergedTrackHit* mergedTrackHits;
		unsigned int nMergedTrackHits;
		const GPUTRDTrack* trdTracks;
		unsigned int nTRDTracks;
		const AliGPUTRDTrackletWord* trdTracklets;
		unsigned int nTRDTracklets;
		const AliGPUTRDTrackletLabels* trdTrackletsMC;
		unsigned int nTRDTrackletsMC;friend class AliGPUReconstruction;
	} mIOPtrs;

	struct InOutMemory
	{
		InOutMemory();
		~InOutMemory();
		InOutMemory(InOutMemory&&);
		InOutMemory& operator=(InOutMemory&&);
		
		std::unique_ptr<AliGPUTPCClusterData[]> clusterData[NSLICES];
		std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSLICES];
		std::unique_ptr<o2::TPC::ClusterNative[]> clustersNative[NSLICES * GPUCA_ROW_COUNT];
		std::unique_ptr<AliGPUTPCSliceOutTrack[]> sliceOutTracks[NSLICES];
		std::unique_ptr<AliGPUTPCSliceOutCluster[]> sliceOutClusters[NSLICES];
		std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
		std::unique_ptr<AliGPUTPCMCInfo[]> mcInfosTPC;
		std::unique_ptr<AliGPUTPCGMMergedTrack[]> mergedTracks;
		std::unique_ptr<AliGPUTPCGMMergedTrackHit[]> mergedTrackHits;
		std::unique_ptr<GPUTRDTrack[]> trdTracks;
		std::unique_ptr<AliGPUTRDTrackletWord[]> trdTracklets;
		std::unique_ptr<AliGPUTRDTrackletLabels[]> trdTrackletsMC;
	} mIOMem;
	
	//Read / Dump / Clear Data
	void ClearIOPointers();
	void AllocateIOMemory();
	using AliGPUChain::DumpData;
	void DumpData(const char* filename);
	using AliGPUChain::ReadData;
	int ReadData(const char* filename);
	virtual void DumpSettings(const char* dir = "");
	virtual void ReadSettings(const char* dir = "");
	
	//Converter functions
	void ConvertNativeToClusterData();
	
	//Getters for external usage of tracker classes
	AliGPUTRDTracker* GetTRDTracker() {return &workers()->trdTracker;}
	AliGPUTPCTracker* GetTPCSliceTrackers() {return workers()->tpcTrackers;}
	const AliGPUTPCTracker* GetTPCSliceTrackers() const {return workers()->tpcTrackers;}
	const AliGPUTPCGMMerger& GetTPCMerger() const {return workers()->tpcMerger;}
	AliGPUTPCGMMerger& GetTPCMerger() {return workers()->tpcMerger;}
	AliGPUDisplay* GetEventDisplay() {return mEventDisplay.get();}
	const AliGPUQA* GetQA() const {return mQA.get();}
	AliGPUQA* GetQA() {return mQA.get();}
	
	//Processing functions
	int RunTPCTrackingSlices();
	virtual int RunTPCTrackingMerger();
	virtual int RunTRDTracking();
	int DoTRDGPUTracking();
	
	//Getters / setters for parameters
	const TPCFastTransform* GetTPCTransform() const {return mTPCFastTransform.get();}
	const AliGPUTRDGeometry* GetTRDGeometry() const {return (AliGPUTRDGeometry*) mTRDGeometry.get();}
	const ClusterNativeAccessExt* GetClusterNativeAccessExt() const {return mClusterNativeAccess.get();}
	void SetTPCFastTransform(std::unique_ptr<TPCFastTransform> tpcFastTransform);
	void SetTRDGeometry(const o2::trd::TRDGeometryFlat& geo);
	void LoadClusterErrors();
	const AliGPUTPCSliceOutput** SliceOutput() const {return (const AliGPUTPCSliceOutput**) &mSliceOutput;}
	
	const void* mConfigDisplay = nullptr;										//Abstract pointer to Standalone Display Configuration Structure
	const void* mConfigQA = nullptr;											//Abstract pointer to Standalone QA Configuration Structure

protected:
	struct AliGPUTrackingFlatObjects : public AliGPUProcessor
	{
		AliGPUChainTracking* fChainTracking = nullptr;
		TPCFastTransform* fTpcTransform = nullptr;
		char* fTpcTransformBuffer = nullptr;
		o2::trd::TRDGeometryFlat* fTrdGeometry = nullptr;
		void* SetPointersFlatObjects(void* mem);
		short mMemoryResFlat = -1;
	};
	
	struct eventStruct //Must consist only of void* ptr that will hold the GPU event ptrs!
	{
		void* selector[NSLICES];
		void* stream[GPUCA_MAX_STREAMS];
		void* init;
		void* constructor;
	};
	
	AliGPUChainTracking(AliGPUReconstruction* rec);
	
	int ReadEvent(int iSlice, int threadId);
	void WriteOutput(int iSlice, int threadId);
	int GlobalTracking(int iSlice, int threadId);
	
	int PrepareProfile();
	int DoProfile();

	//Pointers to tracker classes
	AliGPUTrackingFlatObjects mFlatObjectsShadow; //Host copy of flat objects that will be used on the GPU
	AliGPUTrackingFlatObjects mFlatObjectsDevice; //flat objects that will be used on the GPU
	AliGPUTPCSliceOutput* mSliceOutput[NSLICES];
	
	//Display / QA
	std::unique_ptr<AliGPUDisplay> mEventDisplay;
	bool mDisplayRunning = false;
	std::unique_ptr<AliGPUQA> mQA;
	bool mQAInitialized = false;

	//Ptr to reconstruction detecto objects
	std::unique_ptr<ClusterNativeAccessExt> mClusterNativeAccess;				//Internal memory for clusterNativeAccess
	std::unique_ptr<TPCFastTransform> mTPCFastTransform;						//Global TPC fast transformation object
	std::unique_ptr<o2::trd::TRDGeometryFlat> mTRDGeometry;						//TRD Geometry
	
	HighResTimer timerTPCtracking[NSLICES][10];
	eventStruct mEvents;
	std::ofstream mDebugFile;
	
#ifdef __ROOT__ //ROOT5 BUG: cint doesn't do volatile
#define volatile
#endif
	volatile int fSliceOutputReady;
	volatile char fSliceLeftGlobalReady[NSLICES];
	volatile char fSliceRightGlobalReady[NSLICES];
#ifdef __ROOT__
#undef volatile
#endif
	std::array<char, NSLICES> fGlobalTrackingDone;
	std::array<char, NSLICES> fWriteOutputDone;

private:
	int RunTPCTrackingSlices_internal();
	std::atomic_flag mLockAtomic = ATOMIC_FLAG_INIT;
	
	int HelperReadEvent(int iSlice, int threadId, AliGPUReconstructionHelpers::helperParam* par);
	int HelperOutput(int iSlice, int threadId, AliGPUReconstructionHelpers::helperParam* par);
};

#endif
