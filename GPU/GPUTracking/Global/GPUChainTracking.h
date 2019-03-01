// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTracking.h
/// \author David Rohr

#ifndef GPUCHAINTRACKING_H
#define GPUCHAINTRACKING_H

#include "GPUChain.h"
#include "GPUReconstructionHelpers.h"
#include <atomic>
#include <array>

class GPUTPCSliceOutput;
class GPUTPCSliceOutTrack;
class GPUTPCSliceOutCluster;
class GPUTPCGMMergedTrack;
struct GPUTPCGMMergedTrackHit;
class GPUTRDTrackletWord;
class AliHLTTPCClusterMCLabel;
class GPUTPCMCInfo;
class GPUTRDTracker;
class GPUTPCGPUTracker;
struct GPUTPCClusterData;
struct AliHLTTPCRawCluster;
struct ClusterNativeAccessExt;
struct GPUTRDTrackletLabels;
class GPUDisplay;
class GPUQA;
class GPUTRDGeometry;

namespace o2 { namespace trd { class TRDGeometryFlat; }}
namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative; }}
namespace ali_tpc_common { namespace tpc_fast_transformation { class TPCFastTransform; }}
using TPCFastTransform = ali_tpc_common::tpc_fast_transformation::TPCFastTransform;

class GPUChainTracking : public GPUChain, GPUReconstructionHelpers::helperDelegateBase
{
	friend class GPUReconstruction;
public:
	virtual ~GPUChainTracking();
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
		
		const GPUTPCClusterData* clusterData[NSLICES];
		unsigned int nClusterData[NSLICES];
		const AliHLTTPCRawCluster* rawClusters[NSLICES];
		unsigned int nRawClusters[NSLICES];
		const o2::TPC::ClusterNativeAccessFullTPC* clustersNative;
		const GPUTPCSliceOutTrack* sliceOutTracks[NSLICES];
		unsigned int nSliceOutTracks[NSLICES];
		const GPUTPCSliceOutCluster* sliceOutClusters[NSLICES];
		unsigned int nSliceOutClusters[NSLICES];
		const AliHLTTPCClusterMCLabel* mcLabelsTPC;
		unsigned int nMCLabelsTPC;
		const GPUTPCMCInfo* mcInfosTPC;
		unsigned int nMCInfosTPC;
		const GPUTPCGMMergedTrack* mergedTracks;
		unsigned int nMergedTracks;
		const GPUTPCGMMergedTrackHit* mergedTrackHits;
		unsigned int nMergedTrackHits;
		const GPUTRDTrack* trdTracks;
		unsigned int nTRDTracks;
		const GPUTRDTrackletWord* trdTracklets;
		unsigned int nTRDTracklets;
		const GPUTRDTrackletLabels* trdTrackletsMC;
		unsigned int nTRDTrackletsMC;friend class GPUReconstruction;
	} mIOPtrs;

	struct InOutMemory
	{
		InOutMemory();
		~InOutMemory();
		InOutMemory(InOutMemory&&);
		InOutMemory& operator=(InOutMemory&&);
		
		std::unique_ptr<GPUTPCClusterData[]> clusterData[NSLICES];
		std::unique_ptr<AliHLTTPCRawCluster[]> rawClusters[NSLICES];
		std::unique_ptr<o2::TPC::ClusterNative[]> clustersNative[NSLICES * GPUCA_ROW_COUNT];
		std::unique_ptr<GPUTPCSliceOutTrack[]> sliceOutTracks[NSLICES];
		std::unique_ptr<GPUTPCSliceOutCluster[]> sliceOutClusters[NSLICES];
		std::unique_ptr<AliHLTTPCClusterMCLabel[]> mcLabelsTPC;
		std::unique_ptr<GPUTPCMCInfo[]> mcInfosTPC;
		std::unique_ptr<GPUTPCGMMergedTrack[]> mergedTracks;
		std::unique_ptr<GPUTPCGMMergedTrackHit[]> mergedTrackHits;
		std::unique_ptr<GPUTRDTrack[]> trdTracks;
		std::unique_ptr<GPUTRDTrackletWord[]> trdTracklets;
		std::unique_ptr<GPUTRDTrackletLabels[]> trdTrackletsMC;
	} mIOMem;
	
	//Read / Dump / Clear Data
	void ClearIOPointers();
	void AllocateIOMemory();
	using GPUChain::DumpData;
	void DumpData(const char* filename);
	using GPUChain::ReadData;
	int ReadData(const char* filename);
	virtual void DumpSettings(const char* dir = "") override;
	virtual void ReadSettings(const char* dir = "") override;
	
	//Converter functions
	void ConvertNativeToClusterData();
	
	//Getters for external usage of tracker classes
	GPUTRDTracker* GetTRDTracker() {return &workers()->trdTracker;}
	GPUTPCTracker* GetTPCSliceTrackers() {return workers()->tpcTrackers;}
	const GPUTPCTracker* GetTPCSliceTrackers() const {return workers()->tpcTrackers;}
	const GPUTPCGMMerger& GetTPCMerger() const {return workers()->tpcMerger;}
	GPUTPCGMMerger& GetTPCMerger() {return workers()->tpcMerger;}
	GPUDisplay* GetEventDisplay() {return mEventDisplay.get();}
	const GPUQA* GetQA() const {return mQA.get();}
	GPUQA* GetQA() {return mQA.get();}
	
	//Processing functions
	int RunTPCTrackingSlices();
	virtual int RunTPCTrackingMerger();
	virtual int RunTRDTracking();
	int DoTRDGPUTracking();
	
	//Getters / setters for parameters
	const TPCFastTransform* GetTPCTransform() const {return mTPCFastTransform.get();}
	const GPUTRDGeometry* GetTRDGeometry() const {return (GPUTRDGeometry*) mTRDGeometry.get();}
	const ClusterNativeAccessExt* GetClusterNativeAccessExt() const {return mClusterNativeAccess.get();}
	void SetTPCFastTransform(std::unique_ptr<TPCFastTransform> tpcFastTransform);
	void SetTRDGeometry(const o2::trd::TRDGeometryFlat& geo);
	void LoadClusterErrors();
	
	const void* mConfigDisplay = nullptr;										//Abstract pointer to Standalone Display Configuration Structure
	const void* mConfigQA = nullptr;											//Abstract pointer to Standalone QA Configuration Structure

protected:
	struct GPUTrackingFlatObjects : public GPUProcessor
	{
		GPUChainTracking* fChainTracking = nullptr;
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
	
	GPUChainTracking(GPUReconstruction* rec);
	
	int ReadEvent(int iSlice, int threadId);
	void WriteOutput(int iSlice, int threadId);
	int GlobalTracking(int iSlice, int threadId);
	
	int PrepareProfile();
	int DoProfile();

	//Pointers to tracker classes
	GPUTrackingFlatObjects mFlatObjectsShadow; //Host copy of flat objects that will be used on the GPU
	GPUTrackingFlatObjects mFlatObjectsDevice; //flat objects that will be used on the GPU
	
	//Display / QA
	std::unique_ptr<GPUDisplay> mEventDisplay;
	bool mDisplayRunning = false;
	std::unique_ptr<GPUQA> mQA;
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
	
	int HelperReadEvent(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par);
	int HelperOutput(int iSlice, int threadId, GPUReconstructionHelpers::helperParam* par);
};

#endif
