#include "AliGPUReconstructionImpl.h"
#include "AliGPUReconstructionCommon.h"

#include "AliGPUTPCClusterData.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUTPCSliceOutTrack.h"
#include "AliGPUTPCSliceOutCluster.h"
#include "AliGPUTPCGMMergedTrack.h"
#include "AliGPUTPCGMMergedTrackHit.h"
#include "AliGPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliGPUTPCMCInfo.h"
#include "AliGPUTRDTrack.h"
#include "AliGPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "ClusterNativeAccessExt.h"
#include "AliGPUTRDTrackletLabels.h"
#include "AliGPUCADisplay.h"
#include "AliGPUCAQA.h"
#include "AliGPUMemoryResource.h"
#include "AliGPUCADataTypes.h"

#define GPUCA_LOGGING_PRINTF
#include "AliCAGPULogging.h"

int AliGPUReconstructionCPU::RunTPCTrackingSlices()
{
	bool error = false;
	//int nLocalTracks = 0, nGlobalTracks = 0, nOutputTracks = 0, nLocalHits = 0, nGlobalHits = 0;

	if (mOutputControl.OutputType != AliGPUCAOutputControl::AllocateInternal && mDeviceProcessingSettings.nThreads > 1)
	{
		CAGPUError("fOutputPtr must not be used with multiple threads\n");
		return(1);
	}
	int offset = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (error) continue;
		mWorkers->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
		offset += mIOPtrs.nClusterData[iSlice];
	}
	PrepareEvent();
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel for num_threads(mDeviceProcessingSettings.nThreads)
#endif
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (mWorkers->tpcTrackers[iSlice].ReadEvent())
		{
			CAGPUError("Error initializing cluster data\n");
			error = true;
			continue;
		}
		mWorkers->tpcTrackers[iSlice].SetOutput(&mSliceOutput[iSlice]);
		mWorkers->tpcTrackers[iSlice].Reconstruct();
		mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks = mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTracks;
		mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits = mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTrackHits;
		if (!mParam.rec.GlobalTracking)
		{
			mWorkers->tpcTrackers[iSlice].ReconstructOutput();
			//nOutputTracks += (*mWorkers->tpcTrackers[iSlice].Output())->NTracks();
			//nLocalTracks += mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTracks;
			if (!mDeviceProcessingSettings.eventDisplay)
			{
				mWorkers->tpcTrackers[iSlice].SetupCommonMemory();
			}
		}
	}
	if (error) return(1);

	if (mParam.rec.GlobalTracking)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			int sliceLeft = (iSlice + (NSLICES / 2 - 1)) % (NSLICES / 2);
			int sliceRight = (iSlice + 1) % (NSLICES / 2);
			if (iSlice >= NSLICES / 2)
			{
				sliceLeft += NSLICES / 2;
				sliceRight += NSLICES / 2;
			}
			mWorkers->tpcTrackers[iSlice].PerformGlobalTracking(mWorkers->tpcTrackers[sliceLeft], mWorkers->tpcTrackers[sliceRight], mWorkers->tpcTrackers[sliceLeft].NMaxTracks(), mWorkers->tpcTrackers[sliceRight].NMaxTracks());
		}
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			mWorkers->tpcTrackers[iSlice].ReconstructOutput();
			//printf("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d\n", iSlice, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTracks, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits, mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTrackHits);
			//nLocalTracks += mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks;
			//nGlobalTracks += mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTracks;
			//nLocalHits += mWorkers->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits;
			//nGlobalHits += mWorkers->tpcTrackers[iSlice].CommonMemory()->fNTrackHits;
			//nOutputTracks += (*mWorkers->tpcTrackers[iSlice].Output())->NTracks();
			if (!mDeviceProcessingSettings.eventDisplay)
			{
				mWorkers->tpcTrackers[iSlice].SetupCommonMemory();
			}
		}
	}
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError != 0)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[mWorkers->tpcTrackers[iSlice].GPUParameters()->fGPUError];
			CAGPUError("Error during tracking: %s\n", errorMsg);
			return(1);
		}
	}
	//printf("Slice Tracks Output %d: - Tracks: %d local, %d global -  Hits: %d local, %d global\n", nOutputTracks, nLocalTracks, nGlobalTracks, nLocalHits, nGlobalHits);
	if (mDeviceProcessingSettings.debugMask & 1024)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			mWorkers->tpcTrackers[i].DumpOutput(stdout);
		}
	}
	return 0;
}

int AliGPUReconstructionCPU::RunTPCTrackingMerger()
{
	mWorkers->tpcMerger.Reconstruct();
	mIOPtrs.mergedTracks = mWorkers->tpcMerger.OutputTracks();
	mIOPtrs.nMergedTracks = mWorkers->tpcMerger.NOutputTracks();
	mIOPtrs.mergedTrackHits = mWorkers->tpcMerger.Clusters();
	mIOPtrs.nMergedTrackHits = mWorkers->tpcMerger.NOutputTrackClusters();
	return 0;
}

int AliGPUReconstructionCPU::RunTRDTracking()
{
	HighResTimer timer;
	timer.Start();
	
	if (!mWorkers->trdTracker.IsInitialized()) return 1;
	std::vector<GPUTRDTrack> tracksTPC;
	std::vector<int> tracksTPCLab;
	std::vector<int> tracksTPCId;

	for (unsigned int i = 0;i < mIOPtrs.nMergedTracks;i++)
	{
		const AliGPUTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
		if (!trk.OK()) continue;
		if (trk.Looper()) continue;
		if (mParam.rec.NWaysOuter) tracksTPC.emplace_back(trk.OuterParam());
		else tracksTPC.emplace_back(trk);
		tracksTPCId.push_back(i);
		tracksTPCLab.push_back(-1);
	}

	mWorkers->trdTracker.Reset();

	mWorkers->trdTracker.SetMaxData();
	if (GetDeviceProcessingSettings().memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		AllocateRegisteredMemory(mWorkers->trdTracker.MemoryTracks());
		AllocateRegisteredMemory(mWorkers->trdTracker.MemoryTracklets());
	}

	for (unsigned int iTracklet = 0;iTracklet < mIOPtrs.nTRDTracklets;++iTracklet)
	{
		if (mWorkers->trdTracker.LoadTracklet(mIOPtrs.trdTracklets[iTracklet], mIOPtrs.trdTrackletsMC ? mIOPtrs.trdTrackletsMC[iTracklet].fLabel : nullptr)) return 1;
	}

	for (unsigned int iTrack = 0; iTrack < tracksTPC.size(); ++iTrack)
	{
		if (mWorkers->trdTracker.LoadTrack(tracksTPC[iTrack], tracksTPCLab[iTrack])) return 1;
	}

	mWorkers->trdTracker.DoTracking();
	
	printf("TRD Tracker reconstructed %d tracks\n", mWorkers->trdTracker.NTracks());
	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		printf("TRD tracking time: %'d us\n", (int) (1000000 * timer.GetCurrentElapsedTime()));
	}
	
	return 0;
}
