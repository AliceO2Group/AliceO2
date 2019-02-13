#define GPUCA_ALIGPURECONSTRUCTIONCPU_IMPLEMENTATION
#include "AliGPUReconstructionCPU.h"
#include "AliGPUReconstructionIncludes.h"

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
#include "AliGPUMemoryResource.h"
#include "AliGPUConstantMem.h"

#include "AliGPUQA.h"
#include "AliGPUDisplay.h"

#include "../utils/linux_helpers.h"

#define GPUCA_LOGGING_PRINTF
#include "AliGPULogging.h"

AliGPUReconstruction* AliGPUReconstruction::AliGPUReconstruction_Create_CPU(const AliGPUSettingsProcessing& cfg)
{
	return new AliGPUReconstructionCPU(cfg);
}

template <class T, int I, typename... Args> int AliGPUReconstructionCPUBackend::runKernelBackend(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
{
	if (x.device == krnlDeviceType::Device) throw std::runtime_error("Cannot run device kernel on host");
	unsigned int num = y.num == 0 || y.num == -1 ? 1 : y.num;
	for (unsigned int k = 0;k < num;k++)
	{
		for (unsigned int iB = 0; iB < x.nBlocks; iB++)
		{
			typename T::AliGPUTPCSharedMemory smem;
			T::template Thread<I>(x.nBlocks, 1, iB, 0, smem, T::Worker(*mHostConstantMem)[y.start + k], args...);
		}
	}
	return 0;
}

int AliGPUReconstructionCPU::ReadEvent(int iSlice, int threadId)
{
	if (mDeviceProcessingSettings.debugLevel >= 5) {GPUInfo("Running ReadEvent for slice %d on thread %d\n", iSlice, threadId);}
	timerTPCtracking[iSlice][0].Start();
	if (workers()->tpcTrackers[iSlice].ReadEvent()) return(1);
	timerTPCtracking[iSlice][0].Stop();
	if (mDeviceProcessingSettings.debugLevel >= 5) {GPUInfo("Finished ReadEvent for slice %d on thread %d\n", iSlice, threadId);}
	return(0);
}

void AliGPUReconstructionCPU::WriteOutput(int iSlice, int threadId)
{
	if (mDeviceProcessingSettings.debugLevel >= 5) {GPUInfo("Running WriteOutput for slice %d on thread %d\n", iSlice, threadId);}
	workers()->tpcTrackers[iSlice].SetOutput(&mSliceOutput[iSlice]);
	timerTPCtracking[iSlice][9].Start();
	if (mDeviceProcessingSettings.nDeviceHelperThreads) while (mLockAtomic.test_and_set(std::memory_order_acquire));
	workers()->tpcTrackers[iSlice].WriteOutputPrepare();
	if (mDeviceProcessingSettings.nDeviceHelperThreads) mLockAtomic.clear();
	workers()->tpcTrackers[iSlice].WriteOutput();
	timerTPCtracking[iSlice][9].Stop();
	if (mDeviceProcessingSettings.debugLevel >= 5) {GPUInfo("Finished WriteOutput for slice %d on thread %d\n", iSlice, threadId);}
}

int AliGPUReconstructionCPU::GlobalTracking(int iSlice, int threadId)
{
	if (mDeviceProcessingSettings.debugLevel >= 5) {GPUInfo("GPU Tracker running Global Tracking for slice %d on thread %d\n", iSlice, threadId);}

	int sliceLeft = (iSlice + (NSLICES / 2 - 1)) % (NSLICES / 2);
	int sliceRight = (iSlice + 1) % (NSLICES / 2);
	if (iSlice >= (int) NSLICES / 2)
	{
		sliceLeft += NSLICES / 2;
		sliceRight += NSLICES / 2;
	}
	while (fSliceOutputReady < iSlice || fSliceOutputReady < sliceLeft || fSliceOutputReady < sliceRight);

	timerTPCtracking[iSlice][8].Start();
	workers()->tpcTrackers[iSlice].PerformGlobalTracking(workers()->tpcTrackers[sliceLeft], workers()->tpcTrackers[sliceRight], GPUCA_GPUCA_MAX_TRACKS, GPUCA_GPUCA_MAX_TRACKS);
	timerTPCtracking[iSlice][8].Stop();

	fSliceLeftGlobalReady[sliceLeft] = 1;
	fSliceRightGlobalReady[sliceRight] = 1;
	if (mDeviceProcessingSettings.debugLevel >= 5) {GPUInfo("GPU Tracker finished Global Tracking for slice %d on thread %d\n", iSlice, threadId);}
	return(0);
}

int AliGPUReconstructionCPU::RunTPCTrackingSlices()
{
	if (!(mRecoStepsGPU & RecoStep::TPCSliceTracking) && mOutputControl.OutputType != AliGPUOutputControl::AllocateInternal && mDeviceProcessingSettings.nThreads > 1)
	{
		GPUError("fOutputPtr must not be used with multiple threads\n");
		return(1);
	}

	if (mGPUStuck)
	{
		GPUWarning("This GPU is stuck, processing of tracking for this event is skipped!");
		return(1);
	}
	
	if (mThreadId != GetThread())
	{
		if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("Thread changed, migrating context, Previous Thread: %d, New Thread: %d", mThreadId, GetThread());
		mThreadId = GetThread();
	}

	ActivateThreadContext();
	SetThreadCounts(RecoStep::TPCSliceTracking);
	
	int retVal = RunTPCTrackingSlices_internal();
	if (retVal) SynchronizeGPU();
	if (retVal >= 2)
	{
		ResetHelperThreads(retVal >= 3);
	}
	ReleaseThreadContext();
	return(retVal != 0);
}

int AliGPUReconstructionCPU::RunTPCTrackingSlices_internal()
{
	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("Running TPC Slice Tracker");
	bool doGPU = mRecoStepsGPU & RecoStep::TPCSliceTracking;

	int offset = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		workers()->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
		offset += mIOPtrs.nClusterData[iSlice];
	}
	if (doGPU)
	{
		memcpy((void*) mWorkersShadow, (const void*) workers(), sizeof(*workers()));
		for (unsigned int i = 0;i < mProcessors.size();i++)
		{
			if (mProcessors[i].proc->mDeviceProcessor) mProcessors[i].proc->mDeviceProcessor->InitGPUProcessor(this, AliGPUProcessor::PROCESSOR_TYPE_DEVICE);
		}
	}
	try
	{
		PrepareEvent();
	}
	catch (const std::bad_alloc& e)
	{
		printf("Memory Allocation Error\n");
		return(2);
	}

	bool streamInit[GPUCA_GPUCA_MAX_STREAMS] = {false};
	if (doGPU)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUMem = (char*) mDeviceMemoryBase;
			//Initialize Startup Constants
			*workers()->tpcTrackers[iSlice].NTracklets() = 0;
			*workers()->tpcTrackers[iSlice].NTracks() = 0;
			*workers()->tpcTrackers[iSlice].NTrackHits() = 0;
			workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError = 0;
			workers()->tpcTrackers[iSlice].GPUParameters()->fNextTracklet = ((fConstructorBlockCount + NSLICES - 1 - iSlice) / NSLICES) * fConstructorThreadCount;
			mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUFixedBlockCount = NSLICES > fConstructorBlockCount ? (iSlice < fConstructorBlockCount) : fConstructorBlockCount * (iSlice + 1) / NSLICES - fConstructorBlockCount * (iSlice) / NSLICES;
			mWorkersShadow->tpcTrackers[iSlice].GPUParametersConst()->fGPUiSlice = iSlice;
			mWorkersShadow->tpcTrackers[iSlice].SetGPUTextureBase(mDeviceMemoryBase);
		}

		RunHelperThreads(0);
		if (PrepareTextures()) return(2);

		//Copy Tracker Object to GPU Memory
		if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Copying Tracker objects to GPU");
		if (PrepareProfile()) return 2;

		WriteToConstantMemory((char*) &mDeviceConstantMem->param - (char*) mDeviceConstantMem, &param(), sizeof(AliGPUParam), mNStreams - 1);
		WriteToConstantMemory((char*) mDeviceConstantMem->tpcTrackers - (char*) mDeviceConstantMem, mWorkersShadow->tpcTrackers, sizeof(AliGPUTPCTracker) * NSLICES, mNStreams - 1, &mEvents.init);

		for (int i = 0;i < mNStreams - 1;i++)
		{
			streamInit[i] = false;
		}
		streamInit[mNStreams - 1] = true;
	}
	if (GPUDebug("Initialization (1)", 0)) return(2);

	int streamMap[NSLICES];

	bool error = false;
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel for num_threads(doGPU ? 1 : mDeviceProcessingSettings.nThreads)
#endif
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		AliGPUTPCTracker& trk = workers()->tpcTrackers[iSlice];
		AliGPUTPCTracker& trkShadow = doGPU ? mWorkersShadow->tpcTrackers[iSlice] : trk;;
		
		if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Creating Slice Data (Slice %d)", iSlice);
		if (!doGPU || iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) == 0)
		{
			if (ReadEvent(iSlice, 0))
			{
				GPUError("Error reading event");
				error = 1;
				continue;
			}
		}
		else
		{
			if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Waiting for helper thread %d", iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1);
			while(HelperDone(iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1) < (int) iSlice);
			if (HelperError(iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) - 1))
			{
				error = 1;
				continue;
			}
		}
		if (!doGPU && trk.CheckEmptySlice()) continue;

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			if (!mDeviceProcessingSettings.comparableDebutOutput) mDebugFile << std::endl << std::endl << "Reconstruction: Slice " << iSlice << "/" << NSLICES << std::endl;
			if (mDeviceProcessingSettings.debugMask & 1) trk.DumpSliceData(mDebugFile);
		}
		
		int useStream = (iSlice % mNStreams);
		//Initialize temporary memory where needed
		if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Copying Slice Data to GPU and initializing temporary memory");
		runKernel<AliGPUMemClean16>({fBlockCount, fThreadCount, useStream}, &timerTPCtracking[iSlice][5], krnlRunRangeNone, {}, trkShadow.Data().HitWeights(), trkShadow.Data().NumberOfHitsPlusAlign() * sizeof(*trkShadow.Data().HitWeights()));

		//Copy Data to GPU Global Memory
		timerTPCtracking[iSlice][0].Start();
		TransferMemoryResourceLinkToGPU(trk.Data().MemoryResInput(), useStream);
		TransferMemoryResourceLinkToGPU(trk.Data().MemoryResRows(), useStream);
		TransferMemoryResourceLinkToGPU(trk.MemoryResCommon(), useStream);
		if (GPUDebug("Initialization (3)", useStream)) throw std::runtime_error("memcpy failure");
		timerTPCtracking[iSlice][0].Stop();

		runKernel<AliGPUTPCNeighboursFinder>({GPUCA_ROW_COUNT, fFinderThreadCount, useStream}, &timerTPCtracking[iSlice][1], {iSlice}, {nullptr, streamInit[useStream] ? nullptr : &mEvents.init});
		streamInit[useStream] = true;

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			TransferMemoryResourcesToHost(&trk.Data(), -1, true);
			memcpy(trk.LinkTmpMemory(), Res(trk.Data().MemoryResScratch()).Ptr(), Res(trk.Data().MemoryResScratch()).Size());
			if (mDeviceProcessingSettings.debugMask & 2) trk.DumpLinks(mDebugFile);
		}

		runKernel<AliGPUTPCNeighboursCleaner>({GPUCA_ROW_COUNT - 2, fThreadCount, useStream}, &timerTPCtracking[iSlice][2], {iSlice});

		if (mDeviceProcessingSettings.debugLevel >= 4)
		{
			TransferMemoryResourcesToHost(&trk.Data(), -1, true);
			if (mDeviceProcessingSettings.debugMask & 4) trk.DumpLinks(mDebugFile);
		}

		runKernel<AliGPUTPCStartHitsFinder>({GPUCA_ROW_COUNT - 6, fThreadCount, useStream}, &timerTPCtracking[iSlice][3], {iSlice});

		if (doGPU) runKernel<AliGPUTPCStartHitsSorter>({fBlockCount, fThreadCount, useStream}, &timerTPCtracking[iSlice][4], {iSlice});

		if (mDeviceProcessingSettings.debugLevel >= 2)
		{
			TransferMemoryResourceLinkToHost(trk.MemoryResCommon(), -1);
			if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Obtaining Number of Start Hits from GPU: %d (Slice %d)", *trk.NTracklets(), iSlice);
		}
		
		if (mDeviceProcessingSettings.debugLevel >= 4 && *trk.NTracklets())
		{
			TransferMemoryResourcesToHost(&trk, -1, true);
			if (mDeviceProcessingSettings.debugMask & 32) trk.DumpStartHits(mDebugFile);
		}

		if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
		{
			trk.UpdateMaxData();
			AllocateRegisteredMemory(trk.MemoryResTracklets());
			AllocateRegisteredMemory(trk.MemoryResTracks());
			AllocateRegisteredMemory(trk.MemoryResTrackHits());
		}

		if (!doGPU || mDeviceProcessingSettings.trackletConstructorInPipeline)
		{
			runKernel<AliGPUTPCTrackletConstructor>({fConstructorBlockCount, fConstructorThreadCount, useStream}, &timerTPCtracking[iSlice][6], {iSlice});
			if (mDeviceProcessingSettings.debugLevel >= 3) printf("Slice %d, Number of tracklets: %d\n", iSlice, *trk.NTracklets());
			if (mDeviceProcessingSettings.debugMask & 128) trk.DumpTrackletHits(mDebugFile);
			if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) trk.DumpHitWeights(mDebugFile);
		}

		if (!doGPU || mDeviceProcessingSettings.trackletSelectorInPipeline)
		{
			runKernel<AliGPUTPCTrackletSelector>({fSelectorBlockCount, fSelectorThreadCount, useStream}, &timerTPCtracking[iSlice][7], {iSlice});
			TransferMemoryResourceLinkToHost(trk.MemoryResCommon(), useStream, &mEvents.selector[iSlice]);
			streamMap[iSlice] = useStream;
			if (mDeviceProcessingSettings.debugLevel >= 3) printf("Slice %d, Number of tracks: %d\n", iSlice, *trk.NTracks());
			if (mDeviceProcessingSettings.debugMask & 512) trk.DumpTrackHits(mDebugFile);
		}

		if (!doGPU)
		{
			trk.CommonMemory()->fNLocalTracks = trk.CommonMemory()->fNTracks;
			trk.CommonMemory()->fNLocalTrackHits = trk.CommonMemory()->fNTrackHits;
			if (!param().rec.GlobalTracking)
			{
				WriteOutput(iSlice, 0);
			}
		}
	}
	if (error) return(3);

	if (doGPU)
	{
		ReleaseEvent(&mEvents.init);
		WaitForHelperThreads();

		if (!mDeviceProcessingSettings.trackletSelectorInPipeline)
		{
			if (mDeviceProcessingSettings.trackletConstructorInPipeline)
			{
				SynchronizeGPU();
			}
			else
			{
				for (int i = 0;i < mNStreams;i++) RecordMarker(&mEvents.stream[i], i);
				runKernel<AliGPUTPCTrackletConstructor, 1>({fConstructorBlockCount, fConstructorThreadCount, 0}, &timerTPCtracking[0][6], krnlRunRangeNone, {&mEvents.constructor, mEvents.stream, mNStreams});
				for (int i = 0;i < mNStreams;i++) ReleaseEvent(&mEvents.stream[i]);
				SynchronizeEvents(&mEvents.constructor);
				ReleaseEvent(&mEvents.constructor);
			}

			if (mDeviceProcessingSettings.debugLevel >= 4)
			{
				for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
				{
					TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice], -1, true);
					GPUInfo("Obtained %d tracklets", *workers()->tpcTrackers[iSlice].NTracklets());
					if (mDeviceProcessingSettings.debugMask & 128) workers()->tpcTrackers[iSlice].DumpTrackletHits(mDebugFile);
				}
			}

			unsigned int runSlices = 0;
			int useStream = 0;
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice += runSlices)
			{
				if (runSlices < GPUCA_GPUCA_TRACKLET_SELECTOR_SLICE_COUNT) runSlices++;
				runSlices = CAMath::Min(runSlices, NSLICES - iSlice);
				if (fSelectorBlockCount < runSlices) runSlices = fSelectorBlockCount;

				if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Running HLT Tracklet selector (Stream %d, Slice %d to %d)", useStream, iSlice, iSlice + runSlices);
				runKernel<AliGPUTPCTrackletSelector>({fSelectorBlockCount, fSelectorThreadCount, useStream}, &timerTPCtracking[iSlice][7], {iSlice, (int) runSlices});
				for (unsigned int k = iSlice;k < iSlice + runSlices;k++)
				{
					TransferMemoryResourceLinkToHost(workers()->tpcTrackers[k].MemoryResCommon(), useStream, &mEvents.selector[k]);
					streamMap[k] = useStream;
				}
				useStream++;
				if (useStream >= mNStreams) useStream = 0;
			}
		}

		fSliceOutputReady = 0;

		if (param().rec.GlobalTracking)
		{
			memset((void*) fSliceLeftGlobalReady, 0, sizeof(fSliceLeftGlobalReady));
			memset((void*) fSliceRightGlobalReady, 0, sizeof(fSliceRightGlobalReady));
			fGlobalTrackingDone.fill(0);
			fWriteOutputDone.fill(0);
		}
		RunHelperThreads(1);

		std::array<bool, NSLICES> transferRunning;
		transferRunning.fill(true);
		unsigned int tmpSlice = 0;
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Transfering Tracks from GPU to Host");

			if (tmpSlice == iSlice) SynchronizeEvents(&mEvents.selector[iSlice]);
			while (tmpSlice < NSLICES && (tmpSlice == iSlice || IsEventDone(&mEvents.selector[tmpSlice])))
			{
				ReleaseEvent(&mEvents.selector[tmpSlice]);
				if (*workers()->tpcTrackers[tmpSlice].NTracks() > 0)
				{
					TransferMemoryResourceLinkToHost(workers()->tpcTrackers[tmpSlice].MemoryResTracks(), streamMap[tmpSlice]);
					TransferMemoryResourceLinkToHost(workers()->tpcTrackers[tmpSlice].MemoryResTrackHits(), streamMap[tmpSlice], &mEvents.selector[tmpSlice]);
				}
				else
				{
					transferRunning[tmpSlice] = false;
				}
				tmpSlice++;
			}

			if (mDeviceProcessingSettings.keepAllMemory)
			{
				TransferMemoryResourcesToHost(&workers()->tpcTrackers[iSlice], -1, true);
				if (mDeviceProcessingSettings.debugMask & 256 && !mDeviceProcessingSettings.comparableDebutOutput) workers()->tpcTrackers[iSlice].DumpHitWeights(mDebugFile);
				if (mDeviceProcessingSettings.debugMask & 512) workers()->tpcTrackers[iSlice].DumpTrackHits(mDebugFile);
			}

			if (transferRunning[iSlice]) SynchronizeEvents(&mEvents.selector[iSlice]);
			if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Tracks Transfered: %d / %d", *workers()->tpcTrackers[iSlice].NTracks(), *workers()->tpcTrackers[iSlice].NTrackHits());
			
			workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks = workers()->tpcTrackers[iSlice].CommonMemory()->fNTracks;
			workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits = workers()->tpcTrackers[iSlice].CommonMemory()->fNTrackHits;

			if (mDeviceProcessingSettings.debugLevel >= 3) GPUInfo("Data ready for slice %d, helper thread %d", iSlice, iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1));
			fSliceOutputReady = iSlice;

			if (param().rec.GlobalTracking)
			{
				if (iSlice % (NSLICES / 2) == 2)
				{
					int tmpId = iSlice % (NSLICES / 2) - 1;
					if (iSlice >= NSLICES / 2) tmpId += NSLICES / 2;
					GlobalTracking(tmpId, 0);
					fGlobalTrackingDone[tmpId] = 1;
				}
				for (unsigned int tmpSlice3a = 0;tmpSlice3a < iSlice;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
				{
					unsigned int tmpSlice3 = tmpSlice3a + 1;
					if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
					if (tmpSlice3 >= iSlice) break;

					unsigned int sliceLeft = (tmpSlice3 + (NSLICES / 2 - 1)) % (NSLICES / 2);
					unsigned int sliceRight = (tmpSlice3 + 1) % (NSLICES / 2);
					if (tmpSlice3 >= (int) NSLICES / 2)
					{
						sliceLeft += NSLICES / 2;
						sliceRight += NSLICES / 2;
					}

					if (tmpSlice3 % (NSLICES / 2) != 1 && fGlobalTrackingDone[tmpSlice3] == 0 && sliceLeft < iSlice && sliceRight < iSlice)
					{
						GlobalTracking(tmpSlice3, 0);
						fGlobalTrackingDone[tmpSlice3] = 1;
					}

					if (fWriteOutputDone[tmpSlice3] == 0 && fSliceLeftGlobalReady[tmpSlice3] && fSliceRightGlobalReady[tmpSlice3])
					{
						WriteOutput(tmpSlice3, 0);
						fWriteOutputDone[tmpSlice3] = 1;
					}
				}
			}
			else
			{
				if (iSlice % (mDeviceProcessingSettings.nDeviceHelperThreads + 1) == 0)
				{
					WriteOutput(iSlice, 0);
				}
			}
		}
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++) if (transferRunning[iSlice]) ReleaseEvent(&mEvents.selector[iSlice]);

		if (param().rec.GlobalTracking)
		{
			for (unsigned int tmpSlice3a = 0;tmpSlice3a < NSLICES;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
			{
				unsigned int tmpSlice3 = (tmpSlice3a + 1);
				if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
				if (fGlobalTrackingDone[tmpSlice3] == 0) GlobalTracking(tmpSlice3, 0);
			}
			for (unsigned int tmpSlice3a = 0;tmpSlice3a < NSLICES;tmpSlice3a += mDeviceProcessingSettings.nDeviceHelperThreads + 1)
			{
				unsigned int tmpSlice3 = (tmpSlice3a + 1);
				if (tmpSlice3 % (NSLICES / 2) < 1) tmpSlice3 -= (NSLICES / 2);
				if (fWriteOutputDone[tmpSlice3] == 0)
				{
					while (fSliceLeftGlobalReady[tmpSlice3] == 0 || fSliceRightGlobalReady[tmpSlice3] == 0);
					WriteOutput(tmpSlice3, 0);
				}
			}
		}
		WaitForHelperThreads();
	}
	else
	{
		fSliceOutputReady = NSLICES;
		if (param().rec.GlobalTracking)
		{
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
			{
				GlobalTracking(iSlice, 0);
			}
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
			{
				WriteOutput(iSlice, 0);
			}
		}
	}

	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError != 0)
		{
			const char* errorMsgs[] = GPUCA_GPUCA_ERROR_STRINGS;
			const char* errorMsg = (unsigned) workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError];
			GPUError("GPU Tracker returned Error Code %d (%s) in slice %d (Clusters %d)", workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError, errorMsg, iSlice, workers()->tpcTrackers[iSlice].Data().NumberOfHits());
			return(1);
		}
	}

	if (param().rec.GlobalTracking)
	{
		if (mDeviceProcessingSettings.debugLevel >= 3)
		{
			for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
			{
				GPUInfo("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d", iSlice, workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks, workers()->tpcTrackers[iSlice].CommonMemory()->fNTracks, workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits, workers()->tpcTrackers[iSlice].CommonMemory()->fNTrackHits);
			}
		}
	}

	if (mDeviceProcessingSettings.debugMask & 1024)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			workers()->tpcTrackers[i].DumpOutput(stdout);
		}
	}
	if (DoProfile()) return(1);
	if (mDeviceProcessingSettings.debugLevel >= 2) GPUInfo("TPC Slice Tracker finished");
	return 0;
}

int AliGPUReconstructionCPU::RunTPCTrackingMerger()
{
	if (workers()->tpcMerger.CheckSlices()) return 1;
	HighResTimer timer;
	static double times[8] = {};
	static int nCount = 0;
	if (mDeviceProcessingSettings.resetTimers || !GPUCA_TIMING_SUM)
	{
		for (unsigned int k = 0; k < sizeof(times) / sizeof(times[0]); k++) times[k] = 0;
		nCount = 0;
	}

	SetupGPUProcessor(&workers()->tpcMerger, true);
	
	timer.ResetStart();
	workers()->tpcMerger.UnpackSlices();
	times[0] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.MergeWithingSlices();
	
	times[1] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.MergeSlices();
	
	times[2] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.MergeCEInit();
	
	times[3] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.CollectMergedTracks();
	
	times[4] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.MergeCE();
	
	times[3] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.PrepareClustersForFit();
	
	times[5] += timer.GetCurrentElapsedTime(true);
	RefitMergedTracks(mDeviceProcessingSettings.resetTimers);
	
	times[6] += timer.GetCurrentElapsedTime(true);
	workers()->tpcMerger.Finalize();
	
	times[7] += timer.GetCurrentElapsedTime(true);
	nCount++;
	if (mDeviceProcessingSettings.debugLevel > 0)
	{
		printf("Merge Time:\tUnpack Slices:\t%'7d us\n", (int) (times[0] * 1000000 / nCount));
		printf("\t\tMerge Within:\t%'7d us\n", (int) (times[1] * 1000000 / nCount));
		printf("\t\tMerge Slices:\t%'7d us\n", (int) (times[2] * 1000000 / nCount));
		printf("\t\tMerge CE:\t%'7d us\n", (int) (times[3] * 1000000 / nCount));
		printf("\t\tCollect:\t%'7d us\n", (int) (times[4] * 1000000 / nCount));
		printf("\t\tClusters:\t%'7d us\n", (int) (times[5] * 1000000 / nCount));
		printf("\t\tRefit:\t\t%'7d us\n", (int) (times[6] * 1000000 / nCount));
		printf("\t\tFinalize:\t%'7d us\n", (int) (times[7] * 1000000 / nCount));
	}
	
	mIOPtrs.mergedTracks = workers()->tpcMerger.OutputTracks();
	mIOPtrs.nMergedTracks = workers()->tpcMerger.NOutputTracks();
	mIOPtrs.mergedTrackHits = workers()->tpcMerger.Clusters();
	mIOPtrs.nMergedTrackHits = workers()->tpcMerger.NOutputTrackClusters();
	return 0;
}

int AliGPUReconstructionCPU::RunTRDTracking()
{
	if (!workers()->trdTracker.IsInitialized()) return 1;
	std::vector<GPUTRDTrack> tracksTPC;
	std::vector<int> tracksTPCLab;

	for (unsigned int i = 0;i < mIOPtrs.nMergedTracks;i++)
	{
		const AliGPUTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
		if (!trk.OK()) continue;
		if (trk.Looper()) continue;
		if (param().rec.NWaysOuter) tracksTPC.emplace_back(trk.OuterParam());
		else tracksTPC.emplace_back(trk);
		tracksTPC.back().SetTPCtrackId(i);
		tracksTPCLab.push_back(-1);
	}

	workers()->trdTracker.Reset();

	workers()->trdTracker.SetMaxData();
	if (GetDeviceProcessingSettings().memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
	{
		AllocateRegisteredMemory(workers()->trdTracker.MemoryTracks());
		AllocateRegisteredMemory(workers()->trdTracker.MemoryTracklets());
	}

	for (unsigned int iTracklet = 0;iTracklet < mIOPtrs.nTRDTracklets;++iTracklet)
	{
		if (workers()->trdTracker.LoadTracklet(mIOPtrs.trdTracklets[iTracklet], mIOPtrs.trdTrackletsMC ? mIOPtrs.trdTrackletsMC[iTracklet].fLabel : nullptr)) return 1;
	}

	for (unsigned int iTrack = 0; iTrack < tracksTPC.size(); ++iTrack)
	{
		if (workers()->trdTracker.LoadTrack(tracksTPC[iTrack], tracksTPCLab[iTrack])) return 1;
	}

	workers()->trdTracker.DoTracking();
	
	return 0;
}

int AliGPUReconstructionCPU::RefitMergedTracks(bool resetTimers)
{
	AliGPUReconstructionCPU::runKernel<AliGPUTPCGMMergerTrackFit>({1, 1, 0, krnlDeviceType::CPU}, nullptr, krnlRunRangeNone);
	return 0;
}

int AliGPUReconstructionCPU::RunStandalone()
{
	mStatNEvents++;
	
	const bool needQA = AliGPUQA::QAAvailable() && (mDeviceProcessingSettings.runQA || (mDeviceProcessingSettings.eventDisplay && mIOPtrs.nMCInfosTPC));
	if (needQA && mQAInitialized == false)
	{
		if (mQA->InitQA()) return 1;
		mQAInitialized = true;
	}
	
	static HighResTimer timerTracking, timerMerger, timerQA;
	static int nCount = 0;
	if (mDeviceProcessingSettings.resetTimers)
	{
		timerTracking.Reset();
		timerMerger.Reset();
		timerQA.Reset();
		nCount = 0;
	}

	timerTracking.Start();
	if (RunTPCTrackingSlices()) return 1;
	timerTracking.Stop();

	timerMerger.Start();
	for (unsigned int i = 0; i < NSLICES; i++)
	{
		//printf("slice %d clusters %d tracks %d\n", i, fClusterData[i].NumberOfClusters(), mSliceOutput[i]->NTracks());
		workers()->tpcMerger.SetSliceData(i, mSliceOutput[i]);
	}
	if (RunTPCTrackingMerger()) return 1;
	timerMerger.Stop();

	if (needQA)
	{
		timerQA.Start();
		mQA->RunQA(!mDeviceProcessingSettings.runQA);
		timerQA.Stop();
	}

	nCount++;
	if (mDeviceProcessingSettings.debugLevel >= 0)
	{
		char nAverageInfo[16] = "";
		if (nCount > 1) sprintf(nAverageInfo, " (%d)", nCount);
		printf("Tracking Time: %'d us%s\n", (int) (1000000 * timerTracking.GetElapsedTime() / nCount), nAverageInfo);
		printf("Merging and Refit Time: %'d us\n", (int) (1000000 * timerMerger.GetElapsedTime() / nCount));
		if (mDeviceProcessingSettings.runQA) printf("QA Time: %'d us\n", (int) (1000000 * timerQA.GetElapsedTime() / nCount));
	}
	
	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
			printf("Memory Allocation: Host %'lld / %'lld, Device %'lld / %'lld, %d chunks\n",
			(long long int) ((char*) mHostMemoryPool - (char*) mHostMemoryBase), (long long int) mHostMemorySize, (long long int) ((char*) mDeviceMemoryPool - (char*) mDeviceMemoryBase), (long long int) mDeviceMemorySize, (int) mMemoryResources.size());
		
		const char *tmpNames[10] = {"Initialisation", "Neighbours Finder", "Neighbours Cleaner", "Starts Hits Finder", "Start Hits Sorter", "Weight Cleaner", "Tracklet Constructor", "Tracklet Selector", "Global Tracking", "Write Output"};

		for (int i = 0; i < 10; i++)
		{
			double time = 0;
			for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++)
			{
				time += timerTPCtracking[iSlice][i].GetElapsedTime();
				timerTPCtracking[iSlice][i].Reset();
			}
			time /= NSLICES;
			if (!IsGPU()) time /= mDeviceProcessingSettings.nThreads;

			printf("Execution Time: Task: %20s Time: %'7d us\n", tmpNames[i], (int) (time * 1000000 / nCount));
		}
		printf("Execution Time: Task: %20s Time: %'7d us\n", "Merger", (int) (timerMerger.GetElapsedTime() * 1000000. / nCount));
		if (!GPUCA_TIMING_SUM)
		{
			timerTracking.Reset();
			timerMerger.Reset();
			timerQA.Reset();
			nCount = 0;
		}
	}
	
	if (mRecoSteps.isSet(RecoStep::TRDTracking) && mIOPtrs.nTRDTracklets)
	{
		HighResTimer timer;
		timer.Start();
		if (RunTRDTracking()) return 1;
		if (mDeviceProcessingSettings.debugLevel >= 1)
		{
			printf("TRD tracking time: %'d us\n", (int) (1000000 * timer.GetCurrentElapsedTime()));
		}
	}

	if (mDeviceProcessingSettings.eventDisplay)
	{
		if (!mDisplayRunning)
		{
			if (mEventDisplay->StartDisplay()) return(1);
			mDisplayRunning = true;
		}
		else
		{
			mEventDisplay->ShowNextEvent();
		}

		if (mDeviceProcessingSettings.eventDisplay->EnableSendKey())
		{
			while (kbhit()) getch();
			printf("Press key for next event!\n");
		}

		int iKey;
		do
		{
			Sleep(10);
			if (mDeviceProcessingSettings.eventDisplay->EnableSendKey())
			{
				iKey = kbhit() ? getch() : 0;
				if (iKey == 'q') mDeviceProcessingSettings.eventDisplay->displayControl = 2;
				else if (iKey == 'n') break;
				else if (iKey)
				{
					while (mDeviceProcessingSettings.eventDisplay->sendKey != 0)
					{
						Sleep(1);
					}
					mDeviceProcessingSettings.eventDisplay->sendKey = iKey;
				}
			}
		} while (mDeviceProcessingSettings.eventDisplay->displayControl == 0);
		if (mDeviceProcessingSettings.eventDisplay->displayControl == 2)
		{
			mDisplayRunning = false;
			mDeviceProcessingSettings.eventDisplay->DisplayExit();
			mDeviceProcessingSettings.eventDisplay = nullptr;
			return (2);
		}
		mDeviceProcessingSettings.eventDisplay->displayControl = 0;
		printf("Loading next event\n");

		mEventDisplay->WaitForNextEvent();
	}
	return 0;
}

void AliGPUReconstructionCPU::TransferMemoryInternal(AliGPUMemoryResource* res, int stream, deviceEvent* ev, deviceEvent* evList, int nEvents, bool toGPU, void* src, void* dst) {}
void AliGPUReconstructionCPU::WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream, deviceEvent* ev) {}
int AliGPUReconstructionCPU::GPUDebug(const char* state, int stream) {return 0;}
void AliGPUReconstructionCPU::TransferMemoryResourcesHelper(AliGPUProcessor* proc, int stream, bool all, bool toGPU)
{
	int inc = toGPU ? AliGPUMemoryResource::MEMORY_INPUT : AliGPUMemoryResource::MEMORY_OUTPUT;
	int exc = toGPU ? AliGPUMemoryResource::MEMORY_OUTPUT : AliGPUMemoryResource::MEMORY_INPUT;
	for (unsigned int i = 0;i < mMemoryResources.size();i++)
	{
		AliGPUMemoryResource& res = mMemoryResources[i];
		if (res.mPtr == nullptr) continue;
		if (proc && res.mProcessor != proc) continue;
		if (!(res.mType & AliGPUMemoryResource::MEMORY_GPU) || (res.mType & AliGPUMemoryResource::MEMORY_CUSTOM_TRANSFER)) continue;
		if (!mDeviceProcessingSettings.keepAllMemory && !(all && !(res.mType & exc)) && !(res.mType & inc)) continue;
		if (toGPU) TransferMemoryResourceToGPU(&mMemoryResources[i], stream);
		else TransferMemoryResourceToHost(&mMemoryResources[i], stream);
	}
}

int AliGPUReconstructionCPU::GetThread()
{
	//Get Thread ID
#ifdef _WIN32
	return((int) (size_t) GetCurrentThread());
#else
	return((int) syscall (SYS_gettid));
#endif
}

int AliGPUReconstructionCPU::InitDevice()
{
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
	{
		mHostMemoryPermanent = mHostMemoryBase = operator new(GPUCA_HOST_MEMORY_SIZE);
		mHostMemorySize = GPUCA_HOST_MEMORY_SIZE;
		ClearAllocatedMemory();
	}
	SetThreadCounts();
	mThreadId = GetThread();
	return 0;
}

int AliGPUReconstructionCPU::ExitDevice()
{
	if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
	{
		operator delete(mHostMemoryBase);
		mHostMemoryPool = mHostMemoryBase = mHostMemoryPermanent = nullptr;
		mHostMemorySize = 0;
	}
	return 0;
}

void AliGPUReconstructionCPU::SetThreadCounts()
{
	fThreadCount = fBlockCount = fConstructorBlockCount = fSelectorBlockCount = fConstructorThreadCount = fSelectorThreadCount = fFinderThreadCount = fTRDThreadCount = 1;
}

void AliGPUReconstructionCPU::SetThreadCounts(RecoStep step)
{
	if (IsGPU() && mRecoSteps != mRecoStepsGPU)
	{
		if (!(mRecoStepsGPU & step)) AliGPUReconstructionCPU::SetThreadCounts();
		else SetThreadCounts();
	}
}
