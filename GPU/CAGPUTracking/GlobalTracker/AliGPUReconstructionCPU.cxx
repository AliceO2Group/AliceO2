#define GPUCA_ALIGPURECONSTRUCTIONCPU_IMPLEMENTATION
#include "AliGPUReconstructionCPU.h"
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
#include "AliGPUMemoryResource.h"
#include "AliGPUCADataTypes.h"

#include "AliGPUCAQA.h"
#include "AliGPUCADisplay.h"

#include "../cmodules/linux_helpers.h"

#define GPUCA_LOGGING_PRINTF
#include "AliCAGPULogging.h"

AliGPUReconstruction* AliGPUReconstruction::AliGPUReconstruction_Create_CPU(const AliGPUCASettingsProcessing& cfg)
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

int AliGPUReconstructionCPU::RunTPCTrackingSlices()
{
	//int nLocalTracks = 0, nGlobalTracks = 0, nOutputTracks = 0, nLocalHits = 0, nGlobalHits = 0;

	if (mOutputControl.OutputType != AliGPUCAOutputControl::AllocateInternal && mDeviceProcessingSettings.nThreads > 1)
	{
		CAGPUError("fOutputPtr must not be used with multiple threads\n");
		return(1);
	}
	int offset = 0;
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		workers()->tpcTrackers[iSlice].Data().SetClusterData(mIOPtrs.clusterData[iSlice], mIOPtrs.nClusterData[iSlice], offset);
		offset += mIOPtrs.nClusterData[iSlice];
	}
	PrepareEvent();

	bool error = false;
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel for num_threads(mDeviceProcessingSettings.nThreads)
#endif
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		AliGPUTPCTracker& trk = workers()->tpcTrackers[iSlice];
		timerTPCtracking[iSlice][0].Start();
		if (trk.ReadEvent())
		{
			CAGPUError("Error initializing cluster data\n");
			error = true;
			continue;
		}
		timerTPCtracking[iSlice][0].Stop();
		trk.SetOutput(&mSliceOutput[iSlice]);
		if (trk.CheckEmptySlice()) continue;

		if (mDeviceProcessingSettings.debugLevel >= 6)
		{
			if (!mDeviceProcessingSettings.comparableDebutOutput)
			{
				mDebugFile << std::endl << std::endl << "Slice: " << iSlice << std::endl;
				mDebugFile << "Slice Data:" << std::endl;
			}
			trk.DumpSliceData(mDebugFile);
		}

		runKernel<AliGPUTPCNeighboursFinder>({GPUCA_ROW_COUNT, 1, 0}, &timerTPCtracking[iSlice][1], {iSlice});

		if (mDeviceProcessingSettings.keepAllMemory)
		{
			memcpy(trk.LinkTmpMemory(), Res(trk.Data().MemoryResScratch()).Ptr(), Res(trk.Data().MemoryResScratch()).Size());
		}

		if (mDeviceProcessingSettings.debugLevel >= 6) trk.DumpLinks(mDebugFile);

		runKernel<AliGPUTPCNeighboursCleaner>({GPUCA_ROW_COUNT - 2, 1, 0}, &timerTPCtracking[iSlice][2], {iSlice});

		if (mDeviceProcessingSettings.debugLevel >= 6) trk.DumpLinks(mDebugFile);

		runKernel<AliGPUTPCStartHitsFinder>({GPUCA_ROW_COUNT - 6, 1, 0}, &timerTPCtracking[iSlice][3], {iSlice}); //Why not -6?

		if (mDeviceProcessingSettings.debugLevel >= 6) trk.DumpStartHits(mDebugFile);

		runKernel<AliGPUMemClean16>({1, 1, 0}, &timerTPCtracking[iSlice][5], krnlRunRangeNone, {}, trk.Data().HitWeights(), trk.Data().NumberOfHitsPlusAlign() * sizeof(*trk.Data().HitWeights()));

		if (mDeviceProcessingSettings.memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL)
		{
			trk.UpdateMaxData();
			AllocateRegisteredMemory(trk.MemoryResTracklets());
			AllocateRegisteredMemory(trk.MemoryResTracks());
			AllocateRegisteredMemory(trk.MemoryResTrackHits());
		}

		runKernel<AliGPUTPCTrackletConstructor>({1, 1, 0}, &timerTPCtracking[iSlice][6], {iSlice});
		if (mDeviceProcessingSettings.debugLevel >= 3) printf("Slice %d, Number of tracklets: %d\n", iSlice, *trk.NTracklets());

		if (mDeviceProcessingSettings.debugLevel >= 6) trk.DumpTrackletHits(mDebugFile);
		if (mDeviceProcessingSettings.debugLevel >= 6 && !mDeviceProcessingSettings.comparableDebutOutput) trk.DumpHitWeights(mDebugFile);

		runKernel<AliGPUTPCTrackletSelector>({1, 1, 0}, &timerTPCtracking[iSlice][7], {iSlice});
		if (mDeviceProcessingSettings.debugLevel >= 3) printf("Slice %d, Number of tracks: %d\n", iSlice, *trk.NTracks());

		if (mDeviceProcessingSettings.debugLevel >= 6) trk.DumpTrackHits(mDebugFile);

		trk.CommonMemory()->fNLocalTracks = trk.CommonMemory()->fNTracks;
		trk.CommonMemory()->fNLocalTrackHits = trk.CommonMemory()->fNTrackHits;
		if (!param().rec.GlobalTracking)
		{
			timerTPCtracking[iSlice][9].Start();
			trk.ReconstructOutput();
			timerTPCtracking[iSlice][9].Stop();
			//nOutputTracks += (*trk.Output())->NTracks();
			//nLocalTracks += trk.CommonMemory()->fNTracks;
			if (!mDeviceProcessingSettings.eventDisplay)
			{
				trk.SetupCommonMemory();
			}
		}
	}
	if (error) return(1);

	if (param().rec.GlobalTracking)
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
			timerTPCtracking[iSlice][8].Start();
			workers()->tpcTrackers[iSlice].PerformGlobalTracking(workers()->tpcTrackers[sliceLeft], workers()->tpcTrackers[sliceRight], workers()->tpcTrackers[sliceLeft].NMaxTracks(), workers()->tpcTrackers[sliceRight].NMaxTracks());
			timerTPCtracking[iSlice][8].Stop();
		}
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			timerTPCtracking[iSlice][9].Start();
			workers()->tpcTrackers[iSlice].ReconstructOutput();
			timerTPCtracking[iSlice][9].Stop();
			//printf("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d\n", iSlice, workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks, workers()->tpcTrackers[iSlice].CommonMemory()->fNTracks, workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits, workers()->tpcTrackers[iSlice].CommonMemory()->fNTrackHits);
			//nLocalTracks += workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTracks;
			//nGlobalTracks += workers()->tpcTrackers[iSlice].CommonMemory()->fNTracks;
			//nLocalHits += workers()->tpcTrackers[iSlice].CommonMemory()->fNLocalTrackHits;
			//nGlobalHits += workers()->tpcTrackers[iSlice].CommonMemory()->fNTrackHits;
			//nOutputTracks += (*workers()->tpcTrackers[iSlice].Output())->NTracks();
			if (!mDeviceProcessingSettings.eventDisplay)
			{
				workers()->tpcTrackers[iSlice].SetupCommonMemory();
			}
		}
	}
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError != 0)
		{
			const char* errorMsgs[] = GPUCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[workers()->tpcTrackers[iSlice].GPUParameters()->fGPUError];
			CAGPUError("Error during tracking: %s\n", errorMsg);
			return(1);
		}
	}
	//printf("Slice Tracks Output %d: - Tracks: %d local, %d global -  Hits: %d local, %d global\n", nOutputTracks, nLocalTracks, nGlobalTracks, nLocalHits, nGlobalHits);
	if (mDeviceProcessingSettings.debugMask & 1024)
	{
		for (unsigned int i = 0;i < NSLICES;i++)
		{
			workers()->tpcTrackers[i].DumpOutput(stdout);
		}
	}
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
	
	const bool needQA = AliGPUCAQA::QAAvailable() && (mDeviceProcessingSettings.runQA || (mDeviceProcessingSettings.eventDisplay && mIOPtrs.nMCInfosTPC));
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
	
	if (mDeviceProcessingSettings.runTRDTracker && mIOPtrs.nTRDTracklets)
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
