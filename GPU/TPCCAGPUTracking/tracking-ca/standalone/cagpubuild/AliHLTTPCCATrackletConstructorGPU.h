#include "AliHLTTPCCAGPUConfig.h"

GPUdi() void AliHLTTPCCATrackletConstructor::CopyTrackletTempData( AliHLTTPCCAThreadMemory &rMemSrc, AliHLTTPCCAThreadMemory &rMemDst, AliHLTTPCCATrackParam &tParamSrc, AliHLTTPCCATrackParam &tParamDst)
{
	//Copy Temporary Tracklet data from registers to global mem and vice versa
	rMemDst.fStartRow = rMemSrc.fStartRow;
	rMemDst.fEndRow = rMemSrc.fEndRow;
	rMemDst.fFirstRow = rMemSrc.fFirstRow;
	rMemDst.fLastRow = rMemSrc.fLastRow;
	rMemDst.fCurrIH =  rMemSrc.fCurrIH;
	rMemDst.fGo = rMemSrc.fGo;
	rMemDst.fStage = rMemSrc.fStage;
	rMemDst.fNHits = rMemSrc.fNHits;
	rMemDst.fNMissed = rMemSrc.fNMissed;
	rMemDst.fLastY = rMemSrc.fLastY;
	rMemDst.fLastZ = rMemSrc.fLastZ;

	tParamDst.SetSinPhi( tParamSrc.GetSinPhi() );
	tParamDst.SetDzDs( tParamSrc.GetDzDs() );
	tParamDst.SetQPt( tParamSrc.GetQPt() );
	tParamDst.SetSignCosPhi( tParamSrc.GetSignCosPhi() );
	tParamDst.SetChi2( tParamSrc.GetChi2() );
	tParamDst.SetNDF( tParamSrc.GetNDF() );
	tParamDst.SetCov( 0, tParamSrc.GetCov(0) );
	tParamDst.SetCov( 1, tParamSrc.GetCov(1) );
	tParamDst.SetCov( 2, tParamSrc.GetCov(2) );
	tParamDst.SetCov( 3, tParamSrc.GetCov(3) );
	tParamDst.SetCov( 4, tParamSrc.GetCov(4) );
	tParamDst.SetCov( 5, tParamSrc.GetCov(5) );
	tParamDst.SetCov( 6, tParamSrc.GetCov(6) );
	tParamDst.SetCov( 7, tParamSrc.GetCov(7) );
	tParamDst.SetCov( 8, tParamSrc.GetCov(8) );
	tParamDst.SetCov( 9, tParamSrc.GetCov(9) );
	tParamDst.SetCov( 10, tParamSrc.GetCov(10) );
	tParamDst.SetCov( 11, tParamSrc.GetCov(11) );
	tParamDst.SetCov( 12, tParamSrc.GetCov(12) );
	tParamDst.SetCov( 13, tParamSrc.GetCov(13) );
	tParamDst.SetCov( 14, tParamSrc.GetCov(14) );
	tParamDst.SetX( tParamSrc.GetX() );
	tParamDst.SetY( tParamSrc.GetY() );
	tParamDst.SetZ( tParamSrc.GetZ() );
}

#ifndef HLTCA_GPU_ALTERNATIVE_SCHEDULER
GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(AliHLTTPCCATracker &tracker, AliHLTTPCCASharedMemory &sMem, int Reverse, int RowBlock, int &mustInit)
{
	//Fetch a new trackled to be processed by this thread
	__syncthreads();
	int nextTrackletFirstRun = sMem.fNextTrackletFirstRun;
	if (threadIdx.x == 0)
	{
		sMem.fNTracklets = *tracker.NTracklets();
		if (sMem.fNextTrackletFirstRun)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_START
			const int iSlice = tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + (gridDim.x % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x;
			const int nSliceBlockOffset = gridDim.x * iSlice / tracker.GPUParametersConst()->fGPUnSlices;
			const uint2 &nTracklet = tracker.BlockStartingTracklet()[blockIdx.x - nSliceBlockOffset];

			sMem.fNextTrackletCount = nTracklet.y;
			if (sMem.fNextTrackletCount == 0)
			{
				sMem.fNextTrackletFirstRun = 0;
			}
			else
			{
				if (tracker.TrackletStartHits()[nTracklet.x].RowIndex() / HLTCA_GPU_SCHED_ROW_STEP != RowBlock)
				{
					sMem.fNextTrackletCount = 0;
				}
				else
				{
					sMem.fNextTrackletFirst = nTracklet.x;
				}
			}
#endif //HLTCA_GPU_SCHED_FIXED_START
		}
		else
		{
			const int4 oldPos = *tracker.RowBlockPos(Reverse, RowBlock);
			const int nFetchTracks = CAMath::Max(CAMath::Min(oldPos.x - oldPos.y, HLTCA_GPU_THREAD_COUNT), 0);
			sMem.fNextTrackletCount = nFetchTracks;
			const int nUseTrack = nFetchTracks ? CAMath::AtomicAdd(&(*tracker.RowBlockPos(Reverse, RowBlock)).y, nFetchTracks) : 0;
			sMem.fNextTrackletFirst = nUseTrack;

			const int nFillTracks = CAMath::Min(nFetchTracks, nUseTrack + nFetchTracks - (*((volatile int2*) (tracker.RowBlockPos(Reverse, RowBlock)))).x);
			if (nFillTracks > 0)
			{
				const int nStartFillTrack = CAMath::AtomicAdd(&(*tracker.RowBlockPos(Reverse, RowBlock)).x, nFillTracks);
				if (nFillTracks + nStartFillTrack >= HLTCA_GPU_MAX_TRACKLETS)
				{
					tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW;
				}
				for (int i = 0;i < nFillTracks;i++)
				{
					tracker.RowBlockTracklets(Reverse, RowBlock)[(nStartFillTrack + i) % HLTCA_GPU_MAX_TRACKLETS] = -(blockIdx.x * 1000000 + nFetchTracks * 10000 + oldPos.x * 100 + oldPos.y);	//Dummy filling track
				}
			}
		}
	}
	__syncthreads();
	mustInit = 0;
	if (sMem.fNextTrackletCount == 0)
	{
		return(-2);		//No more track in this RowBlock
	}
	else if (threadIdx.x >= sMem.fNextTrackletCount)
	{
		return(-1);		//No track in this RowBlock for this thread
	}
	else if (nextTrackletFirstRun)
	{
		if (threadIdx.x == 0) sMem.fNextTrackletFirstRun = 0;
		mustInit = 1;
		return(sMem.fNextTrackletFirst + threadIdx.x);
	}
	else
	{
		const int nTrackPos = sMem.fNextTrackletFirst + threadIdx.x;
		mustInit = (nTrackPos < tracker.RowBlockPos(Reverse, RowBlock)->w);
		volatile int* const ptrTracklet = &tracker.RowBlockTracklets(Reverse, RowBlock)[nTrackPos % HLTCA_GPU_MAX_TRACKLETS];
		int nTracklet;
		int nTryCount = 0;
		while ((nTracklet = *ptrTracklet) == -1)
		{
			for (int i = 0;i < 20000;i++)
				sMem.fNextTrackletStupidDummy++;
			nTryCount++;
			if (nTryCount > 30)
			{
				tracker.GPUParameters()->fGPUError = HLTCA_GPU_ERROR_SCHEDULE_COLLISION;
				return(-1);
			}
		};
		return(nTracklet);
	}
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(AliHLTTPCCATracker *pTracker)
{
	//Main Tracklet construction function that calls the scheduled (FetchTracklet) and then Processes the tracklet (mainly UpdataTracklet) and at the end stores the tracklet.
	//Can also dispatch a tracklet to be rescheduled
#ifdef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
	pTracker[0].BlockStartingTracklet()[0].x = HLTCA_GPU_EMULATION_SINGLE_TRACKLET;
	pTracker[0].BlockStartingTracklet()[0].y = 1;
	for (int i = 1;i < gridDim.x;i++)
	{
		pTracker[0].BlockStartingTracklet()[i].x = pTracker[0].BlockStartingTracklet()[i].y = 0;
	}
#endif //HLTCA_GPU_EMULATION_SINGLE_TRACKLET

	GPUshared() AliHLTTPCCASharedMemory sMem;

#ifdef HLTCA_GPU_SCHED_FIXED_START
	if (threadIdx.x == 0)
	{
		sMem.fNextTrackletFirstRun = 1;
	}
	__syncthreads();
#endif //HLTCA_GPU_SCHED_FIXED_START

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (threadIdx.x == 0)
	{
		sMem.fMaxSync = 0;
	}
	int threadSync = 0;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

	for (int iReverse = 0;iReverse < 2;iReverse++)
	{
		for (volatile int iRowBlock = 0;iRowBlock < HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1;iRowBlock++)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_SLICE
			int iSlice = pTracker[0].GPUParametersConst()->fGPUnSlices * (blockIdx.x + (gridDim.x % pTracker[0].GPUParametersConst()->fGPUnSlices != 0 && pTracker[0].GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x;
#else
			for (int iSlice = 0;iSlice < pTracker[0].GPUParametersConst()->fGPUnSlices;iSlice++)
#endif //HLTCA_GPU_SCHED_FIXED_SLICE
			{
				AliHLTTPCCATracker &tracker = pTracker[iSlice];
				if (blockIdx.x != 7 && sMem.fNextTrackletFirstRun && iSlice != (tracker.GPUParametersConst()->fGPUnSlices > gridDim.x ? blockIdx.x : (tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + (gridDim.x % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x)))
				{
					continue;
				}

				int sharedRowsInitialized = 0;

				int iTracklet;
				int mustInit;
				while ((iTracklet = FetchTracklet(tracker, sMem, iReverse, iRowBlock, mustInit)) != -2)
				{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					CAMath::AtomicMax(&sMem.fMaxSync, threadSync);
					__syncthreads();
					threadSync = CAMath::Min(sMem.fMaxSync, 100000000 / blockDim.x / gridDim.x);
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					if (!sharedRowsInitialized)
					{
						for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
						{
							reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
						}
						sharedRowsInitialized = 1;
					}
#ifdef HLTCA_GPU_RESCHED
					short2 storeToRowBlock;
					int storePosition = 0;
					if (threadIdx.x < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = threadIdx.x / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = threadIdx.x % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						sMem.fTrackletStoreCount[nReverse][nRowBlock] = 0;
					}
#else
					mustInit = 1;
#endif //HLTCA_GPU_RESCHED
					__syncthreads();
					AliHLTTPCCATrackParam tParam;
					AliHLTTPCCAThreadMemory rMem;

#ifdef HLTCA_GPU_EMULATION_DEBUG_TRACKLET
					if (iTracklet == HLTCA_GPU_EMULATION_DEBUG_TRACKLET)
					{
						tracker.GPUParameters()->fGPUError = 1;
					}
#endif //HLTCA_GPU_EMULATION_DEBUG_TRACKLET
					AliHLTTPCCAThreadMemory &rMemGlobal = tracker.GPUTrackletTemp()[iTracklet].fThreadMem;
					AliHLTTPCCATrackParam &tParamGlobal = tracker.GPUTrackletTemp()[iTracklet].fParam;
					if (mustInit)
					{
						AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];

						rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
						rMem.fCurrIH = id.HitIndex();
						rMem.fStage = 0;
						rMem.fNHits = 0;
						rMem.fNMissed = 0;

						AliHLTTPCCATrackletConstructor::InitTracklet(tParam);
					}
					else if (iTracklet >= 0)
					{
						CopyTrackletTempData( rMemGlobal, rMem, tParamGlobal, tParam );
					}
					rMem.fItr = iTracklet;
					rMem.fGo = (iTracklet >= 0);

#ifdef HLTCA_GPU_RESCHED
					storeToRowBlock.x = iRowBlock + 1;
					storeToRowBlock.y = iReverse;
					if (iReverse)
					{
						for (int j = HLTCA_ROW_COUNT - 1 - iRowBlock * HLTCA_GPU_SCHED_ROW_STEP;j >= CAMath::Max(0, HLTCA_ROW_COUNT - (iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP);j--)
						{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && !(j >= rMem.fEndRow || ( j >= rMem.fStartRow && j - rMem.fStartRow % 2 == 0)))
								pTracker[0].StageAtSync()[threadSync++ * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (iTracklet >= 0)
							{
								UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
								if (rMem.fNMissed > kMaxRowGap && j <= rMem.fStartRow)
								{
									rMem.fGo = 0;
									break;
								}
							}
						}
							
						if (iTracklet >= 0 && (!rMem.fGo || iRowBlock == HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP))
						{
							StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
						}
					}
					else
					{
						for (int j = CAMath::Max(1, iRowBlock * HLTCA_GPU_SCHED_ROW_STEP);j < CAMath::Min((iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP, HLTCA_ROW_COUNT);j++)
						{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && j >= rMem.fStartRow && (rMem.fStage > 0 || rMem.fCurrIH >= 0 || (j - rMem.fStartRow) % 2 == 0 ))
								pTracker[0].StageAtSync()[threadSync++ * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (iTracklet >= 0)
							{
								UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
								//if (rMem.fNMissed > kMaxRowGap || rMem.fGo == 0) break;	//DR!!! CUDA Crashes with this enabled
							}
						}
						if (rMem.fGo && (rMem.fNMissed > kMaxRowGap || iRowBlock == HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP))
						{
							if ( !tParam.TransportToX( sMem.fRows[ rMem.fEndRow ].X(), tracker.Param().ConstBz(), .999 ) )
							{
								rMem.fGo = 0;
							}
							else
							{
								storeToRowBlock.x = (HLTCA_ROW_COUNT - rMem.fEndRow) / HLTCA_GPU_SCHED_ROW_STEP;
								storeToRowBlock.y = 1;
								rMem.fNMissed = 0;
								rMem.fStage = 2;
							}
						}

						if (iTracklet >= 0 && !rMem.fGo)
						{
							StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
						}
					}

					if (rMem.fGo && (iRowBlock != HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						CopyTrackletTempData( rMem, rMemGlobal, tParam, tParamGlobal );
						storePosition = CAMath::AtomicAdd(&sMem.fTrackletStoreCount[storeToRowBlock.y][storeToRowBlock.x], 1);
					}

					__syncthreads();
					if (threadIdx.x < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = threadIdx.x / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = threadIdx.x % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						if (sMem.fTrackletStoreCount[nReverse][nRowBlock])
						{
							sMem.fTrackletStoreCount[nReverse][nRowBlock] = CAMath::AtomicAdd(&tracker.RowBlockPos(nReverse, nRowBlock)->x, sMem.fTrackletStoreCount[nReverse][nRowBlock]);
						}
					}
					__syncthreads();
					if (iTracklet >= 0 && rMem.fGo && (iRowBlock != HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						tracker.RowBlockTracklets(storeToRowBlock.y, storeToRowBlock.x)[sMem.fTrackletStoreCount[storeToRowBlock.y][storeToRowBlock.x] + storePosition] = iTracklet;
					}
					__syncthreads();
#else
					if (threadIdx.x % HLTCA_GPU_WARP_SIZE == 0)
					{
						sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 160;
						sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 0;
					}
					__syncthreads();
					if (iTracklet >= 0)
					{
						CAMath::AtomicMin(&sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fStartRow);
					}
					__syncthreads();
					if (iTracklet >= 0)
					{
						for (int j = sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE];j < HLTCA_ROW_COUNT;j++)
						{
							UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
							if (!rMem.fGo) break;
						}

						rMem.fNMissed = 0;
						rMem.fStage = 2;
						if ( rMem.fGo )
						{
							if ( !tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999 ) )  rMem.fGo = 0;
						}
						CAMath::AtomicMax(&sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fEndRow);
					}

					__syncthreads();
					if (iTracklet >= 0)
					{
						for (int j = rMem.fEndRow;j >= 0;j--)
						{
							if (!rMem.fGo) break;
							UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam, j);
						}

						StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, tracker, tParam );
					}
#endif //HLTCA_GPU_RESCHED
				}
			}
		}
	}
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(int iTracklet, AliHLTTPCCATracker &tracker)
{
	//Initialize Row Blocks

#ifndef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];
#ifdef HLTCA_GPU_SCHED_FIXED_START
	const int firstDynamicTracklet = tracker.GPUParameters()->fScheduleFirstDynamicTracklet;
	if (iTracklet >= firstDynamicTracklet)
#endif //HLTCA_GPU_SCHED_FIXED_START
	{
#ifdef HLTCA_GPU_SCHED_FIXED_START
		const int firstTrackletInRowBlock = CAMath::Max(firstDynamicTracklet, tracker.RowStartHitCountOffset()[CAMath::Max(id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP * HLTCA_GPU_SCHED_ROW_STEP, 1)].z);
#else
		const int firstTrackletInRowBlock = tracker.RowStartHitCountOffset()[CAMath::Max(id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP * HLTCA_GPU_SCHED_ROW_STEP, 1)].z;
#endif //HLTCA_GPU_SCHED_FIXED_START

		if (iTracklet == firstTrackletInRowBlock)
		{
			const int firstRowInNextBlock = (id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP + 1) * HLTCA_GPU_SCHED_ROW_STEP;
			int trackletsInRowBlock;
			if (firstRowInNextBlock >= HLTCA_ROW_COUNT - 3)
				trackletsInRowBlock = *tracker.NTracklets() - firstTrackletInRowBlock;
			else
#ifdef HLTCA_GPU_SCHED_FIXED_START
				trackletsInRowBlock = CAMath::Max(firstDynamicTracklet, tracker.RowStartHitCountOffset()[firstRowInNextBlock].z) - firstTrackletInRowBlock;
#else
				trackletsInRowBlock = tracker.RowStartHitCountOffset()[firstRowInNextBlock].z - firstTrackletInRowBlock;
#endif //HLTCA_GPU_SCHED_FIXED_START

			tracker.RowBlockPos(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)->x = trackletsInRowBlock;
			tracker.RowBlockPos(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)->w = trackletsInRowBlock;
		}
		tracker.RowBlockTracklets(0, id.RowIndex() / HLTCA_GPU_SCHED_ROW_STEP)[iTracklet - firstTrackletInRowBlock] = iTracklet;
	}
#endif //!HLTCA_GPU_EMULATION_SINGLE_TRACKLET
}

GPUg() void AliHLTTPCCATrackletConstructorInit(int iSlice)
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit
	AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[iSlice];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= *tracker.NTracklets()) return;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(i, tracker);
}

#else //HLTCA_GPU_ALTERNATIVE_SCHEDULER

GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(AliHLTTPCCATracker &tracker, AliHLTTPCCASharedMemory &sMem)
{
	const int nativeslice = blockIdx.x % tracker.GPUParametersConst()->fGPUnSlices;
	const int nTracklets = *tracker.NTracklets();
	__syncthreads();
	if (sMem.fNextTrackletFirstRun == 1)
	{
		if (threadIdx.x == 0)
		{
			sMem.fNextTrackletFirst = (blockIdx.x - nativeslice) / tracker.GPUParametersConst()->fGPUnSlices * HLTCA_GPU_THREAD_COUNT;
			sMem.fNextTrackletFirstRun = 0;
		}
	}
	else
	{
		if (threadIdx.x == 0)
		{
			if (tracker.GPUParameters()->fNextTracklet < nTracklets)
			{
				const int firstTracklet = /*sMem.fNextTrackletCount != HLTCA_GPU_THREAD_COUNT ? -1 :*/ CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, sMem.fNextTrackletCount);
				if (firstTracklet < nTracklets) sMem.fNextTrackletFirst = firstTracklet;
				else sMem.fNextTrackletFirst = -1;
			}
			else
			{
				sMem.fNextTrackletFirst = -1;
			}
		}
	}
	if (threadIdx.x == 0)
	{
		if (sMem.fNextTrackletFirst == -1 && sMem.fNextTrackletCount == HLTCA_GPU_THREAD_COUNT)
		{
			sMem.fNextTrackletFirst = -2;
			sMem.fNextTrackletCount = HLTCA_GPU_THREAD_COUNT;
		}
		else
		{
			sMem.fNextTrackletCount = 0;
		}
	}
	__syncthreads();
	return (sMem.fNextTrackletFirst);
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(AliHLTTPCCATracker *pTracker)
{
	const int nSlices = pTracker[0].GPUParametersConst()->fGPUnSlices;
	const int nativeslice = blockIdx.x % nSlices;
	GPUshared() AliHLTTPCCASharedMemory sMem;
	int currentSlice = -1;

	if (threadIdx.x)
	{
		sMem.fNextTrackletFirstRun = 1;
	}

	for (int iSlice = 0;iSlice < nSlices;iSlice++)
	{
		AliHLTTPCCATracker &tracker = pTracker[(nativeslice + iSlice) % nSlices];
		int iTracklet;
		int keepTracklet = 0;
		int iPhase = 0;

		int iRowStart, iRowEnd, iRowIncrement;

		int iNextLocalTracklet = threadIdx.x;

		AliHLTTPCCATrackParam tParam;
		AliHLTTPCCAThreadMemory rMem;

		int tmpTracklet;
		while ((tmpTracklet = FetchTracklet(tracker, sMem)) != -2)
		{
			if (!keepTracklet)
			{
				if (tmpTracklet >= 0)
				{
					iTracklet = tmpTracklet + iNextLocalTracklet;
				}
				else
				{
					iTracklet = -1;
				}
			}

			if (iSlice != currentSlice)
			{
				if (threadIdx.x == 0)
				{
					sMem.fNTracklets = *tracker.NTracklets();
				}

				for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
				{
					reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
				}
				currentSlice = iSlice;
			}

			if (iTracklet >= 0)
			{
				if (keepTracklet) {}
				else if (iTracklet < sMem.fNTracklets)
				{
					AliHLTTPCCAHitId id = tracker.TrackletStartHits()[iTracklet];

					rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
					rMem.fCurrIH = id.HitIndex();
					rMem.fStage = 0;
					rMem.fNHits = 0;
					rMem.fNMissed = 0;

					AliHLTTPCCATrackletConstructor::InitTracklet(tParam);

					rMem.fItr = iTracklet;
					rMem.fGo = 1;

					keepTracklet = 1;

					iRowStart = rMem.fStartRow;
					iRowEnd = tracker.Param().NRows();
					iRowIncrement = 1;
					iPhase = 0;
				}
				else
				{
					rMem.fGo = 0;
					rMem.fStartRow = rMem.fEndRow = 0;
					iRowStart = iRowEnd = 0;
				}

				for (int j = iRowStart;j != iRowEnd;j+=iRowIncrement)
				{
					UpdateTracklet(1, 1, 0, iTracklet, sMem, rMem, tracker, tParam, j);
				}

				if (iPhase)
				{
					rMem.fGo = 0;
				}
				else
				{
					rMem.fNMissed = 0;
					rMem.fStage = 2;
					if (rMem.fGo) if (!tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999)) rMem.fGo = 0;
					iRowStart = rMem.fEndRow;
					iRowEnd = -1;
					iRowIncrement = -1;
					iPhase = 1;
				}
			}

			if (!rMem.fGo)
			{
				if (iTracklet < sMem.fNTracklets && iTracklet >= 0) StoreTracklet( 1, 1, 0, iTracklet, sMem, rMem, tracker, tParam );
				keepTracklet = 0;
				iNextLocalTracklet = CAMath::AtomicAdd(&sMem.fNextTrackletCount, 1);
			}
		}
	}
}

#endif //HLTCA_GPU_ALTERNATIVE_SCHEDULER

GPUg() void AliHLTTPCCATrackletConstructorGPU()
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU
	AliHLTTPCCATracker *pTracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(pTracker);
}

GPUg() void AliHLTTPCCATrackletConstructorGPUPP(int firstSlice, int sliceCount)
{
	if (blockIdx.x >= sliceCount) return;
	AliHLTTPCCATracker *pTracker = &( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + blockIdx.x];
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPUPP(pTracker);
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPUPP(AliHLTTPCCATracker *tracker)
{
	GPUshared() AliHLTTPCCASharedMemory sMem;
	sMem.fNTracklets = *tracker->NTracklets();

	for (int i = threadIdx.x;i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += blockDim.x)
	{
		reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker->SliceDataRows())[i];
	}

	for (int iTracklet = threadIdx.x;iTracklet < (*tracker->NTracklets() / HLTCA_GPU_THREAD_COUNT + 1) * HLTCA_GPU_THREAD_COUNT;iTracklet += blockDim.x)
	{
		AliHLTTPCCATrackParam tParam;
		AliHLTTPCCAThreadMemory rMem;
		
		if (iTracklet < *tracker->NTracklets())
		{
			AliHLTTPCCAHitId id = tracker->TrackletTmpStartHits()[iTracklet];

			rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
			rMem.fCurrIH = id.HitIndex();
			rMem.fStage = 0;
			rMem.fNHits = 0;
			rMem.fNMissed = 0;

			AliHLTTPCCATrackletConstructor::InitTracklet(tParam);

			rMem.fItr = iTracklet;
			rMem.fGo = 1;
		}

		if (threadIdx.x % HLTCA_GPU_WARP_SIZE == 0)
		{
			sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 160;
			sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE] = 0;
		}
		__syncthreads();
		if (iTracklet < *tracker->NTracklets())
		{
			CAMath::AtomicMin(&sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fStartRow);
		}
		__syncthreads();
		if (iTracklet < *tracker->NTracklets())
		{
			for (int j = sMem.fStartRows[threadIdx.x / HLTCA_GPU_WARP_SIZE];j < HLTCA_ROW_COUNT;j++)
			{
				UpdateTracklet(gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam, j);
				if (!rMem.fGo) break;
			}

			rMem.fNMissed = 0;
			rMem.fStage = 2;
			if ( rMem.fGo )
			{
				if ( !tParam.TransportToX( tracker->Row( rMem.fEndRow ).X(), tracker->Param().ConstBz(), .999 ) )  rMem.fGo = 0;
			}
			CAMath::AtomicMax(&sMem.fEndRows[threadIdx.x / HLTCA_GPU_WARP_SIZE], rMem.fEndRow);
		}

		__syncthreads();
		if (iTracklet < *tracker->NTracklets())
		{
			for (int j = rMem.fEndRow;j >= 0;j--)
			{
				if (!rMem.fGo) break;
				UpdateTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam, j);
			}
			StoreTracklet( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, sMem, rMem, *tracker, tParam );
		}
	}
}