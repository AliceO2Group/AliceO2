#include "AliHLTTPCCAGPUConfig.h"

MEM_TEMPLATE4() GPUdi() void AliHLTTPCCATrackletConstructor::CopyTrackletTempData( MEM_TYPE(AliHLTTPCCAThreadMemory) &rMemSrc, MEM_TYPE2(AliHLTTPCCAThreadMemory) &rMemDst, MEM_TYPE3(AliHLTTPCCATrackParam) &tParamSrc, MEM_TYPE4(AliHLTTPCCATrackParam) &tParamDst)
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

#if defined(HLTCA_GPU_ALTERNATIVE_SCHEDULER) & !defined(HLTCA_GPU_ALTERNATIVE_SCHEDULER_SIMPLE)
	rMemDst.fItr = rMemSrc.fItr;
	rMemDst.fIRow = rMemSrc.fIRow;
	rMemDst.fIRowEnd = rMemSrc.fIRowEnd;
#endif

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
GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, GPUshared() MEM_LOCAL(AliHLTTPCCASharedMemory) &sMem, int Reverse, int RowBlock, int &mustInit)
{
	//Fetch a new trackled to be processed by this thread
	GPUsync();
	int nextTrackletFirstRun = sMem.fNextTrackletFirstRun;
	if (get_local_id(0) == 0)
	{
		sMem.fNTracklets = *tracker.NTracklets();
		if (sMem.fNextTrackletFirstRun)
		{
#ifdef HLTCA_GPU_SCHED_FIXED_START
			const int iSlice = tracker.GPUParametersConst()->fGPUnSlices * (get_group_id(0) + (get_num_groups(0) % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
			const int nSliceBlockOffset = get_num_groups(0) * iSlice / tracker.GPUParametersConst()->fGPUnSlices;
			const uint2 &nTracklet = tracker.BlockStartingTracklet()[get_group_id(0) - nSliceBlockOffset];

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
			const int nFetchTracks = CAMath::Max(CAMath::Min(oldPos.x - oldPos.y, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR), 0);
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
					tracker.RowBlockTracklets(Reverse, RowBlock)[(nStartFillTrack + i) % HLTCA_GPU_MAX_TRACKLETS] = -(get_group_id(0) * 1000000 + nFetchTracks * 10000 + oldPos.x * 100 + oldPos.y);	//Dummy filling track
				}
			}
		}
	}
	GPUsync();
	mustInit = 0;
	if (sMem.fNextTrackletCount == 0)
	{
		return(-2);		//No more track in this RowBlock
	}
	else if (get_local_id(0) >= sMem.fNextTrackletCount)
	{
		return(-1);		//No track in this RowBlock for this thread
	}
	else if (nextTrackletFirstRun)
	{
		if (get_local_id(0) == 0) sMem.fNextTrackletFirstRun = 0;
		mustInit = 1;
		return(sMem.fNextTrackletFirst + get_local_id(0));
	}
	else
	{
		const int nTrackPos = sMem.fNextTrackletFirst + get_local_id(0);
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

MEM_CLASS_PRE2 GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(MEM_LG2(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem)
{
	//Main Tracklet construction function that calls the scheduled (FetchTracklet) and then Processes the tracklet (mainly UpdataTracklet) and at the end stores the tracklet.
	//Can also dispatch a tracklet to be rescheduled
#ifdef HLTCA_GPU_EMULATION_SINGLE_TRACKLET
	pTracker[0].BlockStartingTracklet()[0].x = HLTCA_GPU_EMULATION_SINGLE_TRACKLET;
	pTracker[0].BlockStartingTracklet()[0].y = 1;
	for (int i = 1;i < get_num_groups(0);i++)
	{
		pTracker[0].BlockStartingTracklet()[i].x = pTracker[0].BlockStartingTracklet()[i].y = 0;
	}
#endif //HLTCA_GPU_EMULATION_SINGLE_TRACKLET

	//GPUshared() AliHLTTPCCASharedMemory sMem;

#ifdef HLTCA_GPU_SCHED_FIXED_START
	if (get_local_id(0) == 0)
	{
		sMem.fNextTrackletFirstRun = 1;
	}
	GPUsync();
#endif //HLTCA_GPU_SCHED_FIXED_START

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (get_local_id(0) == 0)
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
			int iSlice = pTracker[0].GPUParametersConst()->fGPUnSlices * (get_group_id(0) + (get_num_groups(0) % pTracker[0].GPUParametersConst()->fGPUnSlices != 0 && pTracker[0].GPUParametersConst()->fGPUnSlices * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
#else
			for (int iSlice = 0;iSlice < pTracker[0].GPUParametersConst()->fGPUnSlices;iSlice++)
#endif //HLTCA_GPU_SCHED_FIXED_SLICE
			{
				AliHLTTPCCATracker &tracker = pTracker[iSlice];
				if (get_group_id(0) != 7 && sMem.fNextTrackletFirstRun && iSlice != (tracker.GPUParametersConst()->fGPUnSlices > get_num_groups(0) ? get_group_id(0) : (tracker.GPUParametersConst()->fGPUnSlices * (get_group_id(0) + (get_num_groups(0) % tracker.GPUParametersConst()->fGPUnSlices != 0 && tracker.GPUParametersConst()->fGPUnSlices * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0))))
				{
					continue;
				}

				int sharedRowsInitialized = 0;

				int iTracklet;
				int mustInit;
				while ((iTracklet = FetchTracklet(tracker, sMem, iReverse, iRowBlock, mustInit)) != -2)
				{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					CAMath::AtomicMaxShared(&sMem.fMaxSync, threadSync);
					GPUsync();
					threadSync = CAMath::Min(sMem.fMaxSync, 100000000 / get_local_size(0) / get_num_groups(0));
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					if (!sharedRowsInitialized)
					{
						for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += get_local_size(0))
						{
							reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker.SliceDataRows())[i];
						}
						sharedRowsInitialized = 1;
					}
#ifdef HLTCA_GPU_RESCHED
					short2 storeToRowBlock;
					int storePosition = 0;
					if (get_local_id(0) < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = get_local_id(0) / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = get_local_id(0) % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						sMem.fTrackletStoreCount[nReverse][nRowBlock] = 0;
					}
#else
					mustInit = 1;
#endif //HLTCA_GPU_RESCHED
					GPUsync();
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
								pTracker[0].StageAtSync()[threadSync++ * get_global_size(0) + get_global_id(0)] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (iTracklet >= 0)
							{
								UpdateTracklet(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam, j);
								if (rMem.fNMissed > kMaxRowGap && j <= rMem.fStartRow)
								{
									rMem.fGo = 0;
									break;
								}
							}
						}
							
						if (iTracklet >= 0 && (!rMem.fGo || iRowBlock == HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP))
						{
							StoreTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam );
						}
					}
					else
					{
						for (int j = CAMath::Max(1, iRowBlock * HLTCA_GPU_SCHED_ROW_STEP);j < CAMath::Min((iRowBlock + 1) * HLTCA_GPU_SCHED_ROW_STEP, HLTCA_ROW_COUNT);j++)
						{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && j >= rMem.fStartRow && (rMem.fStage > 0 || rMem.fCurrIH >= 0 || (j - rMem.fStartRow) % 2 == 0 ))
								pTracker[0].StageAtSync()[threadSync++ * get_global_size(0) + get_global_id(0)] = rMem.fStage + 1;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
							if (iTracklet >= 0)
							{
								UpdateTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam, j);
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
							StoreTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam );
						}
					}

					if (rMem.fGo && (iRowBlock != HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						CopyTrackletTempData( rMem, rMemGlobal, tParam, tParamGlobal );
						storePosition = CAMath::AtomicAddShared(&sMem.fTrackletStoreCount[storeToRowBlock.y][storeToRowBlock.x], 1);
					}

					GPUsync();
					if (get_local_id(0) < 2 * (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1))
					{
						const int nReverse = get_local_id(0) / (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						const int nRowBlock = get_local_id(0) % (HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP + 1);
						if (sMem.fTrackletStoreCount[nReverse][nRowBlock])
						{
							sMem.fTrackletStoreCount[nReverse][nRowBlock] = CAMath::AtomicAdd(&tracker.RowBlockPos(nReverse, nRowBlock)->x, sMem.fTrackletStoreCount[nReverse][nRowBlock]);
						}
					}
					GPUsync();
					if (iTracklet >= 0 && rMem.fGo && (iRowBlock != HLTCA_ROW_COUNT / HLTCA_GPU_SCHED_ROW_STEP || iReverse == 0))
					{
						tracker.RowBlockTracklets(storeToRowBlock.y, storeToRowBlock.x)[sMem.fTrackletStoreCount[storeToRowBlock.y][storeToRowBlock.x] + storePosition] = iTracklet;
					}
					GPUsync();
#else
					if (get_local_id(0) % HLTCA_GPU_WARP_SIZE == 0)
					{
						sMem.fStartRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE] = 160;
						sMem.fEndRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE] = 0;
					}
					GPUsync();
					if (iTracklet >= 0)
					{
						CAMath::AtomicMinShared(&sMem.fStartRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE], rMem.fStartRow);
					}
					GPUsync();
					if (iTracklet >= 0)
					{
						for (int j = sMem.fStartRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE];j < HLTCA_ROW_COUNT;j++)
						{
							UpdateTracklet(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam, j);
							if (!rMem.fGo) break;
						}

						rMem.fNMissed = 0;
						rMem.fStage = 2;
						if ( rMem.fGo )
						{
							if ( !tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999 ) )  rMem.fGo = 0;
						}
						CAMath::AtomicMaxShared(&sMem.fEndRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE], rMem.fEndRow);
					}

					GPUsync();
					if (iTracklet >= 0)
					{
						for (int j = rMem.fEndRow;j >= 0;j--)
						{
							if (!rMem.fGo) break;
							UpdateTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam, j);
						}

						StoreTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam );
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
	int i = get_global_id(0);
	if (i >= *tracker.NTracklets()) return;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorInit(i, tracker);
}

#elif defined(HLTCA_GPU_ALTERNATIVE_SCHEDULER_SIMPLE)

GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &sMem, AliHLTTPCCAThreadMemory& /*rMem*/, MEM_PLAIN(AliHLTTPCCATrackParam)& /*tParam*/)
{
	const int nativeslice = get_group_id(0) % tracker.GPUParametersConst()->fGPUnSlices;
	const int nTracklets = *tracker.NTracklets();
	GPUsync();
	if (sMem.fNextTrackletFirstRun == 1)
	{
		if (get_local_id(0) == 0)
		{
			sMem.fNextTrackletFirst = (get_group_id(0) - nativeslice) / tracker.GPUParametersConst()->fGPUnSlices * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
			sMem.fNextTrackletFirstRun = 0;
		}
	}
	else
	{
		if (get_local_id(0) == 0)
		{
			if (tracker.GPUParameters()->fNextTracklet < nTracklets)
			{
				const int firstTracklet = CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR);
				if (firstTracklet < nTracklets) sMem.fNextTrackletFirst = firstTracklet;
				else sMem.fNextTrackletFirst = -2;
			}
			else
			{
				sMem.fNextTrackletFirst = -2;
			}
		}
	}
	GPUsync();
	return (sMem.fNextTrackletFirst);
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem)
{
	const int nSlices = pTracker[0].GPUParametersConst()->fGPUnSlices;
	const int nativeslice = get_group_id(0) % nSlices;
	int currentSlice = -1;

	if (get_local_id(0))
	{
		sMem.fNextTrackletFirstRun = 1;
	}

	for (int iSlice = 0;iSlice < nSlices;iSlice++)
	{
		GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker = pTracker[(nativeslice + iSlice) % nSlices];
		int iRow, iRowEnd;

		MEM_PLAIN(AliHLTTPCCATrackParam) tParam;
		AliHLTTPCCAThreadMemory rMem;

		int tmpTracklet;
		while ((tmpTracklet = FetchTracklet(tracker, sMem, rMem, tParam)) != -2)
		{
			if (tmpTracklet >= 0)
			{
				rMem.fItr = tmpTracklet + get_local_id(0);
			}
			else
			{
				rMem.fItr = -1;
			}

			if (iSlice != currentSlice)
			{
				if (get_local_id(0) == 0)
				{
					sMem.fNTracklets = *tracker.NTracklets();
				}

				for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(MEM_PLAIN(AliHLTTPCCARow)) / sizeof(int);i += get_local_size(0))
				{
					reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
				}
				currentSlice = iSlice;
				GPUsync();
			}

			if (rMem.fItr < sMem.fNTracklets)
			{
				AliHLTTPCCAHitId id = tracker.TrackletStartHits()[rMem.fItr];

				rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
				rMem.fCurrIH = id.HitIndex();
				rMem.fStage = 0;
				rMem.fNHits = 0;
				rMem.fNMissed = 0;

				AliHLTTPCCATrackletConstructor::InitTracklet(tParam);

				rMem.fGo = 1;


				iRow = rMem.fStartRow;
				iRowEnd = tracker.Param().NRows();
			}
			else
			{
				rMem.fGo = 0;
				rMem.fStartRow = rMem.fEndRow = 0;
				iRow = iRowEnd = 0;
				rMem.fStage = 0;
			}

			for (int k = 0;k < 2;k++)
			{
				for (;iRow != iRowEnd;iRow += rMem.fStage == 2 ? -1 : 1)
				{
					UpdateTracklet(0, 0, 0, 0, sMem, rMem, tracker, tParam, iRow);
				}

				if (rMem.fStage == 2)
				{
					if (rMem.fItr < sMem.fNTracklets)
					{
						StoreTracklet( 0, 0, 0, 0, sMem, rMem, tracker, tParam );
					}
				}
				else
				{
					rMem.fNMissed = 0;
					rMem.fStage = 2;
					if (rMem.fGo) if (!tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999)) rMem.fGo = 0;
					iRow = rMem.fEndRow;
					iRowEnd = -1;
				}
			}
		}
	}
}
 

#else //HLTCA_GPU_ALTERNATIVE_SCHEDULER

GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &sMem, AliHLTTPCCAThreadMemory &rMem, MEM_PLAIN(AliHLTTPCCATrackParam) &tParam)
{
	const int nativeslice = get_group_id(0) % tracker.GPUParametersConst()->fGPUnSlices;
	const int nTracklets = *tracker.NTracklets();
	GPUsync();
	if (get_local_id(0) == 0) sMem.fTrackletStorePos = 0;
	int nStorePos = -1;
	if (sMem.fNextTrackletFirstRun == 1)
	{
		if (get_local_id(0) == 0)
		{
			sMem.fNextTrackletFirst = (get_group_id(0) - nativeslice) / tracker.GPUParametersConst()->fGPUnSlices * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
			sMem.fNextTrackletFirstRun = 0;
			sMem.fNextTrackletCount = HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
		}
	}
	else
	{
		if (sMem.fNextTrackletCount < HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR - HLTCA_GPU_ALTSCHED_MIN_THREADS)
		{
			if (get_local_id(0) == 0)
			{
				sMem.fNextTrackletFirst = -1;
			}
		}
		else
		{
			GPUsync();
			if (rMem.fItr != -1)
			{
				nStorePos = CAMath::AtomicAddShared(&sMem.fTrackletStorePos, 1);
				CopyTrackletTempData(rMem, sMem.swapMemory[nStorePos].fThreadMem, tParam, sMem.swapMemory[nStorePos].fParam);
				rMem.fItr = -1;
			}
			if (get_local_id(0) == 0)
			{
				if (tracker.GPUParameters()->fNextTracklet >= nTracklets)
				{
					sMem.fNextTrackletFirst = -1;
				}
				else
				{
					const int firstTracklet = CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, sMem.fNextTrackletCount);
					if (firstTracklet >= nTracklets)
					{
						sMem.fNextTrackletFirst = -1;
					}
					else
					{
						sMem.fNextTrackletFirst = firstTracklet;
					}
				}
			}
		}
	}

	if (get_local_id(0) == 0)
	{
		if (sMem.fNextTrackletFirst == -1 && sMem.fNextTrackletCount == HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR)
		{
			sMem.fNextTrackletFirst = -2;
			sMem.fNextTrackletCount = HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
		}
		else if (sMem.fNextTrackletFirst >= 0)
		{
			if (sMem.fNextTrackletFirst + sMem.fNextTrackletCount >= nTracklets)
			{
				sMem.fNextTrackletCount = sMem.fNextTrackletFirst + sMem.fNextTrackletCount - nTracklets;
			}
			else
			{
				sMem.fNextTrackletCount = 0;
			}
		}
	}
	GPUsync();
	if (get_local_id(0) < sMem.fTrackletStorePos)
	{
		CopyTrackletTempData(sMem.swapMemory[get_local_id(0)].fThreadMem, rMem, sMem.swapMemory[get_local_id(0)].fParam, tParam);
	}
	return (sMem.fNextTrackletFirst);
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem)
{
	const int nSlices = pTracker[0].GPUParametersConst()->fGPUnSlices;
	const int nativeslice = get_group_id(0) % nSlices;
	//GPUshared() AliHLTTPCCASharedMemory sMem;
	int currentSlice = -1;

	if (get_local_id(0))
	{
		sMem.fNextTrackletFirstRun = 1;
	}

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	if (get_local_id(0) == 0)
	{
		sMem.fMaxSync = 0;
	}
	int threadSync = 0;
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

	for (int iSlice = 0;iSlice < nSlices;iSlice++)
	{
		GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker = pTracker[(nativeslice + iSlice) % nSlices];

		MEM_PLAIN(AliHLTTPCCATrackParam) tParam;
		AliHLTTPCCAThreadMemory rMem;
		rMem.fItr = -1;

		int tmpTracklet;
		while ((tmpTracklet = FetchTracklet(tracker, sMem, rMem, tParam)) != -2)
		{

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					CAMath::AtomicMaxShared(&sMem.fMaxSync, threadSync);
					GPUsync();
					threadSync = CAMath::Min(sMem.fMaxSync, 100000000 / get_local_size(0) / get_num_groups(0));
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE

			if (iSlice != currentSlice)
			{
				if (get_local_id(0) == 0) sMem.fNTracklets = *tracker.NTracklets();

				for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(MEM_PLAIN(AliHLTTPCCARow)) / sizeof(int);i += get_local_size(0))
				{
					reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
				}
				currentSlice = iSlice;
				GPUsync();
			}

			if (tmpTracklet >= 0 && rMem.fItr < 0)
			{
				rMem.fItr = tmpTracklet + (signed) get_local_id(0) - sMem.fTrackletStorePos;
				if (rMem.fItr >= sMem.fNTracklets)
				{
					rMem.fItr = -1;
				}
				else
				{
					AliHLTTPCCAHitId id = tracker.TrackletStartHits()[rMem.fItr];

					rMem.fStartRow = rMem.fEndRow = rMem.fFirstRow = rMem.fLastRow = id.RowIndex();
					rMem.fCurrIH = id.HitIndex();
					rMem.fStage = 0;
					rMem.fNHits = 0;
					rMem.fNMissed = 0;

					AliHLTTPCCATrackletConstructor::InitTracklet(tParam);

					rMem.fGo = 1;

					rMem.fIRow = rMem.fStartRow;
					rMem.fIRowEnd = tracker.Param().NRows();
				}
			}

			if (rMem.fItr >= 0)
			{
				for (int j = 0;j < HLTCA_GPU_ALTSCHED_STEPSIZE && rMem.fIRow != rMem.fIRowEnd;j++,rMem.fIRow += rMem.fStage == 2 ? -1 : 1)
				{
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					if (rMem.fStage == 2)
					{
						if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && !(rMem.fIRow >= rMem.fEndRow || ( rMem.fIRow >= rMem.fStartRow && rMem.fIRow - rMem.fStartRow % 2 == 0)))
							pTracker[0].StageAtSync()[threadSync++ * get_global_size(0) + get_global_id(0)] = rMem.fStage + 1;
					}
					else
					{
						if (rMem.fNMissed <= kMaxRowGap && rMem.fGo && rMem.fIRow >= rMem.fStartRow && (rMem.fStage > 0 || rMem.fCurrIH >= 0 || (rMem.fIRow - rMem.fStartRow) % 2 == 0 ))
							pTracker[0].StageAtSync()[threadSync++ * get_global_size(0) + get_global_id(0)] = rMem.fStage + 1;
					}
#endif //HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
					UpdateTracklet(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam, rMem.fIRow);
				}

				if (rMem.fIRow == rMem.fIRowEnd || rMem.fNMissed > kMaxRowGap)
				{
					if (rMem.fStage >= 2)
					{
						rMem.fGo = 0;
					}
					else if (rMem.fGo)
					{
						rMem.fNMissed = 0;
						rMem.fStage = 2;
						if (!tParam.TransportToX( tracker.Row( rMem.fEndRow ).X(), tracker.Param().ConstBz(), .999)) rMem.fGo = 0;
						rMem.fIRow = rMem.fEndRow;
						rMem.fIRowEnd = -1;
					}
				}

				if (!rMem.fGo)
				{
					StoreTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, tracker, tParam );
					rMem.fItr = -1;
					CAMath::AtomicAddShared(&sMem.fNextTrackletCount, 1);
				}
			}
		}
	}
}

#endif //HLTCA_GPU_ALTERNATIVE_SCHEDULER

#ifndef __OPENCL__
GPUg() void AliHLTTPCCATrackletConstructorGPU()
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU
	AliHLTTPCCATracker *pTracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	GPUshared() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory) sMem;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(pTracker, sMem);
}

GPUg() void AliHLTTPCCATrackletConstructorGPUPP(int firstSlice, int sliceCount)
{
	if (get_group_id(0) >= sliceCount) return;
	AliHLTTPCCATracker *pTracker = &( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + get_group_id(0)];
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPUPP(pTracker);
}

GPUd() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPUPP(AliHLTTPCCATracker *tracker)
{
	GPUshared() AliHLTTPCCASharedMemory sMem;
#if defined(HLTCA_GPU_RESCHED) & !defined(HLTCA_GPU_ALTERNATIVE_SCHEDULER)
#define startRows sMem.fStartRows
#define endRows sMem.fEndRows
#else
	GPUshared() int startRows[HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR / HLTCA_GPU_WARP_SIZE + 1];
	GPUshared() int endRows[HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR / HLTCA_GPU_WARP_SIZE + 1];
#endif
	sMem.fNTracklets = *tracker->NTracklets();

	for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(AliHLTTPCCARow) / sizeof(int);i += get_local_size(0))
	{
		reinterpret_cast<int*>(&sMem.fRows)[i] = reinterpret_cast<int*>(tracker->SliceDataRows())[i];
	}

	for (int iTracklet = get_local_id(0);iTracklet < (*tracker->NTracklets() / HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR + 1) * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;iTracklet += get_local_size(0))
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

		if (get_local_id(0) % HLTCA_GPU_WARP_SIZE == 0)
		{
			startRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE] = 160;
			endRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE] = 0;
		}
		GPUsync();
		if (iTracklet < *tracker->NTracklets())
		{
			CAMath::AtomicMinShared(&startRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE], rMem.fStartRow);
		}
		GPUsync();
		if (iTracklet < *tracker->NTracklets())
		{
			for (int j = startRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE];j < HLTCA_ROW_COUNT;j++)
			{
				UpdateTracklet(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, *tracker, tParam, j);
				if (!rMem.fGo) break;
			}

			rMem.fNMissed = 0;
			rMem.fStage = 2;
			if ( rMem.fGo )
			{
				if ( !tParam.TransportToX( tracker->Row( rMem.fEndRow ).X(), tracker->Param().ConstBz(), .999 ) )  rMem.fGo = 0;
			}
			CAMath::AtomicMaxShared(&endRows[get_local_id(0) / HLTCA_GPU_WARP_SIZE], rMem.fEndRow);
		}

		GPUsync();
		if (iTracklet < *tracker->NTracklets())
		{
			for (int j = rMem.fEndRow;j >= 0;j--)
			{
				if (!rMem.fGo) break;
				UpdateTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, *tracker, tParam, j);
			}
			StoreTracklet( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), sMem, rMem, *tracker, tParam );
		}
	}
}

#endif
