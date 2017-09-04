#include "AliHLTTPCCAGPUConfig.h"

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

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorSingleSlice(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem)
{
	GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker = *pTracker;
	int iRow, iRowEnd;

	MEM_PLAIN(AliHLTTPCCATrackParam) tParam;
	AliHLTTPCCAThreadMemory rMem;

	if (get_local_id(0) == 0) sMem.fNTracklets = *tracker.NTracklets();
	for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(MEM_PLAIN(AliHLTTPCCARow)) / sizeof(int);i += get_local_size(0))
	{
		reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
	}
	GPUsync();

	for (rMem.fItr = get_global_id(0);rMem.fItr < sMem.fNTracklets;rMem.fItr += get_global_size(0))
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

		for (int k = 0;k < 2;k++)
		{
			for (;iRow != iRowEnd;iRow += rMem.fStage == 2 ? -1 : 1)
			{
				UpdateTracklet(0, 0, 0, 0, sMem, rMem, tracker, tParam, iRow);
			}

			if (rMem.fStage == 2)
			{
				StoreTracklet( 0, 0, 0, 0, sMem, rMem, tracker, tParam );
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

#ifndef __OPENCL__
GPUg() void AliHLTTPCCATrackletConstructorGPU()
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU
	AliHLTTPCCATracker *pTracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	GPUshared() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory) sMem;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(pTracker, sMem);
}

GPUg() void AliHLTTPCCATrackletConstructorSingleSlice(int iSlice)
{
	GPUshared() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory) sMem;
	AliHLTTPCCATracker *pTracker = ((AliHLTTPCCATracker*) gAliHLTTPCCATracker) + iSlice;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorSingleSlice(pTracker, sMem);
}
#endif
