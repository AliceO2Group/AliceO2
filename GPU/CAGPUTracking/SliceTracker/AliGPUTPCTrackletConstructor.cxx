// @(#) $Id: AliGPUTPCTrackletConstructor.cxx 27042 2008-07-02 12:06:02Z richterm $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#define GPUCA_CADEBUG 0

#include "AliGPUTPCDef.h"
#include "AliGPUTPCGrid.h"
#include "AliGPUTPCHit.h"
#include "AliGPUTPCTrackParam.h"
#include "AliGPUTPCTracker.h"
#include "AliGPUTPCTracklet.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliTPCCommonMath.h"

MEM_CLASS_PRE2()
GPUd() void AliGPUTPCTrackletConstructor::InitTracklet(MEM_LG2(AliGPUTPCTrackParam) & tParam)
{
	//Initialize Tracklet Parameters using default values
	tParam.InitParam();
}

MEM_CLASS_PRE2()
GPUd() bool AliGPUTPCTrackletConstructor::CheckCov(MEM_LG2(AliGPUTPCTrackParam) & tParam)
{
	bool ok = 1;
	const float *c = tParam.Cov();
	for (int i = 0; i < 15; i++) ok = ok && CAMath::Finite(c[i]);
	for (int i = 0; i < 5; i++) ok = ok && CAMath::Finite(tParam.Par()[i]);
	ok = ok && (tParam.X() > 50);
	if (c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0) ok = 0;
	return (ok);
}

MEM_CLASS_PRE23()
GPUd() void AliGPUTPCTrackletConstructor::StoreTracklet(int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
                                                          GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) & s, AliGPUTPCThreadMemory &r, GPUconstant() MEM_LG2(AliGPUTPCTracker) & tracker, MEM_LG3(AliGPUTPCTrackParam) & tParam)
{
	// reconstruction of tracklets, tracklet store step
	if (r.fNHits && (r.fNHits < TRACKLET_SELECTOR_MIN_HITS(tParam.QPt()) ||
	                 !CheckCov(tParam) ||
	                 CAMath::Abs(tParam.GetQPt()) > tracker.Param().rec.MaxTrackQPt))
	{
		r.fNHits = 0;
	}

	/*printf("Tracklet %d: Hits %3d NDF %3d Chi %8.4f Sign %f Cov: %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\n", r.fItr, r.fNHits, tParam.GetNDF(), tParam.GetChi2(), tParam.GetSignCosPhi(),
	  tParam.Cov()[0], tParam.Cov()[1], tParam.Cov()[2], tParam.Cov()[3], tParam.Cov()[4], tParam.Cov()[5], tParam.Cov()[6], tParam.Cov()[7], tParam.Cov()[8], tParam.Cov()[9],
	  tParam.Cov()[10], tParam.Cov()[11], tParam.Cov()[12], tParam.Cov()[13], tParam.Cov()[14]);*/

	GPUglobalref() MEM_GLOBAL(AliGPUTPCTracklet) &tracklet = tracker.Tracklets()[r.fItr];

	tracklet.SetNHits(r.fNHits);
	CADEBUG(printf("    DONE %d hits\n", r.fNHits))

	if (r.fNHits > 0)
	{
		tracklet.SetFirstRow(r.fFirstRow);
		tracklet.SetLastRow(r.fLastRow);
		tracklet.SetParam(tParam.GetParam());
		int w = tracker.CalculateHitWeight(r.fNHits, tParam.GetChi2(), r.fItr);
		tracklet.SetHitWeight(w);
		for (int iRow = r.fFirstRow; iRow <= r.fLastRow; iRow++)
		{
			calink ih = GETRowHit(iRow);
			if (ih != CALINK_INVAL)
			{
				MAKESharedRef(AliGPUTPCRow, row, tracker.Row(iRow), s.fRows[iRow]);
				tracker.MaximizeHitWeight(row, ih, w);
			}
		}
	}
}

MEM_CLASS_PRE2()
GPUd() void AliGPUTPCTrackletConstructor::UpdateTracklet(int /*nBlocks*/, int /*nThreads*/, int /*iBlock*/, int /*iThread*/,
                                                           GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) & s, AliGPUTPCThreadMemory &r, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) & tracker, MEM_LG2(AliGPUTPCTrackParam) & tParam, int iRow)
{
	// reconstruction of tracklets, tracklets update step
#ifndef EXTERN_ROW_HITS
	AliGPUTPCTracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS

	MAKESharedRef(AliGPUTPCRow, row, tracker.Row(iRow), s.fRows[iRow]);

	float y0 = row.Grid().YMin();
	float stepY = row.HstepY();
	float z0 = row.Grid().ZMin() - tParam.ZOffset();
	float stepZ = row.HstepZ();

	if (r.fStage == 0)
	{ // fitting part
		do
		{

			if (iRow < r.fStartRow || r.fCurrIH == CALINK_INVAL) break;
			if ((iRow - r.fStartRow) & 1)
			{
				SETRowHit(iRow, CALINK_INVAL);
				break; // SG!!! - jump over the row
			}

			cahit2 hh = TEXTUREFetchCons(cahit22, gAliTexRefu2, tracker.HitData(row), r.fCurrIH);

			int oldIH = r.fCurrIH;
			r.fCurrIH = TEXTUREFetchCons(calink, gAliTexRefs, tracker.HitLinkUpData(row), r.fCurrIH);

			float x = row.X();
			float y = y0 + hh.x * stepY;
			float z = z0 + hh.y * stepZ;

			if (iRow == r.fStartRow)
			{
				tParam.SetX(x);
				tParam.SetY(y);
				r.fLastY = y;
				if (tracker.Param().ContinuousTracking)
				{
					tParam.SetZ(0.f);
					r.fLastZ = 0.f;
					tParam.SetZOffset(z);
				}
				else
				{
					tParam.SetZ(z);
					r.fLastZ = z;
					tParam.SetZOffset(0.f);
				}
				CADEBUG(printf("Tracklet %5d: FIT INIT  ROW %3d X %8.3f -", r.fItr, iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
			}
			else
			{

				float err2Y, err2Z;
				float dx = x - tParam.X();
				float dy, dz;
				if (r.fNHits >= 10)
				{
					dy = y - tParam.Y();
					dz = z - tParam.Z();
				}
				else
				{
					dy = y - r.fLastY;
					dz = z - r.fLastZ;
				}
				r.fLastY = y;
				r.fLastZ = z;

				float ri = 1.f / CAMath::Sqrt(dx * dx + dy * dy);
				if (iRow == r.fStartRow + 2)
				{ //SG!!! important - thanks to Matthias
					tParam.SetSinPhi(dy * ri);
					tParam.SetSignCosPhi(dx);
					tParam.SetDzDs(dz * ri);
					//std::cout << "Init. errors... " << r.fItr << std::endl;
					tracker.GetErrors2(iRow, tParam, err2Y, err2Z);
					//std::cout << "Init. errors = " << err2Y << " " << err2Z << std::endl;
					tParam.SetCov(0, err2Y);
					tParam.SetCov(2, err2Z);
				}
				float sinPhi, cosPhi;
				if (r.fNHits >= 10 && CAMath::Abs(tParam.SinPhi()) < GPUCA_MAX_SIN_PHI_LOW)
				{
					sinPhi = tParam.SinPhi();
					cosPhi = CAMath::Sqrt(1 - sinPhi * sinPhi);
				}
				else
				{
					sinPhi = dy * ri;
					cosPhi = dx * ri;
				}
				CADEBUG(printf("%14s: FIT TRACK ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
				if (!tParam.TransportToX(x, sinPhi, cosPhi, tracker.Param().ConstBz, CALINK_INVAL))
				{
					SETRowHit(iRow, CALINK_INVAL);
					break;
				}
				CADEBUG(printf("%15s hits %3d: FIT PROP  ROW %3d X %8.3f -", "", r.fNHits, iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
				tracker.GetErrors2(iRow, tParam.GetZ(), sinPhi, tParam.GetDzDs(), err2Y, err2Z);

				if (r.fNHits >= 10)
				{
					const float kFactor = tracker.Param().rec.HitPickUpFactor * tracker.Param().rec.HitPickUpFactor * 3.5f * 3.5f;
					float sy2 = kFactor * (tParam.GetErr2Y() + err2Y);
					float sz2 = kFactor * (tParam.GetErr2Z() + err2Z);
					if (sy2 > 2.f) sy2 = 2.f;
					if (sz2 > 2.f) sz2 = 2.f;
					dy = y - tParam.Y();
					dz = z - tParam.Z();
					if (dy * dy > sy2 || dz * dz > sz2)
					{
						if (++r.fNMissed >= TRACKLET_CONSTRUCTOR_MAX_ROW_GAP_SEED)
						{
							r.fCurrIH = CALINK_INVAL;
						}
						SETRowHit(iRow, CALINK_INVAL);
						break;
					}
				}

				if (!tParam.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI_LOW))
				{
					SETRowHit(iRow, CALINK_INVAL);
					break;
				}
				CADEBUG(printf("%14s: FIT FILT  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
			}
			SETRowHit(iRow, oldIH);
			r.fNHitsEndRow = ++r.fNHits;
			r.fLastRow = iRow;
			r.fEndRow = iRow;
			r.fNMissed = 0;
			break;
		} while (0);

		/*QQQQprintf("Extrapolate Row %d X %f Y %f Z %f SinPhi %f DzDs %f QPt %f", iRow, tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt());
    for (int i = 0;i < 15;i++) printf(" C%d=%6.2f", i, tParam.GetCov(i));
    printf("\n");*/

		if (r.fCurrIH == CALINK_INVAL)
		{
			r.fStage = 1;
			r.fLastY = tParam.Y(); //Store last spatial position here to start inward following from here
			r.fLastZ = tParam.Z();
			if (CAMath::Abs(tParam.SinPhi()) > GPUCA_MAX_SIN_PHI)
			{
				r.fGo = 0;
			}
		}
	}
	else
	{ // forward/backward searching part
		do
		{
			if (r.fStage == 2 && iRow > r.fEndRow) break;
			if (r.fNMissed > TRACKLET_CONSTRUCTOR_MAX_ROW_GAP)
			{
				r.fGo = 0;
				break;
			}

			r.fNMissed++;

			float x = row.X();
			float err2Y, err2Z;
			CADEBUG(printf("%14s: SEA TRACK ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
			if (!tParam.TransportToX(x, tParam.SinPhi(), tParam.GetCosPhi(), tracker.Param().ConstBz, GPUCA_MAX_SIN_PHI_LOW))
			{
				r.fGo = 0;
				SETRowHit(iRow, CALINK_INVAL);
				break;
			}
			CADEBUG(printf("%14s: SEA PROP  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
			if (row.NHits() < 1)
			{
				SETRowHit(iRow, CALINK_INVAL);
				break;
			}

#ifndef GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
			GPUglobalref() const cahit2 *hits = tracker.HitData(row);
			GPUglobalref() const calink *firsthit = tracker.FirstHitInBin(row);
#endif //!GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
			float fY = tParam.GetY();
			float fZ = tParam.GetZ();
			calink best = CALINK_INVAL;

			{ // search for the closest hit
				tracker.GetErrors2(iRow, *((MEM_LG2(AliGPUTPCTrackParam) *) &tParam), err2Y, err2Z);
				const float kFactor = tracker.Param().rec.HitPickUpFactor * tracker.Param().rec.HitPickUpFactor * 3.5f * 3.5f;
				float sy2 = kFactor * (tParam.GetErr2Y() + err2Y);
				float sz2 = kFactor * (tParam.GetErr2Z() + err2Z);
				if (sy2 > 2.f) sy2 = 2.f;
				if (sz2 > 2.f) sz2 = 2.f;

				int bin, ny, nz;
				row.Grid().GetBinArea(fY, fZ + tParam.ZOffset(), 1.5f, 1.5f, bin, ny, nz);
				float ds = 1e6f;

				for (int k = 0; k <= nz; k++)
				{
					int nBinsY = row.Grid().Ny();
					int mybin = bin + k * nBinsY;
					unsigned int hitFst = TEXTUREFetchCons(calink, gAliTexRefu, firsthit, mybin);
					unsigned int hitLst = TEXTUREFetchCons(calink, gAliTexRefu, firsthit, mybin + ny + 1);
					for (unsigned int ih = hitFst; ih < hitLst; ih++)
					{
						cahit2 hh = TEXTUREFetchCons(cahit2, gAliTexRefu2, hits, ih);
						float y = y0 + hh.x * stepY;
						float z = z0 + hh.y * stepZ;
						float dy = y - fY;
						float dz = z - fZ;
						if (dy * dy < sy2 && dz * dz < sz2)
						{
							float dds = GPUCA_Y_FACTOR * fabs(dy) + fabs(dz);
							if (dds < ds)
							{
								ds = dds;
								best = ih;
							}
						}
					}
				}
			} // end of search for the closest hit

			if (best == CALINK_INVAL)
			{
				SETRowHit(iRow, CALINK_INVAL);
				break;
			}

			cahit2 hh = TEXTUREFetchCons(cahit2, gAliTexRefu2, hits, best);
			float y = y0 + hh.x * stepY;
			float z = z0 + hh.y * stepZ;

			CADEBUG(printf("%14s: SEA Hit %5d, Res %f %f\n", "", best, tParam.Y() - y, tParam.Z() - z);)

			calink oldHit = (r.fStage == 2 && iRow >= r.fStartRow) ? GETRowHit(iRow) : CALINK_INVAL;
			if (oldHit != best && !tParam.Filter(y, z, err2Y, err2Z, GPUCA_MAX_SIN_PHI_LOW, oldHit != CALINK_INVAL))
			{
				SETRowHit(iRow, CALINK_INVAL);
				break;
			}
			SETRowHit(iRow, best);
			r.fNHits++;
			r.fNMissed = 0;
			CADEBUG(printf("%5s hits %3d: SEA FILT  ROW %3d X %8.3f -", "", r.fNHits, iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
			if (r.fStage == 1)
				r.fLastRow = iRow;
			else
				r.fFirstRow = iRow;
		} while (0);
	}
}

GPUd() void AliGPUTPCTrackletConstructor::DoTracklet(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) & tracker, GPUsharedref() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) & s, AliGPUTPCThreadMemory &r)
{
	int iRow = 0, iRowEnd = GPUCA_ROW_COUNT;
	MEM_PLAIN(AliGPUTPCTrackParam)
	tParam;
#ifndef EXTERN_ROW_HITS
	AliGPUTPCTracklet &tracklet = tracker.Tracklets()[r.fItr];
#endif //EXTERN_ROW_HITS
	if (r.fGo)
	{
		AliGPUTPCHitId id = tracker.TrackletStartHits()[r.fItr];

		r.fStartRow = r.fEndRow = r.fFirstRow = r.fLastRow = id.RowIndex();
		r.fCurrIH = id.HitIndex();
		r.fNMissed = 0;
		iRow = r.fStartRow;
		AliGPUTPCTrackletConstructor::InitTracklet(tParam);
	}
	r.fStage = 0;
	r.fNHits = 0;
	//if (tracker.Param().ISlice() != 35 && tracker.Param().ISlice() != 34 || r.fItr == CALINK_INVAL) {StoreTracklet( 0, 0, 0, 0, s, r, tracker, tParam );return;}

	for (int k = 0; k < 2; k++)
	{
		for (; iRow != iRowEnd; iRow += r.fStage == 2 ? -1 : 1)
		{
			if (!r.fGo) break;
			UpdateTracklet(0, 0, 0, 0, s, r, tracker, tParam, iRow);
		}
		if (!r.fGo && r.fStage == 2)
		{
			for (; iRow >= r.fStartRow; iRow--)
			{
				SETRowHit(iRow, CALINK_INVAL);
			}
		}
		if (r.fStage == 2)
		{
			StoreTracklet(0, 0, 0, 0, s, r, tracker, tParam);
		}
		else
		{
			r.fNMissed = 0;
			if ((r.fGo = (tParam.TransportToX(tracker.Row(r.fEndRow).X(), tracker.Param().ConstBz, GPUCA_MAX_SIN_PHI) && tParam.Filter(r.fLastY, r.fLastZ, tParam.Err2Y() * 0.5f, tParam.Err2Z() * 0.5f, GPUCA_MAX_SIN_PHI_LOW, true))))
			{
				CADEBUG(printf("%14s: SEA BACK  ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
				float err2Y, err2Z;
				tracker.GetErrors2(r.fEndRow, tParam, err2Y, err2Z);
				if (tParam.GetCov(0) < err2Y) tParam.SetCov(0, err2Y);
				if (tParam.GetCov(2) < err2Z) tParam.SetCov(2, err2Z);
				CADEBUG(printf("%14s: SEA ADJUS ROW %3d X %8.3f -", "", iRow, tParam.X()); for (int i = 0; i < 5; i++) printf(" %8.3f", tParam.Par()[i]); printf(" -"); for (int i = 0; i < 15; i++) printf(" %8.3f", tParam.Cov()[i]); printf("\n");)
			}
			r.fNHits -= r.fNHitsEndRow;
			r.fStage = 2;
			iRow = r.fEndRow;
			iRowEnd = -1;
		}
	}
}

template <> GPUd() void AliGPUTPCTrackletConstructor::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &sMem, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker)
{
	if (get_local_id(0) == 0) sMem.fNTracklets = *tracker.NTracklets();
#ifdef GPUCA_GPUCODE
	for (unsigned int i = get_local_id(0);i < GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(AliGPUTPCRow)) / sizeof(int);i += get_local_size(0))
	{
		reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
	}
#endif
	GPUbarrier();

	AliGPUTPCThreadMemory rMem;
	for (rMem.fItr = get_global_id(0);rMem.fItr < sMem.fNTracklets;rMem.fItr += get_global_size(0))
	{
		rMem.fGo = 1;
		DoTracklet(tracker, sMem, rMem);
	}
}

template <> GPUd() void AliGPUTPCTrackletConstructor::Thread<1>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &sMem, GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker0)
{
#ifdef GPUCA_GPUCODE
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) *pTracker = &tracker0;
	
	int mySlice = get_group_id(0) % GPUCA_NSLICES;
	int currentSlice = -1;

	if (get_local_id(0) == 0)
	{
		sMem.fNextTrackletFirstRun = 1;
	}

	for (unsigned int iSlice = 0;iSlice < GPUCA_NSLICES;iSlice++)
	{
		GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker = pTracker[mySlice];

		AliGPUTPCThreadMemory rMem;

		while ((rMem.fItr = FetchTracklet(tracker, sMem)) != -2)
		{
			if (rMem.fItr >= 0 && get_local_id(0) < GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR)
			{
				rMem.fItr += get_local_id(0);
			}
			else
			{
				rMem.fItr = -1;
			}

			if (mySlice != currentSlice)
			{
				if (get_local_id(0) == 0)
				{
					sMem.fNTracklets = *tracker.NTracklets();
				}

				for (unsigned int i = get_local_id(0);i < GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(AliGPUTPCRow)) / sizeof(int);i += get_local_size(0))
				{
					reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
				}
				GPUbarrier();
				currentSlice = mySlice;
			}

			if (rMem.fItr >= 0 && rMem.fItr < sMem.fNTracklets)
			{
				rMem.fGo = true;
				DoTracklet(tracker, sMem, rMem);
			}
		}
		if (++mySlice >= GPUCA_NSLICES) mySlice = 0;
	}
#else
	throw std::logic_error("Not supported on CPU");
#endif
}

#ifdef GPUCA_GPUCODE

GPUdi() int AliGPUTPCTrackletConstructor::FetchTracklet(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &sMem)
{
	const int nativeslice = get_group_id(0) % GPUCA_NSLICES;
	const int nTracklets = *tracker.NTracklets();
	GPUbarrier();
	if (get_local_id(0) == 0)
	{
		if (sMem.fNextTrackletFirstRun == 1)
		{
			sMem.fNextTrackletFirst = (get_group_id(0) - nativeslice) / GPUCA_NSLICES * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;
			sMem.fNextTrackletFirstRun = 0;
		}
		else
		{
			if (tracker.GPUParameters()->fNextTracklet < nTracklets)
			{
				const int firstTracklet = CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR);
				if (firstTracklet < nTracklets) sMem.fNextTrackletFirst = firstTracklet;
				else sMem.fNextTrackletFirst = -2;
			}
			else
			{
				sMem.fNextTrackletFirst = -2;
			}
		}
	}
	GPUbarrier();
	return (sMem.fNextTrackletFirst);
}

#else //GPUCA_GPUCODE

int AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGlobalTracking(AliGPUTPCTracker &tracker, AliGPUTPCTrackParam &tParam, int row, int increment, int iTracklet)
{
	AliGPUTPCThreadMemory rMem;
	GPUshared() AliGPUTPCSharedMemory sMem;
	sMem.fNTracklets = *tracker.NTracklets();
	rMem.fItr = iTracklet;
	rMem.fStage = 3;
	rMem.fNHits = rMem.fNMissed = 0;
	rMem.fGo = 1;
	while (rMem.fGo && row >= 0 && row < GPUCA_ROW_COUNT)
	{
		UpdateTracklet(1, 1, 0, 0, sMem, rMem, tracker, tParam, row);
		row += increment;
	}
	if (!CheckCov(tParam)) rMem.fNHits = 0;
	return (rMem.fNHits);
}

#endif //GPUCA_GPUCODE
