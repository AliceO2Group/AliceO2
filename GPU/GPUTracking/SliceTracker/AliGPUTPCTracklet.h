//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKLET_H
#define ALIHLTTPCCATRACKLET_H

#include "AliGPUTPCBaseTrackParam.h"
#include "AliGPUTPCDef.h"
#include "AliGPUTPCGPUConfig.h"

/**
 * @class ALIHLTTPCCATracklet
 *
 * The class describes the reconstructed TPC track candidate.
 * The class is dedicated for internal use by the AliGPUTPCTracker algorithm.
 */
MEM_CLASS_PRE()
class AliGPUTPCTracklet
{
  public:
#if !defined(GPUCA_GPUCODE)
	AliGPUTPCTracklet() : fNHits(0), fFirstRow(0), fLastRow(0), fParam(), fHitWeight(0){};
#endif //!GPUCA_GPUCODE

	GPUhd() int NHits() const
	{
		return fNHits;
	}
	GPUhd() int FirstRow() const { return fFirstRow; }
	GPUhd() int LastRow() const { return fLastRow; }
	GPUhd() int HitWeight() const { return fHitWeight; }
	GPUhd() MakeType(const MEM_LG(AliGPUTPCBaseTrackParam) &) Param() const { return fParam; }
#ifndef EXTERN_ROW_HITS
	GPUhd() int RowHit(int i) const
	{
		return fRowHits[i];
	}
	GPUhd() const int *RowHits() const { return (fRowHits); }
	GPUhd() void SetRowHit(int irow, int ih) { fRowHits[irow] = ih; }
#endif //EXTERN_ROW_HITS

	GPUhd() void SetNHits(int v)
	{
		fNHits = v;
	}
	GPUhd() void SetFirstRow(int v) { fFirstRow = v; }
	GPUhd() void SetLastRow(int v) { fLastRow = v; }
	MEM_CLASS_PRE2()
	GPUhd() void SetParam(const MEM_LG2(AliGPUTPCBaseTrackParam) & v) { fParam = reinterpret_cast<const MEM_LG(AliGPUTPCBaseTrackParam) &>(v); }
	GPUhd() void SetHitWeight(const int w) { fHitWeight = w; }

  private:
	int fNHits;    // N hits
	int fFirstRow; // first TPC row
	int fLastRow;  // last TPC row
	MEM_LG(AliGPUTPCBaseTrackParam)
	fParam; // tracklet parameters
#ifndef EXTERN_ROW_HITS
	calink fRowHits[GPUCA_ROW_COUNT + 1]; // hit index for each TPC row
#endif                                    //EXTERN_ROW_HITS
	int fHitWeight;                       //Hit Weight of Tracklet
};

#endif //ALIHLTTPCCATRACKLET_H
