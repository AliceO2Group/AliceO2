// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTracklet.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKLET_H
#define GPUTPCTRACKLET_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"
#include "GPUTPCGPUConfig.h"

/**
 * @class GPUTPCTracklet
 *
 * The class describes the reconstructed TPC track candidate.
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 */
MEM_CLASS_PRE()
class GPUTPCTracklet
{
  public:
#if !defined(GPUCA_GPUCODE)
	GPUTPCTracklet() : fNHits(0), fFirstRow(0), fLastRow(0), fParam(), fHitWeight(0){};
#endif //!GPUCA_GPUCODE

	GPUhd() int NHits() const
	{
		return fNHits;
	}
	GPUhd() int FirstRow() const { return fFirstRow; }
	GPUhd() int LastRow() const { return fLastRow; }
	GPUhd() int HitWeight() const { return fHitWeight; }
	GPUhd() MakeType(const MEM_LG(GPUTPCBaseTrackParam) &) Param() const { return fParam; }
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
	GPUhd() void SetParam(const MEM_LG2(GPUTPCBaseTrackParam) & v) { fParam = reinterpret_cast<const MEM_LG(GPUTPCBaseTrackParam) &>(v); }
	GPUhd() void SetHitWeight(const int w) { fHitWeight = w; }

  private:
	int fNHits;    // N hits
	int fFirstRow; // first TPC row
	int fLastRow;  // last TPC row
	MEM_LG(GPUTPCBaseTrackParam)
	fParam; // tracklet parameters
#ifndef EXTERN_ROW_HITS
	calink fRowHits[GPUCA_ROW_COUNT + 1]; // hit index for each TPC row
#endif                                    //EXTERN_ROW_HITS
	int fHitWeight;                       //Hit Weight of Tracklet
};

#endif //GPUTPCTRACKLET_H
