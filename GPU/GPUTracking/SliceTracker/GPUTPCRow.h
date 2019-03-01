// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCRow.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCROW_H
#define GPUTPCROW_H

#include "GPUTPCDef.h"
#include "GPUTPCGrid.h"

/**
 * @class GPUTPCRow
 *
 * The GPUTPCRow class is a hit and cells container for one TPC row.
 * It is the internal class of the GPUTPCTracker algorithm.
 *
 */
MEM_CLASS_PRE()
class GPUTPCRow
{
	MEM_CLASS_PRE2()
	friend class GPUTPCSliceData;

  public:
#if !defined(GPUCA_GPUCODE)
	GPUTPCRow();
#endif //!GPUCA_GPUCODE

	GPUhd() int NHits() const { return fNHits; }
	GPUhd() float X() const { return fX; }
	GPUhd() float MaxY() const { return fMaxY; }
	GPUhd() MakeType(const GPUTPCGrid &) Grid() const { return fGrid; }

	GPUhd() float Hy0() const { return fHy0; }
	GPUhd() float Hz0() const { return fHz0; }
	GPUhd() float HstepY() const { return fHstepY; }
	GPUhd() float HstepZ() const { return fHstepZ; }
	GPUhd() float HstepYi() const { return fHstepYi; }
	GPUhd() float HstepZi() const { return fHstepZi; }
	GPUhd() int FullSize() const { return fFullSize; }
	GPUhd() int HitNumberOffset() const { return fHitNumberOffset; }
	GPUhd() unsigned int FirstHitInBinOffset() const { return fFirstHitInBinOffset; }

  private:
	int fNHits;            // number of hits
	float fX;              // X coordinate of the row
	float fMaxY;           // maximal Y coordinate of the row
	GPUTPCGrid fGrid; // grid of hits

	// hit packing:
	float fHy0;     // offset
	float fHz0;     // offset
	float fHstepY;  // step size
	float fHstepZ;  // step size
	float fHstepYi; // inverse step size
	float fHstepZi; // inverse step size

	int fFullSize;        // size of this row in Tracker::fRowData
	int fHitNumberOffset; // index of the first hit in the hit array, used as
	// offset in GPUTPCSliceData::LinkUp/DownData/HitDataY/...
	unsigned int fFirstHitInBinOffset; // offset in Tracker::fRowData to find the FirstHitInBin
};

#endif //GPUTPCROW_H
