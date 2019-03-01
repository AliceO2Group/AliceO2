// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCHitArea.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCHITAREA_H
#define GPUTPCHITAREA_H

#include "GPUTPCDef.h"

class GPUTPCHit;
class GPUTPCGrid;
MEM_CLASS_PRE()
class GPUTPCTracker;
MEM_CLASS_PRE()
class GPUTPCRow;
MEM_CLASS_PRE()
class GPUTPCSliceData;

/**
 * @class GPUTPCHitArea
 *
 * This class is used to _iterate_ over the hit data via GetNext
 */
class GPUTPCHitArea
{
  public:
	MEM_TEMPLATE()
	GPUd() void Init(const MEM_TYPE(GPUTPCRow) & row, GPUglobalref() const MEM_GLOBAL(GPUTPCSliceData) & slice, float y, float z, float dy, float dz);

	/**
     * look up the next hit in the requested area.
     * Sets h to the coordinates and returns the index for the hit data
     */
	MEM_TEMPLATE()
	GPUd() int GetNext(GPUconstantref() const MEM_CONSTANT(GPUTPCTracker) & tracker, const MEM_TYPE(GPUTPCRow) & row,
	                   GPUglobalref() const MEM_GLOBAL(GPUTPCSliceData) & slice, GPUTPCHit *h);

	float Y() const { return fY; }
	float Z() const { return fZ; }
	float MinZ() const { return fMinZ; }
	float MaxZ() const { return fMaxZ; }
	float MinY() const { return fMinY; }
	float MaxY() const { return fMaxY; }
	int BZmax() const { return fBZmax; }
	int BDY() const { return fBDY; }
	int IndYmin() const { return fIndYmin; }
	int Iz() const { return fIz; }
	int HitYfst() const { return fHitYfst; }
	int HitYlst() const { return fHitYlst; }
	int Ih() const { return fIh; }
	int Ny() const { return fNy; }
	int HitOffset() const { return fHitOffset; }

  protected:
	float fY;       // search coordinates
	float fZ;       // search coordinates
	float fMinZ;    // search coordinates
	float fMaxZ;    // search coordinates
	float fMinY;    // search coordinates
	float fMaxY;    // search coordinates
	int fBZmax;     // maximal Z bin index
	int fBDY;       // Y distance of bin indexes
	int fIndYmin;   // minimum index for
	int fIz;        // current Z bin index (incremented while iterating)
	int fHitYfst;   //
	int fHitYlst;   //
	int fIh;        // some XXX index in the hit data
	int fNy;        // Number of bins in Y direction
	int fHitOffset; // global hit offset XXX what's that?
};

#endif //GPUTPCHITAREA_H
