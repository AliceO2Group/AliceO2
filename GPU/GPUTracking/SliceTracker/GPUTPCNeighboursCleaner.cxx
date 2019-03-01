// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNeighboursCleaner.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCNeighboursCleaner.h"
#include "GPUTPCTracker.h"
#include "GPUCommonMath.h"

template <> GPUd() void GPUTPCNeighboursCleaner::Thread<0>(int /*nBlocks*/, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUTPCSharedMemory) &s, workerType &tracker)
{
	// *
	// * kill link to the neighbour if the neighbour is not pointed to the cluster
	// *

	if (iThread == 0)
	{
		s.fIRow = iBlock + 2;
		if (s.fIRow <= GPUCA_ROW_COUNT - 3)
		{
			s.fIRowUp = s.fIRow + 2;
			s.fIRowDn = s.fIRow - 2;
			s.fNHits = tracker.Row(s.fIRow).NHits();
		}
	}
    GPUbarrier();
    
	if (s.fIRow <= GPUCA_ROW_COUNT - 3)
	{
#ifdef GPUCA_GPUCODE
		int Up = s.fIRowUp;
		int Dn = s.fIRowDn;
		GPUglobalref() const MEM_GLOBAL(GPUTPCRow) &row = tracker.Row(s.fIRow);
		GPUglobalref() const MEM_GLOBAL(GPUTPCRow) &rowUp = tracker.Row(Up);
		GPUglobalref() const MEM_GLOBAL(GPUTPCRow) &rowDn = tracker.Row(Dn);
#else
		const GPUTPCRow &row = tracker.Row(s.fIRow);
		const GPUTPCRow &rowUp = tracker.Row(s.fIRowUp);
		const GPUTPCRow &rowDn = tracker.Row(s.fIRowDn);
#endif

		// - look at up link, if it's valid but the down link in the row above doesn't link to us remove
		//   the link
		// - look at down link, if it's valid but the up link in the row below doesn't link to us remove
		//   the link
		for (int ih = iThread; ih < s.fNHits; ih += nThreads)
		{
			calink up = tracker.HitLinkUpData(row, ih);
			if (up != CALINK_INVAL)
			{
				calink upDn = tracker.HitLinkDownData(rowUp, up);
				if ((upDn != (calink) ih)) tracker.SetHitLinkUpData(row, ih, CALINK_INVAL);
			}
			calink dn = tracker.HitLinkDownData(row, ih);
			if (dn != CALINK_INVAL)
			{
				calink dnUp = tracker.HitLinkUpData(rowDn, dn);
				if (dnUp != (calink) ih) tracker.SetHitLinkDownData(row, ih, CALINK_INVAL);
			}
		}
	}
}
