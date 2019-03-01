// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCRow.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCRow.h"

#if !defined(GPUCA_GPUCODE)
GPUTPCRow::GPUTPCRow() : fNHits(0), fX(0), fMaxY(0), fGrid(),
	fHy0(0), fHz0(0), fHstepY(0), fHstepZ(0), fHstepYi(0), fHstepZi(0),
	fFullSize(0), fHitNumberOffset(0), fFirstHitInBinOffset(0)
{
	// dummy constructor
}

#endif
