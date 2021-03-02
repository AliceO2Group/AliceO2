// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUMemorySizeScalers.cxx
/// \author David Rohr

#include "GPUMemorySizeScalers.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUMemorySizeScalers::rescaleMaxMem(size_t newAvailableMemory)
{
  GPUMemorySizeScalers tmp;
  tpcMaxPeaks = (double)tmp.tpcMaxPeaks * newAvailableMemory / tmp.availableMemory;
  tpcMaxClusters = (double)tmp.tpcMaxClusters * newAvailableMemory / tmp.availableMemory;
  tpcMaxStartHits = (double)tmp.tpcMaxStartHits * newAvailableMemory / tmp.availableMemory;
  tpcMaxRowStartHits = (double)tmp.tpcMaxRowStartHits * newAvailableMemory / tmp.availableMemory;
  tpcMaxTracklets = (double)tmp.tpcMaxTracklets * newAvailableMemory / tmp.availableMemory;
  tpcMaxTrackletHits = (double)tmp.tpcMaxTrackletHits * newAvailableMemory / tmp.availableMemory;
  tpcMaxSectorTracks = (double)tmp.tpcMaxSectorTracks * newAvailableMemory / tmp.availableMemory;
  tpcMaxSectorTrackHits = (double)tmp.tpcMaxSectorTrackHits * newAvailableMemory / tmp.availableMemory;
  tpcMaxMergedTracks = (double)tmp.tpcMaxMergedTracks * newAvailableMemory / tmp.availableMemory;
  tpcMaxMergedTrackHits = (double)tmp.tpcMaxMergedTrackHits * newAvailableMemory / tmp.availableMemory;
  availableMemory = newAvailableMemory;
}
