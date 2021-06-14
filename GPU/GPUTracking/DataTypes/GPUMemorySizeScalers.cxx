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
#include "GPULogging.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUMemorySizeScalers::rescaleMaxMem(size_t newAvailableMemory)
{
  GPUMemorySizeScalers tmp;
  double scaleFactor = (double)newAvailableMemory / tmp.availableMemory;
  GPUInfo("Rescaling buffer size limits from %lu to %lu bytes of memory (factor %f)", tmp.availableMemory, newAvailableMemory, scaleFactor);
  tpcMaxPeaks = (double)tmp.tpcMaxPeaks * scaleFactor;
  tpcMaxClusters = (double)tmp.tpcMaxClusters * scaleFactor;
  tpcMaxStartHits = (double)tmp.tpcMaxStartHits * scaleFactor;
  tpcMaxRowStartHits = (double)tmp.tpcMaxRowStartHits * scaleFactor;
  tpcMaxTracklets = (double)tmp.tpcMaxTracklets * scaleFactor;
  tpcMaxTrackletHits = (double)tmp.tpcMaxTrackletHits * scaleFactor;
  tpcMaxSectorTracks = (double)tmp.tpcMaxSectorTracks * scaleFactor;
  tpcMaxSectorTrackHits = (double)tmp.tpcMaxSectorTrackHits * scaleFactor;
  tpcMaxMergedTracks = (double)tmp.tpcMaxMergedTracks * scaleFactor;
  tpcMaxMergedTrackHits = (double)tmp.tpcMaxMergedTrackHits * scaleFactor;
  availableMemory = newAvailableMemory;
}
