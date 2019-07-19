// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUMemorySizeScalers.h
/// \author David Rohr

#ifndef O2_GPU_GPUMEMORYSIZESCALERS_H
#define O2_GPU_GPUMEMORYSIZESCALERS_H

#include "GPUDef.h"

namespace GPUCA_NAMESPACE::gpu
{

struct GPUMemorySizeScalers {
  // Input sizes
  double nTPCHits = 0;
  double nTRDTracklets = 0;
  double nITSTracks = 0;

  // Offset
  double offset = 1000.;

  // Scaling Factors
  double tpcTrackletsPerHit = 0.08;
  double tpcSectorTracksPerHit = 0.02;
  double tpcSectorTrackHitsPerHit = 0.8f;
  double tpcTracksPerHit = 0.012;
  double tpcTrackHitsPerHit = 0.7;

  double NTPCTracklets(double tpcHits) { return offset + tpcHits * tpcTrackletsPerHit; }
  double NTPCSectorTracks(double tpcHits) { return offset + tpcHits * tpcSectorTracksPerHit; }
  double NTPCSectorTrackHits(double tpcHits) { return offset + tpcHits * tpcSectorTrackHitsPerHit; }
  double NTPCTracks(double tpcHits) { return offset + tpcHits * tpcTracksPerHit; }
  double NTPCTrackHits(double tpcHits) { return offset + tpcHits * tpcTrackHitsPerHit; }
  double NTPCMaxRowStartHits(double tpcHits) { return offset + NTPCTracklets(tpcHits) / GPUCA_ROW_COUNT * 3.; }
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
