// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_STRANGENESS_TRACKING_ITSCLUSTERBINRANGE_H
#define O2_STRANGENESS_TRACKING_ITSCLUSTERBINRANGE_H

#include <algorithm>
#include <numbers>

namespace o2::strangeness_tracking::clustertracking
{
// An inclusive spatial-bin rectangle, traversing each periodic phi bin once.
struct BinRange {
  int firstZ = -1, lastZ = -1, firstPhi = 0, lastPhi = -1, nPhi = 0;
  bool empty() const { return firstZ < 0 || firstZ > lastZ || nPhi <= 0; }
  int phiCount() const { return empty() ? 0 : (lastPhi - firstPhi + nPhi) % nPhi + 1; }
  int phiBin(int offset) const { return (firstPhi + offset) % nPhi; }
};

inline BinRange makeBinRange(int firstZ, int firstPhi, int lastZ, int lastPhi, int nZ, int nPhi, float phiHalfWidth)
{
  // Preserve getBinsRect's empty/out-of-layer sentinel before clamping.
  if (firstZ < 0 || lastZ < 0 || nZ <= 0 || nPhi <= 0) {
    return {};
  }
  firstZ = std::clamp(firstZ, 0, nZ - 1);
  lastZ = std::clamp(lastZ, 0, nZ - 1);
  if (firstZ > lastZ) {
    return {};
  }
  // Float normalization can round a tiny negative angle to exactly 2*pi,
  // yielding bin nPhi. Keep coarse support conservatively; the caller's
  // exact spatial/fit gate remains authoritative.
  // Coincident endpoint bins can also enclose almost a full turn. In that
  // case the periodic index difference loses the winding of the interval.
  constexpr float pi = std::numbers::pi_v<float>;
  const bool fullPhi = phiHalfWidth >= pi || (firstPhi == lastPhi && phiHalfWidth >= 0.5f * pi);
  if (fullPhi || firstPhi < 0 || firstPhi >= nPhi || lastPhi < 0 || lastPhi >= nPhi) {
    firstPhi = 0;
    lastPhi = nPhi - 1;
  }
  return {firstZ, lastZ, firstPhi, lastPhi, nPhi};
}
} // namespace o2::strangeness_tracking::clustertracking
#endif
