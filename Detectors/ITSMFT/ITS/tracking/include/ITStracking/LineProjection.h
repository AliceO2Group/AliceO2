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

/// \file LineProjection.h
/// \brief small types shared by the host and device seeding vertexers, describing lines
///        projected onto the beam line: their time interval and their z-window bounds.

#ifndef O2_ITS_TRACKING_LINE_PROJECTION_H_
#define O2_ITS_TRACKING_LINE_PROJECTION_H_

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "DataFormatsITS/TimeEstBC.h"
#include <cstdint>
#include <vector>

namespace o2::its
{

// Half-open [lo, hi) range of sorted-line slots falling inside one z-window.
struct LineWindow {
  int lo{0};
  int hi{0};
};

// Every line gets its own z window: from the number n of time-compatible lines within +-hMax,
// h = hMax (n/4)^-alpha, clamped to [hMin, hMax]. Peaks compare count/h.
constexpr int AdaptiveZWindowNRef = 4;         // up to this many lines the window keeps its full width hMax
constexpr float AdaptiveSeedWindowScale = 2.f; // the seed fit uses the lines within +-2h (at most +-hMax)

GPUhdi() float computeAdaptiveZWindow(const int n, const float alpha, const float hMin, const float hMax)
{
  if (n <= AdaptiveZWindowNRef) {
    return hMax;
  }
  const float h = hMax * o2::gpu::CAMath::Pow(static_cast<float>(n) / AdaptiveZWindowNRef, -alpha);
  return o2::gpu::CAMath::Min(o2::gpu::CAMath::Max(h, hMin), hMax);
}

// True if line j (density dj over half-width hj, at zj) beats line i: higher dj/hj, ties to the smaller z
GPUhdi() bool isDenser(const int dj, const float hj, const float zj, const int di, const float hi, const float zi)
{
  const float a = dj * hi;
  const float b = di * hj;
  return a > b || (a == b && zj < zi);
}

// Per-line quality estimators carried alongside the projected line.
struct LineQuality {
  float chi2{-1.f};
  float pt{-1.f};
};

// Host mirror of the per-peak inputs needed to recompute, off-device, which lines a vertex candidate accepted.
// MC-only: filled just to feed the majority-vote labels.
struct PeakMembershipHost {
  std::vector<int> peakLineIdx;  // per compacted peak: the sorted line slot it came from
  std::vector<LineWindow> win;   // per sorted line: [lo, hi) z-window bounds
  std::vector<TimeEstBC> times;  // per sorted line: its time interval
  std::vector<int> sortedToLine; // sorted slot -> global line index

  void clear()
  {
    peakLineIdx.clear();
    win.clear();
    times.clear();
    sortedToLine.clear();
  }
};

namespace gpu
{

// Spelled gpu::LineWindow at the device call sites; qualified lookup does not reach the
// enclosing namespace, so the alias has to be declared here.
using LineWindow = o2::its::LineWindow;

// Device SoA view of the lines projected onto the beam line.
struct LineProjSoA {
  float* z{nullptr};              // projected z at the beamline; sort key and binary-search key, kept dense
  o2::its::TimeEstBC* t{nullptr}; // time interval of the line
  int* idx{nullptr};              // sorted slot -> original line index
  int* rof{nullptr};              // ROF of the line
};

// One vertex candidate produced by the device fit, before the host emit step
struct VertexCand {
  float x, y, z;
  float rms2[6];
  float avgDist2;
  int nGood;
  float seed[3];
  o2::its::TimeEstBC time;
  int size;
  uint8_t ok;   // 1 if the candidate passed the fit cuts
  uint8_t keep; // 1 if it survived duplicate suppression (subset of ok)
};

} // namespace gpu

} // namespace o2::its

#endif
