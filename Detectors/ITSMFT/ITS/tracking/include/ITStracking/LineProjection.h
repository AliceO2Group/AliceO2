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
  uint8_t fine;
};

} // namespace gpu

} // namespace o2::its

#endif
