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

/// \file GPUTPCGMMergerRebuildInternal.h
/// \author David Rohr

#ifndef GPUTPCGMMERGERREBUILDINTERNAL_H
#define GPUTPCGMMERGERREBUILDINTERNAL_H

#include "GPUTPCGMMerger.h"
#include "GPUCommonMath.h"

namespace o2::gpu
{

struct GPUTPCGMMerger::trackRebuildHelperCandidate { // Per track, row and candidate, the row is implicit
  uint16_t idHigh, idLow;                            // 2 x 16 bit instead of uint32_t for a size of 6 bytes without packing, high part first. Combined value: 0: empty (in the first candidate: row not searched), 1: row searched but no candidate, otherwise (sector << 24) | (id + 2), with id the index of the cluster in its sector row (below 2^24 - 2)
  CAMath::half_t error;
  GPUdi() uint32_t getValue() const { return ((uint32_t)idHigh << 16) | idLow; }
  GPUdi() void setValue(uint32_t v)
  {
    idHigh = v >> 16;
    idLow = v & 0xFFFF;
  }
  GPUdi() bool isEmpty() const { return getValue() == 0; }
  GPUdi() bool hasCluster() const { return getValue() >= 2; }
  GPUdi() void setNoCandidate() { setValue(1); }
  GPUdi() uint32_t getId() const { return (getValue() & 0xFFFFFF) - 2; } // Only with hasCluster(), convert with GPUTPCGMMerger::CandidateId / CandidateClusterId
  GPUdi() uint8_t getSector() const { return idHigh >> 8; }              // Only with hasCluster()
  GPUdi() void set(uint32_t id, uint8_t sector, CAMath::half_t e)
  {
    setValue(((uint32_t)sector << 24) | (id + 2));
    error = e;
  }
};
static_assert(sizeof(GPUTPCGMMerger::trackRebuildHelperCandidate) == 6);

struct GPUTPCGMMerger::trackRebuildHelper { // Per track
  uint8_t reverse;
  uint8_t highInclRowLow;
  uint8_t highInclRowHigh;
};

struct GPUTPCGMMerger::trackRebuildHelperRow { // Per track and row: weight (lower 28 bits) and best (upper 4 bits)
  uint32_t data;
  enum : uint32_t { bestRowRemoved = 14,                  // Row removed (extrapolation reverted)
                    bestNoCandidate = 15 };               // No acceptable candidate
  GPUdi() uint32_t getBest() const { return data >> 28; } // 0: undecided, 1 - 13: index of selected candidate + 1, or bestRowRemoved / bestNoCandidate
  GPUdi() void setBest(uint32_t best) { data = (data & 0x0FFFFFFF) | (best << 28); }
  GPUdi() int32_t getWeight() const { return data & 0x0FFFFFFF; }
  GPUdi() int32_t setWeight(float w) { return (data = (data & 0xF0000000) | (w > 0.5f ? (w < 2147483648.f ? CAMath::Float2UIntReint(w) - 0x3F000000 : 0x0FFFFFFF) : 0)) & 0x0FFFFFFF; }
};

} // namespace o2::gpu

#endif
