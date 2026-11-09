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

/// \file GPUDisplayTypes.h
/// \author David Rohr

#ifndef GPUDISPLAYTYPES_H
#define GPUDISPLAYTYPES_H

#include "utils/vecpod.h"
#include <atomic>
#include <cstdint>
#include <ostream>

namespace o2::gpu
{
struct GPUDisplayMultiSliderValue { // Value of a Qt GUI multi slider (GUI_MULTI_SLIDER): positions in the order of the display, selected one (-1: none), end of the range, progress
  vecpod<float> values;
  int16_t selected = -1;
  uint16_t progress = 0; // Fraction (of 65535) of the last position highlighted, e.g. a running animation; small types: GPUDisplayGUIValue must not grow
  float max = 0.f;
  bool operator==(const GPUDisplayMultiSliderValue&) const = default;
  friend std::ostream& operator<<(std::ostream& out, const GPUDisplayMultiSliderValue& v) // For the qconfig printout
  {
    for (float x : v.values) {
      out << x << " ";
    }
    return out << "(selected " << v.selected << ")";
  }
};

template <class T>
struct GPUDisplayCopyableAtomic : std::atomic<T> { // std::atomic deletes the copy, here it copies the value
  using std::atomic<T>::atomic;
  GPUDisplayCopyableAtomic(const GPUDisplayCopyableAtomic& o) : std::atomic<T>(o.load()) {}
  GPUDisplayCopyableAtomic& operator=(const GPUDisplayCopyableAtomic& o)
  {
    this->store(o.load());
    return *this;
  }
};
} // namespace o2::gpu

#endif
