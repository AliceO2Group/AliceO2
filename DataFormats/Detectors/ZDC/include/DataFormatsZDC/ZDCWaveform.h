// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ZDC_WAVEFORM_H
#define ZDC_WAVEFORM_H

#include "ZDCBase/Constants.h"
#include <array>
#include <cmath>
#include <Rtypes.h>

/// \file ZDCWaveform.h
/// \brief Container class to store the interpolated waveform of the ZDC
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct ZDCWaveform {

  uint8_t sig = 0; /// Signal id
  float inter[NTimeBinsPerBC * TSN] = {0};

  ZDCWaveform() = default;

  ZDCWaveform(uint8_t mych, float* waveform)
  {
    set(mych, waveform);
  }

  inline void set(uint8_t ch, float* waveform)
  {
    sig = ch;
    for (int i = 0; i < (NTimeBinsPerBC * TSN); i++) {
      inter[i] = waveform[i];
    }
  }

  const float* waveform() const
  {
    return inter;
  }

  int ch() const
  {
    return sig;
  }

  void print() const;

  ClassDefNV(ZDCWaveform, 1);
};
} // namespace zdc
} // namespace o2

#endif
