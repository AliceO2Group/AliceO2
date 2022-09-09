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

#ifndef ZDC_NOISECALIB_PARAM_H
#define ZDC_NOISECALIB_PARAM_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <vector>

/// \file NoiseParam.h
/// \brief Noise calibration data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct NoiseParam {
  NoiseParam() = default;
  float noise[NChannels] = {0};      // RMS of noise
  uint64_t entries[NChannels] = {0}; // Number of processed entries
  void setCalib(uint32_t ich, float val);
  float getCalib(uint32_t ich) const;
  void print() const;

  ClassDefNV(NoiseParam, 1);
};

} // namespace zdc
} // namespace o2

#endif
