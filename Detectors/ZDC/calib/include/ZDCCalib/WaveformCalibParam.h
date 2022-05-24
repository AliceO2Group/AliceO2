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

#ifndef ZDC_WAVEFORMCALIB_PARAM_H
#define ZDC_WAVEFORMCALIB_PARAM_H

#include "GPUCommonRtypes.h"
#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibData.h"
#include <array>
#include <vector>

/// \file WaveformCalibParam.h
/// \brief Waveform calibration data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct WaveformCalibChParam {
  std::vector<float> shape;
  int ampMinID = 0;
  void print() const;
  ClassDefNV(WaveformCalibChParam, 1);
};

struct WaveformCalibParam {

  std::array<WaveformCalibChParam, NChannels> channels; // configuration per channel

  void assign(const WaveformCalibData &data);

  void print() const;

  ClassDefNV(WaveformCalibParam, 1);
};

} // namespace zdc
} // namespace o2

#endif
