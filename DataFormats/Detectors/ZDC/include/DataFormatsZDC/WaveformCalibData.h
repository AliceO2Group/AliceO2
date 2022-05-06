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

#ifndef ZDC_WAVEFORMCALIB_DATA_H_
#define ZDC_WAVEFORMCALIB_DATA_H_

#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include <array>
#include <Rtypes.h>

/// \file WaveformCalibData.h
/// \brief Waveform calibration intermediate data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct WaveformCalibData {
  static constexpr int NH = WaveformCalibConfig::NH;
  uint64_t mCTimeBeg = 0; /// Time of processed time frame
  uint64_t mCTimeEnd = 0; /// Time of processed time frame
  std::array<float, WaveformCalibConfig::NBT * TSN> mWave[NH];
  mEntries[NH] = {0};
  WaveformCalibData& operator+=(const WaveformCalibData& other);
  int getEntries(int ih) const;
  void print() const;
  void setCreationTime(uint64_t ctime);
  ClassDefNV(WaveformCalibData, 1);
};

} // namespace zdc
} // namespace o2

#endif
