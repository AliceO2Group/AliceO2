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

#ifndef ZDC_WAVEFORMCALIB_DATA_H
#define ZDC_WAVEFORMCALIB_DATA_H

#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include <array>

/// \file WaveformCalibData.h
/// \brief Waveform calibration intermediate data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct WaveformCalibChData {
  static constexpr int NBT = WaveformCalib_NBT;
  static constexpr int NW = WaveformCalib_NW;

  int mFirstValid = 0;               /// First bin with valid data
  int mLastValid = 0;                /// Last bin with valid data
  uint32_t mEntries = 0;             /// Number of waveforms added
  std::array<float, NW> mData = {0}; /// Waveform

  WaveformCalibChData& operator+=(const WaveformCalibChData& other);
  int getEntries() const;
  int getFirstValid() const;
  int getLastValid() const;
  void clear();
  void setN(int n);
  ClassDefNV(WaveformCalibChData, 1);
};

struct WaveformCalibData {
  static constexpr int NBB = WaveformCalib_NBB;
  static constexpr int NBA = WaveformCalib_NBA;
  static constexpr int NBT = WaveformCalib_NBT;
  static constexpr int NW = WaveformCalib_NW;

  uint64_t mCTimeBeg = 0; /// Time of processed time frame
  uint64_t mCTimeEnd = 0; /// Time of processed time frame
  int mN = 0;             /// Number of bunches in waveform
  int mPeak = 0;          /// Peak position

  std::array<WaveformCalibChData, NChannels> mWave;
  WaveformCalibData& operator+=(const WaveformCalibData& other);
  inline void setFirstValid(int isig, int ipos)
  {
    if (ipos > mWave[isig].mFirstValid) {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
      printf("WaveformCalibChData::%s isig=%-2d mFirstValid %5d -> %5d\n", __func__, isig, mWave[isig].mFirstValid, ipos);
#endif
      mWave[isig].mFirstValid = ipos;
    }
  }
  inline void setLastValid(int isig, int ipos)
  {
    if (ipos < mWave[isig].mLastValid) {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
      printf("WaveformCalibChData::%s isig=%-2d mLastValid %5d -> %5d\n", __func__, isig, mWave[isig].mLastValid, ipos);
#endif
      mWave[isig].mLastValid = ipos;
    }
  }
  inline void addEntry(int isig)
  {
    mWave[isig].mEntries++;
  }
  int getEntries(int is) const;
  int getFirstValid(int is) const;
  int getLastValid(int is) const;
  void print() const;
  void clear();
  void clearWaveforms();
  void setCreationTime(uint64_t ctime);
  void setN(int n);
  int saveDebugHistos(const std::string fn);
  ClassDefNV(WaveformCalibData, 1);
};

} // namespace zdc
} // namespace o2

#endif
