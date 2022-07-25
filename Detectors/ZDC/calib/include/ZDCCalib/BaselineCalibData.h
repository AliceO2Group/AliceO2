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

#ifndef ZDC_BASELINECALIB_DATA_H
#define ZDC_BASELINECALIB_DATA_H

#include "Framework/Logger.h"
#include "ZDCBase/Constants.h"
#include "ZDCCalib/BaselineCalibConfig.h"
#include <array>
#include <vector>
#include <gsl/span>

/// \file BaselineCalibData.h
/// \brief Baseline calibration intermediate data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct BaselineCalibBinData {
  BaselineCalibBinData() = default;
  BaselineCalibBinData(uint8_t myid, uint16_t myibin, uint32_t mycont)
  {
    id = myid;
    ibin = myibin;
    cont = mycont;
  }
  uint8_t id = 0xff; // channel ID
  uint16_t ibin = 0; // bin number
  uint32_t cont = 0; // channel counts
  void print() const
  {
    LOGF(info, "BaselineCalibBinData ch=%2u bin=%4u cont=%u", id, ibin, cont);
  }
  ClassDefNV(BaselineCalibBinData, 1);
};

// N.B. Overflow bits are included however in the ALICE data taking conditions
// an overflow could occur only during a run longer than 104 hours. An overflow
// is therefore a sign of a hidden problem

struct BaselineCalibSummaryData {
  BaselineCalibSummaryData() = default;
  uint64_t mCTimeBeg = 0;                    /// Time of processed time frame
  uint64_t mCTimeEnd = 0;                    /// Time of processed time frame
  bool mOverflow = false;                    /// Overflow of one channel
  std::array<bool, NChannels> mOverflowCh{}; /// Channel overflow information
  std::vector<BaselineCalibBinData> mData;   /// Data of not empty bins
  void clear();
  void print() const;
  ClassDefNV(BaselineCalibSummaryData, 1);
};

struct BaselineCalibChData {
  BaselineCalibChData() = default;
  static constexpr int NW = BaselineRange; /// 2^16 bins
  std::array<uint32_t, NW> mData = {0};    /// Histogram container
  bool mOverflow = false;                  /// Overflow flag (cannot accept more data)

  uint64_t getEntries() const;
  int getStat(uint64_t& en, double& mean, double& var) const;
  inline bool isOverflow() { return mOverflow; };
  void clear();
  ClassDefNV(BaselineCalibChData, 1);
};

struct BaselineCalibData {
  BaselineCalibData() = default;

  uint64_t mCTimeBeg = 0; /// Time of processed time frame
  uint64_t mCTimeEnd = 0; /// Time of processed time frame
  bool mOverflow = false; /// Overflow at least one ZDC channel

  BaselineCalibChData mHisto[NChannels]; /// Histogram for single channel
  BaselineCalibSummaryData mSummary;

  BaselineCalibData& operator+=(const BaselineCalibData& other);
  BaselineCalibData& operator=(const BaselineCalibSummaryData& s);
  // BaselineCalibData& operator+=(const BaselineCalibSummaryData& s);
  BaselineCalibData& operator+=(const BaselineCalibSummaryData* s);

  inline void addEntry(int isig, zdcBaseline_t val)
  {
    if (!mHisto[isig].mOverflow) {
      int ibin = val - BaselineMin;
      if (mHisto[isig].mData[ibin] < 0xffffffff) {
        mHisto[isig].mData[ibin]++;
      } else {
        mHisto[isig].mOverflow = true;
        mOverflow = true;
      }
    }
  }
  uint64_t getEntries(int is) const;
  int getStat(int is, uint64_t& en, double& mean, double& var) const;
  void print() const;
  void clear();
  void setCreationTime(uint64_t ctime);
  void setN(int n);
  BaselineCalibSummaryData& getSummary();
  int saveDebugHistos(const std::string fn, float factor);
  ClassDefNV(BaselineCalibData, 1);
};

} // namespace zdc
} // namespace o2

#endif
