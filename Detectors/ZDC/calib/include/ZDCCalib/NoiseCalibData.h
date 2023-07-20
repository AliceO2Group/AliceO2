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

#ifndef ZDC_NOISECALIB_DATA_H
#define ZDC_NOISECALIB_DATA_H

#include "Framework/Logger.h"
#include "ZDCBase/Constants.h"
#include <array>
#include <vector>
#include <map>
#include <gsl/span>

/// \file NoiseCalibData.h
/// \brief Format of noise calibration intermediate data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

// Messageable representation of noise calibration data
struct NoiseCalibBinData {
  NoiseCalibBinData() = default;
  NoiseCalibBinData(uint8_t myid, uint32_t myibin, uint32_t mycont)
  {
    ibin = (myid << 24) | (myibin & 0x00ffffff);
    cont = mycont;
  }
  uint32_t ibin = 0xff000000; /// Encoded channel ID / bin number
  uint32_t cont = 0;          /// Channel counts
  inline uint32_t id() const
  { // Channel id
    return (ibin >> 24);
  }
  inline uint32_t bin() const
  { // Bin number
    return (ibin & 0x00ffffff);
  }
  inline uint32_t counts() const
  { // Channel counts
    return cont;
  }
  void print() const
  {
    LOGF(info, "NoiseCalibBinData ch=%2u bin=%4u cont=%u", id(), bin(), counts());
  }
  ClassDefNV(NoiseCalibBinData, 1);
};

// Container of the messageable representation of noise calibration data
struct NoiseCalibSummaryData {
  NoiseCalibSummaryData() = default;
  uint64_t mCTimeBeg = 0;                    /// Time of processed time frame
  uint64_t mCTimeEnd = 0;                    /// Time of processed time frame
  bool mOverflow = false;                    /// Overflow of one channel
  std::array<bool, NChannels> mOverflowCh{}; /// Channel overflow information
  std::vector<NoiseCalibBinData> mData;      /// Data of not empty bins
  void clear();
  void print() const;
  ClassDefNV(NoiseCalibSummaryData, 1);
};

// Working representation of noise channel data
struct NoiseCalibChData {
  NoiseCalibChData() = default;
  // Variance intermediate data are uint32_t and sparse (~25% channels are filled)
  // and histogram limits are not known in advance -> use map
  std::map<uint32_t, uint32_t> mData; /// Map histogram container
  bool mOverflow = false;             /// Overflow flag (cannot accept more data)
  uint64_t getEntries() const;
  uint32_t getMaxBin() const;
  int getStat(uint64_t& en, double& mean, double& var) const;
  inline bool isOverflow() { return mOverflow; };
  void clear();
  ClassDefNV(NoiseCalibChData, 1);
};

// Working representation of noise data
struct NoiseCalibData {
  NoiseCalibData() = default;

  uint64_t mCTimeBeg = 0; /// Time of processed time frame
  uint64_t mCTimeEnd = 0; /// Time of processed time frame
  bool mOverflow = false; /// Overflow at least one ZDC channel

  NoiseCalibChData mHisto[NChannels]; /// Sparse histogram of single channels
  NoiseCalibSummaryData mSummary;     /// Serialized data to be dispatched

  NoiseCalibData& operator+=(const NoiseCalibData& other);
  NoiseCalibData& operator=(const NoiseCalibSummaryData& s);
  // NoiseCalibData& operator+=(const NoiseCalibSummaryData& s);
  NoiseCalibData& operator+=(const NoiseCalibSummaryData* s);

  inline void addEntry(int isig, uint32_t val)
  {
    if (!mHisto[isig].mOverflow) {
      int ibin = 0x00ffffff;
      // Overflow in bin number
      if (val < 0x00ffffff) {
        ibin = val;
      }
      // Overflow in data
      if (mHisto[isig].mData[ibin] < 0xffffffff) {
        mHisto[isig].mData[ibin]++;
      } else {
        mHisto[isig].mOverflow = true;
        mOverflow = true;
      }
    }
  }

  static constexpr int NHA = 3;

  uint64_t getEntries(int is) const;
  uint32_t getMaxBin(int is) const;
  int getStat(int is, uint64_t& en, double& mean, double& var) const;
  void print() const;
  void clear();
  void setCreationTime(uint64_t ctime);
  void mergeCreationTime(uint64_t ctime);
  void setN(int n);
  NoiseCalibSummaryData& getSummary();
  int saveDebugHistos(const std::string fn, bool is_epn = false);
  ClassDefNV(NoiseCalibData, 1);
};

} // namespace zdc
} // namespace o2

#endif
