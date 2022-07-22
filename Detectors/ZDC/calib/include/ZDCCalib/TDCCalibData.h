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

#ifndef _ZDC_TDCCALIB_DATA_H
#define _ZDC_TDCCALIB_DATA_H

#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file   TDCCalibData.h
/// \brief  TDC calibration intermediate data
/// \author luca.quaglia@cern.ch

namespace o2
{
namespace zdc
{

struct TDCCalibData {
  static constexpr int NTDC = 10; /// ZNAC, ZNAS, ZPAC, ZPAS, ZEM1, ZEM2, ZNCC, ZNCS, ZPCC, ZPCS
  uint64_t mCTimeBeg = 0;         /// Time of processed time frame
  uint64_t mCTimeEnd = 0;         /// Time of processed time frame
  static constexpr const char* CTDC[NTDC] = {"ZNAC", "ZNAS", "ZPAC", "ZPAS", "ZEM1", "ZEM2", "ZNCC", "ZNCS", "ZPCC", "ZPCS"};
  int entries[NTDC] = {0};
  TDCCalibData& operator+=(const TDCCalibData& other);
  int getEntries(int ih) const;
  void print() const;
  void setCreationTime(uint64_t ctime);
  ClassDefNV(TDCCalibData, 1);
};

} // namespace zdc
} // namespace o2

#endif
