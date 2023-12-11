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

#ifndef _ZDC_INTERCALIB_DATA_H
#define _ZDC_INTERCALIB_DATA_H

#include "ZDCBase/Constants.h"
#include <array>
#include <Rtypes.h>

/// \file InterCalibData.h
/// \brief Intercalibration intermediate data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct InterCalibData {
  static constexpr int NPAR = 6;     /// Dimension of matrix (1 + 4 coefficients + offset)
  static constexpr int NH = 9;       /// ZNA, ZPA, ZNC, ZPC, ZEM, ZNI, ZPI, ZPAX, ZPCX
  double mSum[NH][NPAR][NPAR] = {0}; /// Cumulated sums
  uint64_t mCTimeBeg = 0;            /// Time of processed time frame
  uint64_t mCTimeEnd = 0;            /// Time of processed time frame
  static constexpr const char* DN[NH] = {"ZNA", "ZPA", "ZNC", "ZPC", "ZEM", "ZNI", "ZPI", "ZPAX", "ZPCX"};
  InterCalibData& operator+=(const InterCalibData& other);
  int getEntries(int ih) const;
  void print() const;
  void setCreationTime(uint64_t ctime);
  ClassDefNV(InterCalibData, 2);
};

} // namespace zdc
} // namespace o2

#endif
