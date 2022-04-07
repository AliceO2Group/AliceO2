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

#ifndef _ZDC_INTERCALIB_DATA_H_
#define _ZDC_INTERCALIB_DATA_H_

#include "DetectorsCalibration/TimeSlot.h"
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
  static constexpr int NH = 5;       /// ZNA, ZPA, ZNC, ZPC, ZEM
  double mSum[NH][NPAR][NPAR] = {0}; /// Cumulated sums
  static constexpr const char* DN[NH] = {"ZNA", "ZPA", "ZNC", "ZPC", "ZEM"};
  InterCalibData& operator+=(const InterCalibData& other);
  void print() const;
  ClassDefNV(InterCalibData, 1);
};

} // namespace zdc
} // namespace o2

#endif
