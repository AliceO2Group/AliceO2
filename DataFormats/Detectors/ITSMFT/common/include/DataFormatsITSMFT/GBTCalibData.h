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

/// \file GBTCalibData.h
/// \brief Calibration data from GBT data

#ifndef ALICEO2_ITSMFT_GBTCALIBDATA_H_
#define ALICEO2_ITSMFT_GBTCALIBDATA_H_

namespace o2
{
namespace itsmft
{

struct GBTCalibData {
  uint64_t calibCounter = 0;   // calibCounter from GBT calibration word
  uint64_t calibUserField = 0; // calibUserField from GBT calibration word

  void clear()
  {
    calibCounter = calibUserField = 0;
  }

  ClassDefNV(GBTCalibData, 1);
};

} // namespace itsmft
} // namespace o2

#endif
