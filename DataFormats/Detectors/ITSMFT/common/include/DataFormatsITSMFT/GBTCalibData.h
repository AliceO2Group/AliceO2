// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
