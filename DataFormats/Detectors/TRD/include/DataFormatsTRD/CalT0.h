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

/// \file CalT0.h
/// \brief Object with T0 values per chamber to be written into the CCDB

#ifndef ALICEO2_CALT0_H
#define ALICEO2_CALT0_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include <array>

namespace o2
{
namespace trd
{

class CalT0
{
 public:
  CalT0() = default;
  CalT0(const CalT0&) = default;
  ~CalT0() = default;

  void setT0(int iDet, float t0) { mT0[iDet] = t0; }
  void setT0av(float t0) { mT0av = t0; }

  float getT0(int iDet) const { return mT0[iDet]; }
  // getT0av() returns the average T0 obtained by fitting the data from all chambers combined
  float getT0av() const { return mT0av; }
  // calcT0av() returns the average T0 from all individual chambers
  float calcT0av() const
  {
    if (mT0.size() == 0) {
      return -1;
    }
    float sum = 0;
    int counts = 0;
    for (int iDet = 0; iDet < constants::MAXCHAMBER; ++iDet) {
      if (mT0[iDet] > -5) {
        sum += mT0[iDet];
        ++counts;
      }
    }

    if (counts > 0) {
      return sum / counts;
    } else {
      return -2;
    }
  }

 private:
  std::array<float, constants::MAXCHAMBER> mT0{}; ///< calibrated T0 per TRD chamber
  float mT0av{-1};                                ///< average T0 obtained from fitting the PH data from all chambers combined

  ClassDefNV(CalT0, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_CALT0_H
