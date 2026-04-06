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

/// \file CalVdriftExB.h
/// \brief Object with vDrift and ExB values per chamber to be written into the CCDB

#ifndef ALICEO2_CALVDRIFTEXB_H
#define ALICEO2_CALVDRIFTEXB_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include <array>

namespace o2
{
namespace trd
{

class CalVdriftExB
{
 public:
  CalVdriftExB() = default;
  CalVdriftExB(const CalVdriftExB&) = default;
  ~CalVdriftExB() = default;

  void setVdrift(int iDet, float vd) { mVdrift[iDet] = vd; }
  void setExB(int iDet, float exb) { mExB[iDet] = exb; }

  float getVdrift(int iDet, bool defaultAvg = true) const
  {
    // if defaultAvg = false, we take the value stored whatever it is
    // if defaultAvg = true and we have default value or bad value stored, we take the average on all chambers instead
    if (!defaultAvg || (isGoodExB(iDet) && isGoodVdrift(iDet)))
      return mVdrift[iDet];
    else {
      if (std::fabs(mMeanVdrift + 999.) < 1e-6)
        mMeanVdrift = getAverageVdrift();
      return mMeanVdrift;
    }
  }
  float getExB(int iDet, bool defaultAvg = true) const
  {
    if (!defaultAvg || (isGoodExB(iDet) && isGoodVdrift(iDet)))
      return mExB[iDet];
    else {
      if (std::fabs(mMeanExB + 999.) < 1e-6)
        mMeanExB = getAverageExB();
      return mMeanExB;
    }
  }

  float getAverageVdrift() const
  {
    float averageVdrift = 0.;
    int ngood = 0;

    for (int iDet = 0; iDet < constants::MAXCHAMBER; iDet++) {
      if (isGoodExB(iDet) && isGoodVdrift(iDet)) {
        // Both values need to be correct to declare a chamber as well calibrated
        ngood++;
        averageVdrift += mVdrift[iDet];
      }
    }
    if (ngood == 0) {
      // we should make sure it never happens
      return constants::VDRIFTDEFAULT;
    }
    averageVdrift /= ngood;
    return averageVdrift;
  }

  float getAverageExB() const
  {
    float averageExB = 0.;
    int ngood = 0;

    for (int iDet = 0; iDet < constants::MAXCHAMBER; iDet++) {
      if (isGoodExB(iDet) && isGoodVdrift(iDet)) {
        // Both values need to be correct to declare a chamber as well calibrated
        ngood++;
        averageExB += mExB[iDet];
      }
    }
    if (ngood == 0) {
      // we should make sure it never happens
      return constants::EXBDEFAULT;
    }
    averageExB /= ngood;
    return averageExB;
  }

  bool isGoodExB(int iDet) const
  {
    // check if value is well calibrated or not
    // default calibration if not enough entries
    // close to boundaries indicate a failed fit
    if (std::fabs(mExB[iDet] - constants::EXBDEFAULT) > 1e-6 &&
        std::fabs(mExB[iDet] - constants::EXBMIN) > 0.01 &&
        std::fabs(mExB[iDet] - constants::EXBMAX) > 0.01)
      return true;
    else
      return false;
  }

  bool isGoodVdrift(int iDet) const
  {
    // check if value is well calibrated or not
    // default calibration if not enough entries
    // close to boundaries indicate a failed fit
    if (std::fabs(mVdrift[iDet] - constants::VDRIFTDEFAULT) > 1e-6 &&
        std::fabs(mVdrift[iDet] - constants::VDRIFTMIN) > 0.1 &&
        std::fabs(mVdrift[iDet] - constants::VDRIFTMAX) > 0.1)
      return true;
    else
      return false;
  }

 private:
  std::array<float, constants::MAXCHAMBER> mVdrift{}; ///< calibrated drift velocity per TRD chamber
  std::array<float, constants::MAXCHAMBER> mExB{};    ///< calibrated Lorentz angle per TRD chamber
  mutable float mMeanVdrift{-999.};                   ///! average drift velocity, calculated only once
  mutable float mMeanExB{-999.};                      ///! average lorentz angle, calculated only once

  ClassDefNV(CalVdriftExB, 2);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_CALVDRIFTEXB_H
