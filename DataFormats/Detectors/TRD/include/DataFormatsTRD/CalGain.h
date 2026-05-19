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

/// \file CalGain.h
/// \brief Object with MPV dEdx values per chamber to be written into the CCDB

#ifndef ALICEO2_CALGAIN_H
#define ALICEO2_CALGAIN_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include <array>

namespace o2
{
namespace trd
{

class CalGain
{
 public:
  CalGain() = default;
  CalGain(const CalGain&) = default;
  ~CalGain() = default;

  void setMPVdEdx(int iDet, float mpv) { mMPVdEdx[iDet] = mpv; }

  float getMPVdEdx(int iDet, bool defaultAvg = true) const
  {
    // if defaultAvg = false, we take the value stored whatever it is
    // if defaultAvg = true and we have default value or bad value stored, we take the average on all chambers instead
    if (!defaultAvg || isGoodGain(iDet))
      return mMPVdEdx[iDet];
    else {
      if (std::fabs(mMeanGain + 999.) < 1e-6)
        mMeanGain = getAverageGain();
      return mMeanGain;
    }
  }

  float getAverageGain() const
  {
    float averageGain = 0.;
    int ngood = 0;

    for (int iDet = 0; iDet < constants::MAXCHAMBER; iDet++) {
      if (isGoodGain(iDet)) {
        // The chamber has correct calibration
        ngood++;
        averageGain += mMPVdEdx[iDet];
      }
    }
    if (ngood == 0) {
      // we should make sure it never happens
      return constants::MPVDEDXDEFAULT;
    }
    averageGain /= ngood;
    return averageGain;
  }

  bool isGoodGain(int iDet) const
  {
    if (std::fabs(mMPVdEdx[iDet] - constants::MPVDEDXDEFAULT) > 1e-6)
      return true;
    else
      return false;
  }

 private:
  std::array<float, constants::MAXCHAMBER> mMPVdEdx{}; ///< Most probable value of dEdx distribution per TRD chamber
  mutable float mMeanGain{-999.};                      ///! average gain, calculated only once

  ClassDefNV(CalGain, 2);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_CALGAIN_H
