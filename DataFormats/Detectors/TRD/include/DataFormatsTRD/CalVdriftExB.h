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

  float getVdrift(int iDet) const { return mVdrift[iDet]; }
  float getExB(int iDet) const { return mExB[iDet]; }

 private:
  std::array<float, constants::MAXCHAMBER> mVdrift{}; ///< calibrated drift velocity per TRD chamber
  std::array<float, constants::MAXCHAMBER> mExB{};    ///< calibrated Lorentz angle per TRD chamber

  ClassDefNV(CalVdriftExB, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_CALVDRIFTEXB_H
