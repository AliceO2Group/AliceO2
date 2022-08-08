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

  float getT0(int iDet) const { return mT0[iDet]; }

 private:
  std::array<float, constants::MAXCHAMBER> mT0{}; ///< calibrated T0 per TRD chamber

  ClassDefNV(CalT0, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_CALT0_H
