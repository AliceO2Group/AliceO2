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

  float getMPVdEdx(int iDet) const { return mMPVdEdx[iDet]; }

 private:
  std::array<float, constants::MAXCHAMBER> mMPVdEdx{}; ///< Most probable value of dEdx distribution per TRD chamber

  ClassDefNV(CalGain, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_CALGAIN_H
