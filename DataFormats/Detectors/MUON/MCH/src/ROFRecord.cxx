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

#include "DataFormatsMCH/ROFRecord.h"

#include <fmt/format.h>
#include <iostream>
#include <stdexcept>

#include "CommonConstants/LHCConstants.h"
#include "Framework/Logger.h"

namespace o2::mch
{
std::ostream& operator<<(std::ostream& os, const ROFRecord& rof)
{
  os << fmt::format("{} FirstIdx: {:5d} LastIdx: {:5d} Width: {:2d} BCs",
                    rof.getBCData().asString(), rof.getFirstIdx(), rof.getLastIdx(),
                    rof.getBCWidth());
  return os;
}

//__________________________________________________________________________
/// return a pair consisting of the ROF time with error (in mus) relative to the reference IR 'startIR'
/// and a flag telling if it is inside the TF starting at 'startIR' and containing 'nOrbits' orbits.
/// if printError = true, print an error message in case the ROF is outside the TF
std::pair<ROFRecord::Time, bool> ROFRecord::getTimeMUS(const BCData& startIR, uint32_t nOrbits, bool printError) const
{
  auto bcDiff = mBCData.differenceInBC(startIR);
  float tMean = (bcDiff + 0.5 * mBCWidth) * o2::constants::lhc::LHCBunchSpacingMUS;
  float tErr = 0.5 * mBCWidth * o2::constants::lhc::LHCBunchSpacingMUS;
  bool isInTF = bcDiff >= 0 && bcDiff < nOrbits * o2::constants::lhc::LHCMaxBunches;
  if (printError && !isInTF) {
    LOGP(alarm, "ATTENTION: wrong bunches diff. {} for current IR {} wrt 1st TF orbit {}, source:MCH",
         bcDiff, mBCData, startIR);
  }
  return std::make_pair(Time(tMean, tErr), isInTF);
}

} // namespace o2::mch
