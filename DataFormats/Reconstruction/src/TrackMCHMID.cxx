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

/// \file TrackMCHMID.h
/// \brief Implementation of the MUON track
///
/// \author Philippe Pillot, Subatech

#include "ReconstructionDataFormats/TrackMCHMID.h"

#include <iostream>

#include "CommonConstants/LHCConstants.h"
#include "Framework/Logger.h"

namespace o2
{
namespace dataformats
{

//__________________________________________________________________________
/// write the content of the track to the output stream
std::ostream& operator<<(std::ostream& os, const o2::dataformats::TrackMCHMID& track)
{
  os << track.getMCHRef() << " + " << track.getMIDRef() << " = "
     << track.getIR() << " matching chi2/NDF: " << track.getMatchChi2OverNDF();
  return os;
}

//__________________________________________________________________________
/// write the content of the track to the standard output
void TrackMCHMID::print() const
{
  std::cout << *this << std::endl;
}

//__________________________________________________________________________
/// return a pair consisting of the track time with error (in mus) relative to the reference IR 'startIR'
/// and a flag telling if it is inside the TF starting at 'startIR' and containing 'nOrbits' orbits.
/// if printError = true, print an error message in case the track is outside the TF
std::pair<TrackMCHMID::Time, bool> TrackMCHMID::getTimeMUS(const InteractionRecord& startIR, uint32_t nOrbits,
                                                           bool printError) const
{
  auto bcDiff = mIR.differenceInBC(startIR);
  float tMean = (bcDiff + 0.5) * o2::constants::lhc::LHCBunchSpacingMUS;
  float tErr = 0.5 * o2::constants::lhc::LHCBunchSpacingMUS;
  bool isInTF = bcDiff >= 0 && bcDiff < nOrbits * o2::constants::lhc::LHCMaxBunches;
  if (printError && !isInTF) {
    LOGP(alarm, "ATTENTION: wrong bunches diff. {} for current IR {} wrt 1st TF orbit {}, source:MCH-MID",
         bcDiff, mIR, startIR);
  }
  return std::make_pair(Time(tMean, tErr), isInTF);
}

} // namespace dataformats
} // namespace o2
