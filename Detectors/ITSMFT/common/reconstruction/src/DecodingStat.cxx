// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChipStat.cxx
/// \brief Alpide Chip decoding statistics

#include <bitset>
#include "ITSMFTReconstruction/DecodingStat.h"
#include "Framework/Logger.h"

using namespace o2::itsmft;
constexpr std::array<std::string_view, ChipStat::NErrorsDefined> ChipStat::ErrNames;

///_________________________________________________________________
/// print chip decoding statistics
void ChipStat::print(bool skipEmpty) const
{
  uint32_t nErr = 0;
  for (int i = NErrorsDefined; i--;) {
    nErr += errorCounts[i];
  }
  LOGF(INFO, "Chip#%5d NHits: %9zu  errors: %u", chipID, nHits, nErr);
  for (int i = 0; i < NErrorsDefined; i++) {
    if (!skipEmpty || errorCounts[i]) {
      LOGF(INFO, "%-70s: %u", ErrNames[i].data(), errorCounts[i]);
    }
  }
}

///_________________________________________________________________
uint32_t ChipStat::getNErrors() const
{
  uint32_t nerr = 0;
  for (int i = NErrorsDefined; i--;) {
    nerr += errorCounts[i];
  }
  return nerr;
}

///_________________________________________________________________
/// print link decoding statistics
void GBTLinkDecodingStat::print(bool skipEmpty) const
{
  int nErr = 0;
  for (int i = NErrorsDefined; i--;) {
    nErr += errorCounts[i];
  }
  LOGF(INFO, "GBTLink#0x%d Packet States Statistics (total packets: %d, triggers: %d)", ruLinkID, nPackets, nTriggers);
  for (int i = 0; i < GBTDataTrailer::MaxStateCombinations; i++) {
    if (packetStates[i]) {
      std::bitset<GBTDataTrailer::NStatesDefined> patt(i);
      LOGF(INFO, "counts for triggers B[%s] : %d", patt.to_string().c_str(), packetStates[i]);
    }
  }
  LOGF(INFO, "Decoding errors: %u", nErr);
  for (int i = 0; i < NErrorsDefined; i++) {
    if (!skipEmpty || errorCounts[i]) {
      LOGF(INFO, "%-70s: %u", ErrNames[i].data(), errorCounts[i]);
    }
  }
}
