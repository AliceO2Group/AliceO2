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
void ChipStat::addErrors(uint32_t mask, uint16_t id)
{
  if (mask) {
    for (int i = NErrorsDefined; i--;) {
      if (mask & (0x1 << i)) {
        if (!errorCounts[i]) {
          LOGP(ERROR, "New error registered on the link: chip#{}: {}", id, ErrNames[i]);
        }
        errorCounts[i]++;
      }
    }
  }
}

///_________________________________________________________________
/// print chip decoding statistics
void ChipStat::print(bool skipNoErr, const std::string& pref) const
{
  uint32_t nErr = 0;
  for (int i = NErrorsDefined; i--;) {
    nErr += errorCounts[i];
  }
  if (!skipNoErr || nErr) {
    LOGP(INFO, "{}#{:x} NHits: {}  errors: {}", pref.c_str(), id, nHits, nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipNoErr || errorCounts[i]) {
        LOGP(INFO, "Err.: {}: {}", ErrNames[i].data(), errorCounts[i]);
      }
    }
  }
}

///_________________________________________________________________
/// print link decoding statistics
void GBTLinkDecodingStat::print(bool skipNoErr) const
{
  int nErr = 0;
  for (int i = NErrorsDefined; i--;) {
    nErr += errorCounts[i];
  }
  if (!skipNoErr || nErr) {
    LOGP(INFO, "FEEID#{%s} Packet States Statistics (total packets: {}, triggers: {})", ruLinkID, nPackets, nTriggers);
    for (int i = 0; i < GBTDataTrailer::MaxStateCombinations; i++) {
      if (packetStates[i]) {
        std::bitset<GBTDataTrailer::NStatesDefined> patt(i);
        LOGP(INFO, "counts for triggers B{:s}: {}", patt.to_string().c_str(), packetStates[i]);
      }
    }
    LOGP(INFO, "Decoding errors: {}", nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipNoErr || errorCounts[i]) {
        LOGF(INFO, "{<}: {}", ErrNames[i].data(), errorCounts[i]);
      }
    }
  }
}
