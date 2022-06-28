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

/// \file ChipStat.cxx
/// \brief Alpide Chip decoding statistics

#include <bitset>
#include "ITSMFTReconstruction/DecodingStat.h"
#include "ITSMFTReconstruction/PixelData.h"
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
uint32_t ChipStat::addErrors(uint32_t mask, uint16_t chID, int verbosity)
{
  uint32_t res = 0;
  if (mask) {
    for (int i = NErrorsDefined; i--;) {
      if (mask & (0x1 << i)) {
        res |= ErrActions[i] & ErrActPropagate;
        if (verbosity > -1 && (!errorCounts[i] || verbosity > 1)) {
          LOGP(info, "New error registered on the FEEID:{:#04x}: chip#{}: {}", feeID, chID, ErrNames[i]);
          res |= ErrActions[i] & ErrActDump;
        }
        errorCounts[i]++;
      }
    }
  }
  return res;
}

///_________________________________________________________________
/// print link decoding statistics
uint32_t ChipStat::addErrors(const ChipPixelData& d, int verbosity)
{
  uint32_t res = 0;
  if (d.getErrorFlags()) {
    for (int i = NErrorsDefined; i--;) {
      if (d.getErrorFlags() & (0x1 << i)) {
        res |= ErrActions[i] & ErrActPropagate;
        if (verbosity > -1 && (!errorCounts[i] || verbosity > 1)) {
          LOGP(info, "New error registered at bc/orbit {}/{} on the FEEID:{:#04x} chip#{}: {}{}",
               d.getInteractionRecord().bc, d.getInteractionRecord().orbit,
               feeID, int16_t(d.getChipID()), ErrNames[i], d.getErrorDetails(i));
          res |= ErrActions[i] & ErrActDump;
        }
        errorCounts[i]++;
      }
    }
  }
  return res;
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
    std::string rep = fmt::format("{}#{:#04x} NHits: {}  errors: {}", pref.c_str(), feeID, nHits, nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipNoErr || errorCounts[i]) {
        rep += fmt::format(" | Err.: {}: {}", ErrNames[i].data(), errorCounts[i]);
      }
    }
    LOG(info) << rep;
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
    std::string rep = fmt::format("FEEID#{:#04x} Packet States Statistics (total packets: {}, triggers: {})", ruLinkID, nPackets, nTriggers);
    bool countsSeen = false;
    for (int i = 0; i < GBTDataTrailer::MaxStateCombinations; i++) {
      if (packetStates[i]) {
        if (!countsSeen) {
          countsSeen = true;
          rep += " | counts for triggers: ";
        } else {
          rep += ", ";
        }
        std::bitset<GBTDataTrailer::NStatesDefined> patt(i);
        rep += fmt::format("b{:s}: {}", patt.to_string().c_str(), packetStates[i]);
      }
    }
    rep += fmt::format(" | Decoding errors: {}", nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipNoErr || errorCounts[i]) {
        rep += fmt::format(" [{}: {}]", ErrNames[i].data(), errorCounts[i]);
      }
    }
    LOG(info) << rep;
  }
}
