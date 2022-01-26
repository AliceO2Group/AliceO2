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

#include "DataFormatsZDC/RecEvent.h"

using namespace o2::zdc;

void RecEvent::print() const
{
  for (auto bcdata : mRecBC) {
    bcdata.ir.print();
    int fe, ne, ft, nt, fi, ni;
    bcdata.getRef(fe, ne, ft, nt, fi, ni);
    for (int ie = 0; ie < ne; ie++) {
      mEnergy[fe + ie].print();
    }
    for (int it = 0; it < nt; it++) {
      mTDCData[ft + it].print();
    }
    // TODO: event info
  }
}

uint32_t RecEvent::addInfo(const RecEventAux& reca, const std::array<bool, NChannels>& vec, const uint16_t code)
{
  // Prepare list of channels interested by this message
  int cnt = 0;
  std::array<int, NChannels> active;
  for (uint8_t ich = 0; ich < NChannels; ich++) {
    if (vec[ich]) {
      active[cnt] = ich;
      cnt++;
    }
  }
  if (cnt == 0) {
    return cnt;
  }
  // Add bunch crossing info to current event if needed
  if (mRecBC.size() == 0 || reca.ir != mRecBC.back().ir) {
    addBC(reca);
  }
  if (cnt <= 3) {
    // Transmission of single channels
    for (uint8_t i = 0; i < cnt; i++) {
      addInfo(active[i], code);
    }
  } else {
    // Transmission of channel pattern
    uint16_t ch = 0x1f;
    addInfo(ch, code);
    uint16_t info = 0x8000;
    uint8_t i = 0;
    for (; i < cnt && active[i] < 15; i++) {
      info = info | (0x1 << active[i]);
    }
    addInfo(info);
#ifdef O2_ZDC_DEBUG
    if (info & 0x7fff) {
      for (uint8_t ich = 0; ich < 15; ich++) {
        if (vec[ich]) {
          printf(" %s", ChannelNames[ich].data());
        }
      }
      printf("\n");
    }
#endif
    info = 0x8000;
    for (; i < cnt; i++) {
      info = info | (0x1 << (active[i] - 15));
    }
#ifdef O2_ZDC_DEBUG
    if (info & 0x7fff) {
      for (uint8_t ich = 15; ich < NChannels; ich++) {
        if (vec[ich]) {
          printf(" %s", ChannelNames[ich].data());
        }
      }
      printf("\n");
    }
#endif
    addInfo(info);
  }
  return cnt;
}

uint32_t RecEvent::addInfos(const RecEventAux& reca)
{
  // Reconstruction messages
  uint32_t ninfo = 0;
  ninfo += addInfo(reca, reca.tdcPedQC, MsgTDCPedQC);
  ninfo += addInfo(reca, reca.tdcPedMissing, MsgTDCPedMissing);
  ninfo += addInfo(reca, reca.adcPedOr, MsgADCPedOr);
  ninfo += addInfo(reca, reca.adcPedQC, MsgADCPedQC);
  ninfo += addInfo(reca, reca.adcPedMissing, MsgADCPedMissing);
  ninfo += addInfo(reca, reca.offPed, MsgOffPed);
  ninfo += addInfo(reca, reca.pilePed, MsgPilePed);
  ninfo += addInfo(reca, reca.pileTM, MsgPileTM);
  ninfo += addInfo(reca, reca.adcPedMissing, MsgADCMissingwTDC);
  return ninfo;
}
