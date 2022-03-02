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
  ninfo += addInfo(reca, reca.genericE, MsgGeneric);             //  0
  ninfo += addInfo(reca, reca.tdcPedQC, MsgTDCPedQC);            //  1
  ninfo += addInfo(reca, reca.tdcPedMissing, MsgTDCPedMissing);  //  2
  ninfo += addInfo(reca, reca.adcPedOr, MsgADCPedOr);            //  3
  ninfo += addInfo(reca, reca.adcPedQC, MsgADCPedQC);            //  4
  ninfo += addInfo(reca, reca.adcPedMissing, MsgADCPedMissing);  //  5
  ninfo += addInfo(reca, reca.offPed, MsgOffPed);                //  6
  ninfo += addInfo(reca, reca.pilePed, MsgPilePed);              //  7
  ninfo += addInfo(reca, reca.pileTM, MsgPileTM);                //  8
  ninfo += addInfo(reca, reca.adcPedMissing, MsgADCMissingwTDC); //  9
  ninfo += addInfo(reca, reca.tdcPileEvC, MsgTDCPileEvC);        // 10
  ninfo += addInfo(reca, reca.tdcPileEvE, MsgTDCPileEvE);        // 11
  ninfo += addInfo(reca, reca.tdcPileM1C, MsgTDCPileM1C);        // 12
  ninfo += addInfo(reca, reca.tdcPileM1E, MsgTDCPileM1E);        // 13
  ninfo += addInfo(reca, reca.tdcPileM2C, MsgTDCPileM2C);        // 14
  ninfo += addInfo(reca, reca.tdcPileM2E, MsgTDCPileM2E);        // 15
  ninfo += addInfo(reca, reca.tdcPileM3C, MsgTDCPileM3C);        // 16
  ninfo += addInfo(reca, reca.tdcPileM3E, MsgTDCPileM3E);        // 17
  ninfo += addInfo(reca, reca.tdcSigE, MsgTDCSigE);              // 18
  return ninfo;
}
