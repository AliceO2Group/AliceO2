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

#include "Framework/Logger.h"
#include "DataFormatsZDC/RecEventFlat.h"

using namespace o2::zdc;

void RecEventFlat::init(std::vector<o2::zdc::BCRecData>* RecBC, std::vector<o2::zdc::ZDCEnergy>* Energy, std::vector<o2::zdc::ZDCTDCData>* TDCData, std::vector<uint16_t>* Info)
{
  mRecBC = RecBC;
  mEnergy = Energy;
  mTDCData = TDCData;
  mInfo = Info;
  mEntry = 0;
  mNEntries = mRecBC->size();
}

int RecEventFlat::next()
{
  if (mEntry >= mNEntries) {
    return 0;
  }

  tdcPedEv.fill(false);
  tdcPedOr.fill(false);
  tdcPedQC.fill(false);
  tdcPedMissing.fill(false);
  adcPedEv.fill(false);
  adcPedOr.fill(false);
  adcPedQC.fill(false);
  adcPedMissing.fill(false);
  ezdc.clear();
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    TDCVal[itdc].clear();
    TDCAmp[itdc].clear();
  }
  auto& curb = mRecBC->at(mEntry);

  // Decode event info
  int istart = curb.refi.getFirstEntry();
  int istop = istart + curb.refi.getEntries();
  int infoState = 0;
  uint16_t code = 0;
  uint32_t map = 0;
  for (int i = istart; i < istop; i++) {
    uint16_t info = mInfo->at(i);
    printf("0x%04x\n", info);
    if (infoState == 0) {
      if (info & 0x8000) {
        LOGF(ERROR, "Inconsistent info stream at word %d: 0x%4u", i, info);
        break;
      }
      code = info & 0x03ff;
      uint8_t ch = (info >> 10) & 0x1f;
      if (ch == 0x1f) {
        infoState = 1;
      } else if (ch < NChannels) {
        decodeInfo(ch, code);
      } else {
        LOGF(ERROR, "Info about non existing channel: %u", ch);
      }
    } else if (infoState == 1) {
      if (info & 0x8000) {
        map = info & 0x7fff;
      } else {
        LOGF(ERROR, "Inconsistent info stream at word %d: 0x%4u", i, info);
        break;
      }
      infoState = 2;
    } else if (infoState == 2) {
      if (info & 0x8000) {
        uint32_t maph = info & 0x7fff;
        map = (maph << 15) | map;
        decodeMapInfo(map, code);
      } else {
        LOGF(ERROR, "Inconsistent info stream at word %d: 0x%4u", i, info);
        break;
      }
      infoState = 0;
    }
  }

  ir = curb.ir;
  channels = curb.channels;
  triggers = curb.triggers;
  // Decode energy
  istart = curb.refe.getFirstEntry();
  istop = istart + curb.refe.getEntries();
  for (int i = istart; i < istop; i++) {
    auto myenergy = mEnergy->at(i);
    auto ch = myenergy.ch();
    ezdc[ch] = myenergy.energy();
    // Assign implicit event info
    if (adcPedOr[ch] == false && adcPedQC[ch] == false && adcPedMissing[ch] == false) {
      adcPedEv[ch] = true;
    }
  }
  // Decode TDCs
  istart = curb.reft.getFirstEntry();
  istop = istart + curb.reft.getEntries();
  for (int i = istart; i < istop; i++) {
    auto mytdc = mTDCData->at(i);
    auto ch = mytdc.ch();
    if (ch < NTDCChannels) {
      TDCVal[ch].push_back(mytdc.val);
      TDCAmp[ch].push_back(mytdc.amp);
      // Assign implicit event info
      if (tdcPedQC[ch] == false && tdcPedMissing[ch] == false) {
        tdcPedOr[ch] = true;
      }
    }
  }

  mEntry++;
  return mEntry;
}

void RecEventFlat::decodeMapInfo(uint32_t map, uint16_t code)
{
#ifdef O2_ZDC_DEBUG
  printf("decodeMapInfo%08x code=%u\n", map, code);
#endif
  for (uint8_t ch = 0; ch < NChannels; ch++) {
    if (map & (0x1 << ch)) {
      decodeInfo(ch, code);
    }
  }
}

void RecEventFlat::decodeInfo(uint8_t ch, uint16_t code)
{
  if (mVerbosity != DbgZero) {
    printf("Info: ch=%2d code=%4u %s\n", ch, code, code < MsgEnd ? MsgText[code].data() : "undefined");
  }
  if (code == MsgTDCPedQC) {
    tdcPedQC[ch] = true;
  }
  if (code == MsgTDCPedMissing) {
    tdcPedMissing[ch] = true;
  }
  if (code == MsgADCPedOr) {
    adcPedOr[ch] = true;
  }
  if (code == MsgADCPedQC) {
    adcPedQC[ch] = true;
  }
  if (code == MsgADCPedMissing) {
    adcPedMissing[ch] = true;
  }
}

void RecEventFlat::print() const
{
  ir.print();
}
