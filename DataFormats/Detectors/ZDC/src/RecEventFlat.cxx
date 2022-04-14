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

#include "TString.h"
#include "Framework/Logger.h"
#include "DataFormatsZDC/RecEventFlat.h"

using namespace o2::zdc;

void RecEventFlat::init(const std::vector<o2::zdc::BCRecData>* RecBC, const std::vector<o2::zdc::ZDCEnergy>* Energy, const std::vector<o2::zdc::ZDCTDCData>* TDCData, const std::vector<uint16_t>* Info)
{
  mRecBC = *RecBC;
  mEnergy = *Energy;
  mTDCData = *TDCData;
  mInfo = *Info;
  mEntry = 0;
  mNEntries = mRecBC.size();
}

void RecEventFlat::init(const gsl::span<const o2::zdc::BCRecData> RecBC, const gsl::span<const o2::zdc::ZDCEnergy> Energy, const gsl::span<const o2::zdc::ZDCTDCData> TDCData, const gsl::span<const uint16_t> Info)
{
  mRecBC = RecBC;
  mEnergy = Energy;
  mTDCData = TDCData;
  mInfo = Info;
  mEntry = 0;
  mNEntries = mRecBC.size();
}

void RecEventFlat::clearBitmaps()
{
  tdcPedEv.fill(false);
  tdcPedOr.fill(false);
  tdcPedQC.fill(false);
  tdcPedMissing.fill(false);
  adcPedEv.fill(false);
  adcPedOr.fill(false);
  adcPedQC.fill(false);
  adcPedMissing.fill(false);
  adcMissingwTDC.fill(false);
  offPed.fill(false);
  pilePed.fill(false);
  pileTM.fill(false);
  tdcPileEvC.fill(false);
  tdcPileEvE.fill(false);
  tdcPileM1C.fill(false);
  tdcPileM1E.fill(false);
  tdcPileM2C.fill(false);
  tdcPileM2E.fill(false);
  tdcPileM3C.fill(false);
  tdcPileM3E.fill(false);
  // End_of_messages
  // Other bitmaps
  isBeg.fill(false);
  isEnd.fill(false);
}

int RecEventFlat::next()
{
  ezdcDecoded = 0;
  if (mEntry >= mNEntries) {
    return 0;
  }
  // Reconstruction messages
  clearBitmaps();
  // Channel data
  ezdc.clear();
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    TDCVal[itdc].clear();
    TDCAmp[itdc].clear();
  }

  // Get References
  mCurB = mRecBC[mEntry];
  mCurB.getRef(mFirstE, mNE, mFirstT, mNT, mFirstI, mNI);
  mStopE = mFirstE + mNE;
  mStopT = mFirstT + mNT;
  mStopI = mFirstI + mNI;

  ir = mCurB.ir;
  channels = mCurB.channels;
  triggers = mCurB.triggers;

  // Decode event info
  mDecodedInfo.clear();
  int infoState = 0;
  uint16_t code = 0;
  uint32_t map = 0;
  for (int i = mFirstI; i < mStopI; i++) {
    uint16_t info = mInfo[i];
    if (infoState == 0) {
      if (info & 0x8000) {
        LOGF(error, "Inconsistent info stream at word %d: 0x%4u", i, info);
        break;
      }
      code = info & 0x03ff;
      uint8_t ch = (info >> 10) & 0x1f;
      if (ch == 0x1f) {
        infoState = 1;
      } else if (ch < NChannels) {
        decodeInfo(ch, code);
      } else {
        LOGF(error, "Info about non existing channel: %u", ch);
      }
    } else if (infoState == 1) {
      if (info & 0x8000) {
        map = info & 0x7fff;
      } else {
        LOGF(error, "Inconsistent info stream at word %d: 0x%4u", i, info);
        break;
      }
      infoState = 2;
    } else if (infoState == 2) {
      if (info & 0x8000) {
        uint32_t maph = info & 0x7fff;
        map = (maph << 15) | map;
        decodeMapInfo(map, code);
      } else {
        LOGF(error, "Inconsistent info stream at word %d: 0x%4u", i, info);
        break;
      }
      infoState = 0;
    }
  }

  // Decode energy
  for (int i = mFirstE; i < mStopE; i++) {
    auto myenergy = mEnergy[i];
    auto ch = myenergy.ch();
    ezdc[ch] = myenergy.energy();
    // Assign implicit event info
    if (adcPedOr[ch] == false && adcPedQC[ch] == false && adcPedMissing[ch] == false) {
      adcPedEv[ch] = true;
    }
    ezdcDecoded |= (0x1 << ch);
  }

  // Decode TDCs
  for (int i = mFirstT; i < mStopT; i++) {
    auto mytdc = mTDCData[i];
    auto ch = mytdc.ch();
    if (ch < NTDCChannels) {
      if (mytdc.isBeg()) {
        isBeg[ch] = true;
      }
      if (mytdc.isEnd()) {
        isEnd[ch] = true;
      }
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
    printf("%9u.%04u Info: ch=%2d (%s) code=%-4u (%s)\n", ir.orbit, ir.bc, ch, ch < NChannels ? ChannelNames[ch].data() : "N.D.",
           code, code < MsgEnd ? MsgText[code].data() : "undefined");
  }
  // Reconstruction messages
  switch (code) {
    case MsgGeneric:
      genericE[ch] = true;
      break;
    case MsgTDCPedQC:
      tdcPedQC[ch] = true;
      break;
    case MsgTDCPedMissing:
      tdcPedMissing[ch] = true;
      break;
    case MsgADCPedOr:
      adcPedOr[ch] = true;
      break;
    case MsgADCPedQC:
      adcPedQC[ch] = true;
      break;
    case MsgADCPedMissing:
      adcPedMissing[ch] = true;
      break;
    case MsgOffPed:
      offPed[ch] = true;
      break;
    case MsgPilePed:
      pilePed[ch] = true;
      break;
    case MsgPileTM:
      pileTM[ch] = true;
      break;
    case MsgADCMissingwTDC:
      adcMissingwTDC[ch] = true;
      break;
    case MsgTDCPileEvC:
      tdcPileEvC[ch] = true;
      break;
    case MsgTDCPileEvE:
      tdcPileEvE[ch] = true;
      break;
    case MsgTDCPileM1C:
      tdcPileM1C[ch] = true;
      break;
    case MsgTDCPileM1E:
      tdcPileM1E[ch] = true;
      break;
    case MsgTDCPileM2C:
      tdcPileM2C[ch] = true;
      break;
    case MsgTDCPileM2E:
      tdcPileM2E[ch] = true;
      break;
    case MsgTDCPileM3C:
      tdcPileM3C[ch] = true;
      break;
    case MsgTDCPileM3E:
      tdcPileM3E[ch] = true;
      break;
    case MsgTDCSigE:
      tdcSigE[ch] = true;
      break;
      // End_of_messages
    default:
      LOG(error) << "Not managed info code: " << code;
      return;
  }
  mDecodedInfo.emplace_back((code & 0x03ff) | ((ch & 0x1f) << 10));
}

void RecEventFlat::print() const
{
  printf("%9u.%04u ", ir.orbit, ir.bc);
  printf("nE %2d pos %d nT %2d pos %d  nI %2d pos %d\n", mNE, mFirstE, mNT, mFirstT, mNI, mFirstI);
  printf("%9u.%04u ", ir.orbit, ir.bc);
  printf("Read:");
  for (int ic = 0; ic < NDigiChannels; ic++) {
    if (ic % NChPerModule == 0) {
      if (ic == 0) {
        printf(" %d[", ic / NChPerModule);
      } else {
        printf("] %d[", ic / NChPerModule);
      }
    }
    if (channels & (0x1 << ic)) {
      printf("R");
    } else {
      printf(" ");
    }
  }
  printf("]\n");
  printf("%9u.%04u ", ir.orbit, ir.bc);
  printf("Hits:");
  for (int ic = 0; ic < NDigiChannels; ic++) {
    if (ic % NChPerModule == 0) {
      if (ic == 0) {
        printf(" %d[", ic / NChPerModule);
      } else {
        printf("] %d[", ic / NChPerModule);
      }
    }
    bool is_hit = triggers & (0x1 << ic);
    bool is_trig = mTriggerMask & (0x1 << ic);
    if (is_trig) {
      if (is_hit) {
        printf("T");
      } else {
        printf(".");
      }
    } else {
      if (is_hit) {
        printf("H");
      } else {
        printf(" ");
      }
    }
  }
  printf("]\n");
}

void RecEventFlat::printDecodedMessages() const
{
  const std::array<bool, NChannels>* maps[MsgEnd];
  maps[0] = &genericE;
  maps[1] = &tdcPedQC;
  maps[2] = &tdcPedMissing;
  maps[3] = &adcPedOr;
  maps[4] = &adcPedQC;
  maps[5] = &adcPedMissing;
  maps[6] = &offPed;
  maps[7] = &pilePed;
  maps[8] = &pileTM;
  maps[9] = &adcMissingwTDC;
  maps[10] = &tdcPileEvC;
  maps[11] = &tdcPileEvE;
  maps[12] = &tdcPileM1C;
  maps[13] = &tdcPileM1E;
  maps[14] = &tdcPileM2C;
  maps[15] = &tdcPileM2E;
  maps[16] = &tdcPileM3C;
  maps[17] = &tdcPileM3E;
  maps[18] = &tdcSigE;
  // End_of_messages

  for (int32_t imsg = 0; imsg < MsgEnd; imsg++) {
    TString msg = TString::Format("%-30s:", MsgText[imsg].data());
    if (maps[imsg] == nullptr) {
      continue;
    }
    bool found = false;
    for (int32_t isig = 0; isig < NChannels; isig++) {
      if (maps[imsg]->at(isig) == true) {
        found = true;
        msg += TString::Format(" %s", ChannelNames[isig].data());
      } else {
        msg += "     ";
      }
    }
    if (found) {
      printf("%u.%04u Info %s\n", ir.orbit, ir.bc, msg.Data());
    }
  }
}
