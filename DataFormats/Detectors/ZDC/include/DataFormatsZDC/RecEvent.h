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

#ifndef _ZDC_RECEVENT_H_
#define _ZDC_RECEVENT_H_

#include "Framework/Logger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "DataFormatsZDC/ZDCWaveform.h"
#include "DataFormatsZDC/RecEventAux.h"
#include "ZDCBase/Constants.h"
#include "MathUtils/Cartesian.h"
#include <Rtypes.h>
#include <array>
#include <vector>
#include <map>

/// \file RecEvent.h
/// \brief Class to describe reconstructed ZDC event (single BC with signal in one of detectors)
/// \author pietro.cortese@cern.ch, ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{
struct RecEvent {
  std::vector<o2::zdc::BCRecData> mRecBC;      /// Interaction record and references to data
  std::vector<o2::zdc::ZDCEnergy> mEnergy;     /// ZDC energy
  std::vector<o2::zdc::ZDCTDCData> mTDCData;   /// ZDC TDC
  std::vector<uint16_t> mInfo;                 /// Event quality information
  std::vector<o2::zdc::ZDCWaveform> mWaveform; /// ZDC waveform

  // Add new bunch crossing
  inline void addBC(const RecEventAux& reca)
  {
#ifdef O2_ZDC_DEBUG
    printf("addBC %u.%-4u En_start:%-2lu TDC_start:%-2lu Info_start:%-2lu ch=0x%08x tr=0x%08x\n", reca.ir.orbit, reca.ir.bc, mEnergy.size(), mTDCData.size(), mInfo.size(), reca.channels, reca.triggers);
#endif
    mRecBC.emplace_back(mEnergy.size(), mTDCData.size(), mInfo.size(), mWaveform.size(), reca.ir);
    mRecBC.back().channels = reca.channels;
    mRecBC.back().triggers = reca.triggers;
  }

  // Add energy
  inline void addEnergy(uint8_t ch, float energy)
  {
#ifdef O2_ZDC_DEBUG
    printf("ch:%-2u [%s] Energy %9.2f\n", ch, ChannelNames[ch].data(), energy);
#endif
    mEnergy.emplace_back(ch, energy);
    mRecBC.back().addEnergy();
  }

  // Add TDC - int16_t
  inline void addTDC(uint8_t ch, int16_t val, int16_t amp, bool isbeg = false, bool isend = false)
  {
#ifdef O2_ZDC_DEBUG
    printf("ch:%-2u [%s] TDC %4d Amp. %4d%s%s\n", ch, ChannelNames[TDCSignal[ch]].data(), val, amp, isbeg ? " B" : "", isend ? " E" : "");
#endif
    mTDCData.emplace_back(ch, val, amp, isbeg, isend);
    mRecBC.back().addTDC();
  }

  // Add TDC - float
  inline void addTDC(uint8_t ch, float val, float amp, bool isbeg = false, bool isend = false)
  {
#ifdef O2_ZDC_DEBUG
    printf("ch:%-2u [%s] TDC %6.0f Amp. %6.0f%s%s\n", ch, ChannelNames[TDCSignal[ch]].data(), val, amp, isbeg ? " B" : "", isend ? " E" : "");
#endif
    mTDCData.emplace_back(ch, val, amp, isbeg, isend);
    mRecBC.back().addTDC();
  }

  // Add event information
  inline void addInfo(uint16_t info)
  {
#ifdef O2_ZDC_DEBUG
    printf("addInfo info=%u 0x%04x\n", info, info);
#endif
    mInfo.emplace_back(info);
    mRecBC.back().addInfo();
  }

  inline void addInfo(uint8_t ch, uint16_t code)
  {
    if (ch >= NChannels && ch != 0x1f) {
      LOGF(error, "Adding info (0x%x) for not existent channel %u", code, ch);
      return;
    }
    uint16_t info = (code & 0x03ff) | ((ch & 0x1f) << 10);
#ifdef O2_ZDC_DEBUG
    printf("addInfo ch=%u code=%u \"%s\" info=%u 0x%04x\n", ch, code, MsgText[code].data(), info, info);
#endif
    mInfo.emplace_back(info);
    mRecBC.back().addInfo();
  }

  uint32_t addInfo(const RecEventAux& reca, const std::array<bool, NChannels>& vec, const uint16_t code);
  uint32_t addInfos(const RecEventAux& reca);

  // Add waveform
  inline void addWaveform(uint8_t ch, std::vector<float>& wave)
  {
#ifdef O2_ZDC_DEBUG
    printf("ch:%-2u [%s] Waveform\n", ch, ChannelNames[ch].data());
#endif
    if (wave.size() == NIS) {
      mWaveform.emplace_back(ch, wave.data());
      mRecBC.back().addWaveform();
    } else {
      LOG(error) << __func__ << ": ch " << int(ch) << " inconsistent waveform size " << wave.size();
    }
  }

  void print() const;
  // TODO: remove persitency of this object (here for debugging)
  ClassDefNV(RecEvent, 2);
};

} // namespace zdc
} // namespace o2

#endif
