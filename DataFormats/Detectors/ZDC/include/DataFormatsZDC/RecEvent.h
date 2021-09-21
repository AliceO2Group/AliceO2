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
  std::vector<o2::zdc::BCRecData> mRecBC;    /// Interaction record and references to data
  std::vector<o2::zdc::ZDCEnergy> mEnergy;   /// ZDC energy
  std::vector<o2::zdc::ZDCTDCData> mTDCData; /// ZDC TDC
  std::vector<uint16_t> mInfo;               /// Event quality information
  // Add new bunch crossing without data
  inline void addBC(o2::InteractionRecord ir)
  {
    mRecBC.emplace_back(mEnergy.size(), mTDCData.size(), mInfo.size(), ir);
  }
  inline void addBC(o2::InteractionRecord ir, uint32_t channels, uint32_t triggers)
  {
    mRecBC.emplace_back(mEnergy.size(), mTDCData.size(), mInfo.size(), ir);
    mRecBC.back().channels = channels;
    mRecBC.back().triggers = triggers;
  }
  // Add energy
  inline void addEnergy(uint8_t ch, float energy)
  {
    mEnergy.emplace_back(ch, energy);
    mRecBC.back().addEnergy();
  }
  // Add TDC
  inline void addTDC(uint8_t ch, int16_t val, int16_t amp)
  {
    mTDCData.emplace_back(ch, val, amp);
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
    printf("addInfo ch=%u code=%u info=%u 0x%04x\n", ch, code, info, info);
#endif
    mInfo.emplace_back(info);
    mRecBC.back().addInfo();
  }

  void addInfo(const std::array<bool, NChannels>& vec, const uint16_t code)
  {
    // Prepare list of channels interested by this message
    int cnt = 0;
    std::array<int, NChannels> active;
    for (uint8_t ich = 0; ich < NChannels; ich++) {
      if (vec[ich] == true) {
        active[cnt] = ich;
        cnt++;
      }
    }
    if (cnt == 0) {
      return;
    }
#ifdef O2_ZDC_DEBUG
    printf("addInfo(");
    for (uint8_t ich = 0; ich < NChannels; ich++) {
      if (vec[ich] == true) {
        printf("1");
      } else {
        printf("0");
      }
    }
    printf(", code=%u \"%s\") %d active.\n", code, MsgText[code].data(), cnt);
#endif
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
      info = 0x8000;
      for (; i < cnt; i++) {
        info = info | (0x1 << (active[i] - 15));
      }
      addInfo(info);
    }
  }

  void print() const;
  // TODO: remove persitency of this object (here for debugging)
  ClassDefNV(RecEvent, 1);
};

} // namespace zdc
} // namespace o2

#endif
