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

#ifndef ZDC_TDC_DATA_H
#define ZDC_TDC_DATA_H

#include "Framework/Logger.h"
#include "ZDCBase/Constants.h"
#include <array>
#include <TMath.h>
#include <Rtypes.h>

/// \file ZDCTDCData.h
/// \brief Container class to store a TDC hit in a ZDC channel
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct ZDCTDCDataErr {

  static uint32_t mErrVal[NTDCChannels]; // Errors in encoding TDC values
  static uint32_t mErrId;                // Errors with TDC Id

  static void print()
  {
    if (mErrId > 0) {
      LOG(error) << "TDCId was out of range #times = " << mErrId;
    }
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      if (mErrVal[itdc] > 0) {
        LOG(error) << "TDCVal itdc=" << itdc << " " << ChannelNames[TDCSignal[itdc]] << " was out of range #times = " << mErrVal[itdc];
      }
    }
  }
};

struct ZDCTDCData {

  uint8_t id = 0xff; // channel ID
  int16_t val = 0;   // tdc value
  float amp = 0;     // tdc amplitude

  ZDCTDCData() = default;

  ZDCTDCData(uint8_t ida, int16_t vala, float ampa, bool isbeg = false, bool isend = false)
  {
    // TDC value and amplitude are encoded externally
    id = ida < NTDCChannels ? ida : 0xf;
    id = id | (isbeg ? 0x80 : 0x00);
    id = id | (isend ? 0x40 : 0x00);

    if (ida < NTDCChannels) {
      val = vala;
      amp = ampa;
    } else {
      val = kMaxShort;
      amp = FInfty;
#ifdef O2_ZDC_DEBUG
      LOG(error) << __func__ << "TDC Id = " << int(ida) << " is out of range";
#endif
      ZDCTDCDataErr::mErrId++;
    }
  }

  ZDCTDCData(uint8_t ida, float vala, float ampa, bool isbeg = false, bool isend = false)
  {
    // TDC value and amplitude are encoded externally but argument is float
    id = ida < NTDCChannels ? ida : 0xf;
    id = id | (isbeg ? 0x80 : 0x00);
    id = id | (isend ? 0x40 : 0x00);

    if (ida >= NTDCChannels) {
      val = kMaxShort;
      amp = FInfty;
#ifdef O2_ZDC_DEBUG
      LOG(error) << __func__ << "TDC Id = " << int(ida) << " is out of range";
#endif
      ZDCTDCDataErr::mErrId++;
      return;
    }

    auto TDCVal = std::nearbyint(vala);

    if (TDCVal < kMinShort) {
      int itdc = int(id);
#ifdef O2_ZDC_DEBUG
      LOG(error) << __func__ << "TDCVal itdc=" << itdc << " " << ChannelNames[TDCSignal[itdc]] << " = " << TDCVal << " is out of range";
#endif
      ZDCTDCDataErr::mErrVal[itdc]++;
      TDCVal = kMinShort;
    }

    if (TDCVal > kMaxShort) {
      int itdc = int(ida);
#ifdef O2_ZDC_DEBUG
      LOG(error) << __func__ << "TDCVal itdc=" << itdc << " " << ChannelNames[TDCSignal[itdc]] << " = " << TDCVal << " is out of range";
#endif
      ZDCTDCDataErr::mErrVal[itdc]++;
      TDCVal = kMaxShort;
    }

    val = TDCVal;
    amp = ampa;
  }

  inline float amplitude() const
  {
    return amp;
  }

  inline float value() const
  {
    // Return decoded value (ns)
    return FTDCVal * val;
  }

  inline int ch() const
  {
    return (id & 0x0f);
  }

  inline bool isBeg() const
  {
    return id & 0x80 ? true : false;
  }

  inline bool isEnd() const
  {
    return id & 0x40 ? true : false;
  }

  void print() const;

  ClassDefNV(ZDCTDCData, 2);
};
} // namespace zdc
} // namespace o2

#endif
