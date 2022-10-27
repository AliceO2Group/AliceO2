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

#ifndef O2_ZDC_BC_DATA_H_
#define O2_ZDC_BC_DATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <gsl/span>

/// \file BCData.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{
class ChannelData;

struct __attribute__((__packed__)) ModuleTriggerMap {
  unsigned Alice_0 : 1;
  unsigned Alice_1 : 1;
  unsigned Alice_2 : 1;
  unsigned Alice_3 : 1;
  unsigned Auto_m : 1;
  unsigned Auto_0 : 1;
  unsigned Auto_1 : 1;
  unsigned Auto_2 : 1;
  unsigned Auto_3 : 1;
  unsigned AliceErr : 1;
  unsigned AutoErr : 1;
  unsigned empty : 5;
};

union ModuleTriggerMapData {
  uint16_t w;
  struct ModuleTriggerMap f;
  void reset();
};

struct BCData {
  /// we are going to refer to at most 26 channels, so 5 bits for the NChannels and 27 for the reference
  /// 220803 we are transmitting 32 ch: need to increase to 6 bit
  o2::dataformats::RangeRefComp<6> ref;
  o2::InteractionRecord ir;
  std::array<uint16_t, NModules> moduleTriggers{};
  uint32_t channels = 0;    // pattern of channels it refers to
  uint32_t triggers = 0;    // pattern of triggered channels (not necessarily stored) in this BC
  uint8_t ext_triggers = 0; // pattern of ALICE triggers

  BCData() = default;
  BCData(int first, int ne, o2::InteractionRecord iRec, uint32_t chSto, uint32_t chTrig, uint8_t extTrig)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    ir = iRec;
    channels = chSto;
    triggers = chTrig;
    ext_triggers = extTrig;
  }
  BCData(const BCData&) = default;

  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  void print(uint32_t triggerMask = 0, int diff = 0) const;

  ClassDefNV(BCData, 3);
};
} // namespace zdc
} // namespace o2

#endif
