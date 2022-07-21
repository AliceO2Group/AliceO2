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
//
// DataFormats/Detectors/ZDC/include/DataFormatsZDC/RawEventData.h

#ifndef ALICEO2_ZDC_RAWEVENTDATA_H_
#define ALICEO2_ZDC_RAWEVENTDATA_H_
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <iostream>
#include <gsl/span>

/// \file RawEventData.h
/// \brief Container of ZDC raw data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

constexpr unsigned short Id_w0 = 0x0;
constexpr unsigned short Id_w1 = 0x1;
constexpr unsigned short Id_w2 = 0x2;
constexpr unsigned short Id_wn = 0x3;
constexpr int NWPerGBTW = 4;

struct __attribute__((__packed__)) ChannelDataV0 {
  // First GBT word
  unsigned fixed_0 : 2;
  unsigned board : 4;
  unsigned ch : 2;
  unsigned offset : 16;
  unsigned hits : 12;
  unsigned bc : 12;
  unsigned orbit : 32;
  unsigned empty_0 : 16;
  unsigned empty_1 : 32;

  // Second GBT word
  unsigned fixed_1 : 2;
  unsigned dLoss : 1;
  unsigned error : 1;
  unsigned Alice_0 : 1;
  unsigned Alice_1 : 1;
  unsigned Alice_2 : 1;
  unsigned Alice_3 : 1;
  unsigned s00 : 12;
  unsigned s01 : 12;
  unsigned s02 : 12;
  unsigned s03 : 12;
  unsigned s04 : 12;
  unsigned s05 : 12;
  unsigned empty_2 : 16;
  unsigned empty_3 : 32;

  // Third GBT word
  unsigned fixed_2 : 2;
  unsigned Hit : 1;
  unsigned Auto_m : 1;
  unsigned Auto_0 : 1;
  unsigned Auto_1 : 1;
  unsigned Auto_2 : 1;
  unsigned Auto_3 : 1;
  unsigned s06 : 12;
  unsigned s07 : 12;
  unsigned s08 : 12;
  unsigned s09 : 12;
  unsigned s10 : 12;
  unsigned s11 : 12;
  unsigned empty_4 : 16;
  unsigned empty_5 : 32;
};

union EventChData {
  UInt_t w[NWPerBc][NWPerGBTW];
  struct ChannelDataV0 f;
  void reset();
};

struct EventData {
  EventChData data[NModules][NChPerModule] = {0};
  void print() const;
  void reset();
  ClassDefNV(EventData, 1);
};

} // namespace zdc
} // namespace o2
#endif
