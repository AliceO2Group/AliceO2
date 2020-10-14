// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//

#ifndef ALICEO2_ZDC_RAWEVENTDATA_H_
#define ALICEO2_ZDC_RAWEVENTDATA_H_
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <iostream>
#include <gsl/span>

// #include "Headers/RAWDataHeader.h"
// #include <Framework/Logger.h>
// #include <utility>
// #include <cstring>

/// \file BCData.h
/// \brief Container of ZDC raw data
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

constexpr unsigned short id_w0 = 0x0;
constexpr unsigned short id_w1 = 0x1;
constexpr unsigned short id_w2 = 0x2;

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

  // Second GBT word
  unsigned fixed_1 : 2;
  unsigned error : 2;
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
  unsigned empty_1 : 16;

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
  unsigned empty_2 : 16;
};

struct __attribute__((__packed__)) ChannelDataV0W0 {
  unsigned fixed_0 : 2;
  unsigned board : 4;
  unsigned ch : 2;
  unsigned offset : 16;
  unsigned hits : 12;
  unsigned bc : 12;
  UInt_t orbit;
  unsigned empty_0 : 16;
};

struct __attribute__((__packed__)) ChannelDataV0W1 {
  unsigned fixed_1 : 2;
  unsigned error : 2;
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
  unsigned empty_1 : 16;
};

struct __attribute__((__packed__)) ChannelDataV0W2 {
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
  unsigned empty_2 : 16;
};

struct EventData {
  union {
    UInt_t w[o2::zdc::NWPerBc][3];
    struct ChannelDataV0 f;
  } data[o2::zdc::NModules][o2::zdc::NChPerModule];
  void print() const;
  ClassDefNV(EventData, 1);
};

} // namespace zdc
} // namespace o2
#endif
