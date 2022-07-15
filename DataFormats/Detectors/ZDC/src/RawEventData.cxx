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
// DataFormats/Detectors/ZDC/src/RawEventData.cxx
#include "DataFormatsZDC/RawEventData.h"

using namespace o2::zdc;

// ClassImp(EventData);

//______________________________________________________________________________
void EventChData::reset()
{
  static constexpr int payloadSize = NWPerGBTW * sizeof(UInt_t);
  memset((void*)&w[0][0], 0, payloadSize);
}

//______________________________________________________________________________
void EventData::print() const
{
  for (Int_t im = 0; im < o2::zdc::NModules; im++) {
    for (Int_t ic = 0; ic < o2::zdc::NChPerModule; ic++) {
      if (data[im][ic].f.fixed_0 == Id_w0 && data[im][ic].f.fixed_1 == Id_w1 && data[im][ic].f.fixed_2 == Id_w2) {
        // Not empty event
        auto f = data[im][ic].f;
        // Word 0
        printf("%04x %08x %08x ", data[im][ic].w[0][2], data[im][ic].w[0][1], data[im][ic].w[0][0]);
        printf("orbit %-9u bc %-4u hits %-4u offset %+6i Board %2u Ch %1u\n", f.orbit, f.bc, f.hits, f.offset - 32768, f.ch, f.board);
        // Word 1
        printf("%04x %08x %08x ", data[im][ic].w[1][2], data[im][ic].w[1][1], data[im][ic].w[1][0]);
        printf("     %s %s %s %s 0-5 ", f.Alice_0 ? "A0" : "  ", f.Alice_1 ? "A1" : "  ", f.Alice_2 ? "A2" : "  ", f.Alice_3 ? "A3" : "  ");
        printf(" %5d %5d %5d %5d %5d %5d EC=%u\n", f.s00, f.s01, f.s02, f.s03, f.s04, f.s05, f.error);
        // Word 2
        printf("%04x %08x %08x ", data[im][ic].w[2][2], data[im][ic].w[2][1], data[im][ic].w[2][0]);
        printf("%s %s %s %s %s %s 6-b ", f.Hit ? "H" : " ", f.Auto_m ? "TM" : "  ", f.Auto_0 ? "T0" : "  ", f.Auto_1 ? "T1" : "  ", f.Auto_2 ? "T2" : "  ", f.Auto_3 ? "T3" : "  ");
        printf(" %5d %5d %5d %5d %5d %5d\n", f.s06, f.s07, f.s08, f.s09, f.s10, f.s11);
      } else if (data[im][ic].f.fixed_0 == 0 && data[im][ic].f.fixed_1 == 0 && data[im][ic].f.fixed_2 == 0) {
        // Empty channel
      } else {
        // Wrong data format. Insert warning.
      }
    }
  }
}

//______________________________________________________________________________
void EventData::reset()
{
  static constexpr int payloadSize = NModules * NChPerModule * NWPerGBTW * sizeof(UInt_t);
  memset((void*)&data[0][0], 0, payloadSize);
}
