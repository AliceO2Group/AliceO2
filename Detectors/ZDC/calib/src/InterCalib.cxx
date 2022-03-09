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

#include <TROOT.h>
#include <TPad.h>
#include <TString.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include "ZDCCalib/InterCalib.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int InterCalib::init(){
  return 0;
}
int InterCalib::run(){
  o2::zdc::RecEventFlat ev;
  ev.init(RecBCPtr, EnergyPtr, TDCDataPtr, InfoPtr);
  while (ev.next()) {
    int printed = 0;
      if (ev.getNInfo() > 0) {
        auto& decodedInfo = ev.getDecodedInfo();
        for (uint16_t info : decodedInfo) {
          uint8_t ch = (info >> 10) & 0x1f;
          uint16_t code = info & 0x03ff;;
          hmsg->Fill(ch, code);
        }
        ev.print();
        printed = 1;
      }
      if (ev.getNEnergy() > 0 && ev.mCurB.triggers == 0) {
        printf("%9u.%04u Untriggered bunch\n", ev.mCurB.ir.orbit, ev.mCurB.ir.bc);
        if (printed == 0) {
          ev.print();
        }
      }
      heznac->Fill(ev.EZNAC());
      auto tdcid = o2::zdc::TDCZNAC;
      auto nhit = ev.NtdcV(tdcid);
      if (ev.NtdcA(tdcid) != nhit) {
        fprintf(stderr, "Mismatch in TDC data\n");
        continue;
      }
      if (nhit > 0) {
        double bc_d = uint32_t(ev.ir.bc / 100);
        double bc_m = uint32_t(ev.ir.bc % 100);
        hbznac->Fill(bc_m, -bc_d);
        for (int ihit = 0; ihit < nhit; ihit++) {
          htznac->Fill(ev.tdcV(tdcid, ihit), ev.tdcA(tdcid, ihit));
        }
      }
    }
  }
  return 0;
}
