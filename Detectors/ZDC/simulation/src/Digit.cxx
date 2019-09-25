// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCSimulation/Digit.h"
#include "ZDCBase/Constants.h"

/// \file Digit.cxx

using namespace o2::zdc;

void Digit::print() const
{

  const char* nameDet[] = {"ZNA", "ZPA", "ZEM", "ZNC", "ZPC"};
  const char* namesZH[] = {"Com", "Ch1", "Ch2", "Ch3", "Ch4", "Sum"};
  const char* namesZE[] = {"Ch1", "Ch2"};

  mIntRecord.print();

  for (int id = 0; id < 5; id++) {
    int idet = id + 1;
    const char** nameChan = idet == ZEM ? namesZE : namesZH;
    int nChan = idet == ZEM ? NChannelsZEM : NChannelsZN;

    for (int ic = 0; ic < nChan; ic++) {
      for (int slot = 0; slot < 4; slot++) {
        const auto& slotData = getChannel(toChannel(idet, ic), slot);
        printf("%3s:%-3s/%d |", nameDet[id], nameChan[ic], slot);
        for (int ib = 0; ib < NTimeBinsPerBC; ib++) {
          printf("%5d ", slotData.data[ib]);
        }
        printf("\n");
      }
    }
  }
}
