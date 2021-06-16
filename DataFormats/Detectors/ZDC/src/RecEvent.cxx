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

#include "DataFormatsZDC/RecEvent.h"

using namespace o2::zdc;

void RecEvent::print() const
{
  for (auto bcdata : mRecBC) {
    bcdata.ir.print();
    int fe, ne, ft, nt, fi, ni;
    bcdata.getRef(fe, ne, ft, nt, fi, ni);
    for (int ie = 0; ie < ne; ie++) {
      mEnergy[fe + ie].print();
    }
    for (int it = 0; it < nt; it++) {
      mTDCData[ft + it].print();
    }
    // TODO: event info
  }
}
