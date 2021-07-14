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

#include "DataFormatsZDC/RecEventFlat.h"

using namespace o2::zdc;

void RecEventFlat::init(std::vector<o2::zdc::BCRecData>* RecBC, std::vector<o2::zdc::ZDCEnergy>* Energy, std::vector<o2::zdc::ZDCTDCData>* TDCData, std::vector<uint16_t>* Info)
{
  mRecBC = RecBC;
  mEnergy = Energy;
  mTDCData = TDCData;
  mInfo = Info;
  mEntry = 0;
  mNEntries = mRecBC->size();
  printf("There are %u entries\n",mNEntries);
}

int RecEventFlat::next()
{
  if (mEntry >= mNEntries) {
    return 0;
  }
  ezdc.clear();
  auto& curb = mRecBC->at(mEntry);
  ir = curb.ir;
  channels = curb.channels;
  triggers = curb.triggers;
  int istart = curb.refe.getFirstEntry();
  int istop = istart + curb.refe.getEntries();
  for (int i = istart; i < istop; i++) {
    auto& myenergy = mEnergy->at(i);
    ezdc[myenergy.ch()] = myenergy.energy();
  }

  return mEntry;
}

void RecEventFlat::print() const
{
  ir.print();
}
