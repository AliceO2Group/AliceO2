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
//  Chip.cxx: structure to store the TOF digits in Chips - useful
// for clusterization purposes
//  ALICEO2
//
#include <cstring>
#include <tuple>

#include <TMath.h>
#include <TObjArray.h>

#include "IOTOFSimulation/Chip.h"

using namespace o2::iotof;

ClassImp(o2::iotof::Chip);

//_______________________________________________________________________
Chip::Chip(Int_t index)
  : mChipIndex(index)
{
}
//_______________________________________________________________________
void Chip::addDigit(UShort_t row, UShort_t col, Int_t charge, double time, o2::MCCompLabel label)
{
  ULong64_t key = Digit::getOrderingKey(mChipIndex, row, col);
  mDigits.emplace(std::make_pair(key, LabeledDigit(mChipIndex, row, col, charge, time, label)));
}
