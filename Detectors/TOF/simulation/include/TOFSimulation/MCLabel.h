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

// Declaration of a transient MC label class for TOF

#ifndef ALICEO2_TOF_MCLABEL_H_
#define ALICEO2_TOF_MCLABEL_H_

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace tof
{
class __attribute__((packed)) MCLabel : public o2::MCCompLabel
{
  // We use __attribute__((packed)), since we have unaligned stored data of MCCompLabel
 private:
  Int_t mTDC = -1;

 public:
  MCLabel() = default;
  MCLabel(Int_t trackID, Int_t eventID, Int_t srcID, Int_t tdc) : o2::MCCompLabel(trackID, eventID, srcID, false), mTDC(tdc) {}
  Int_t getTDC() const { return mTDC; }
};
} // namespace tof
} // namespace o2

#endif
