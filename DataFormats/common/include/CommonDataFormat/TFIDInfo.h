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

#ifndef ALICEO2_TFIDINFO_H
#define ALICEO2_TFIDINFO_H

#include <Rtypes.h>

namespace o2
{
namespace framework
{
class ProcessingContext;
}
namespace dataformats
{
struct TFIDInfo { // helper info to patch DataHeader

  uint32_t firstTForbit = -1U;
  uint32_t tfCounter = -1U;
  uint32_t runNumber = -1U;
  uint32_t startTime = -1U; // same as timeslot
  uint64_t creation = -1UL;

  bool isDummy() { return tfCounter == -1U; }
  void fill(uint32_t firstTForbit_, uint32_t tfCounter_, uint32_t runNumber_, uint32_t startTime_, uint64_t creation_)
  {
    firstTForbit = firstTForbit_;
    tfCounter = tfCounter_;
    runNumber = runNumber_;
    startTime = startTime_;
    creation = creation_;
  }

  ClassDefNV(TFIDInfo, 2);
};
} // namespace dataformats
} // namespace o2

#endif
