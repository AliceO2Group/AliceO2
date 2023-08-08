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

#ifndef O2_TRDMCMEVENT_H
#define O2_TRDMCMEVENT_H

#include "CommonDataFormat/InteractionRecord.h"
#include <fairlogger/Logger.h>

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TrapRegInfo.h"
#include "DataFormatsTRD/TrapRegisters.h"
#include "DataFormatsTRD/Digit.h"

#include <string>
#include <map>
#include <unordered_map>
#include <array>
#include <vector>
#include <bitset>
#include <gsl/span>

namespace o2
{
namespace trd
{

/*!
  A class to hold a configuration event from a single mcm.
  All packed registers are simply stored in an array.
  Unpacking and packing is internal to this class.
*/

class MCMEvent
{

 public:
  MCMEvent() = default;
  MCMEvent(int mcmid) { setMCMId(mcmid); }
  ~MCMEvent() = default;
  // get and set the mcmid, this could be extracted from the location of this class in other objects. Its just easier to have it handy here.
  int32_t const getMCMId() { return mMCMId; }
  void setMCMId(const int32_t mcmid) { mMCMId = mcmid; }

  // get and set a specific register. Get/Set the value from its internal compressed format.
  bool setRegister(const uint32_t data, const uint32_t regidx, const TrapRegInfo& trapreg);

  // bool setRegister(const uint32_t data, const uint32_t regidx, const uint32_t base, const uint32_t wordnumber, const uint32_t shift, const uint32_t mask);
  const uint32_t getRegister(const uint32_t regidx, const TrapRegInfo& trapreg) const;

  const uint32_t getvalue(const uint32_t index) { return mRegisterData[index]; };

 private:
  std::array<uint32_t, kTrapRegistersSize> mRegisterData{0}; // a block of mcm register data.
  int32_t mMCMId{-1};                                        // the id of this mcm. -1 to know when it has not been set yet.
  ClassDefNV(MCMEvent, 2);
};

} // namespace trd
} // namespace o2

#endif
