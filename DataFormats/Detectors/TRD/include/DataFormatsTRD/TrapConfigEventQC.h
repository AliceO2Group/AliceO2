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

#ifndef O2_TRDTRAPCONFIGEVENTQC_H
#define O2_TRDTRAPCONFIGEVENTQC_H

#include <string>
#include <vector>
#include "TObject.h"

namespace o2
{
namespace trd
{

struct TrapConfigEventQCItem {
  uint32_t mMCMId;                     //!< index of this mcm, this removes the need for an index.
  uint16_t mRegistersSuccessfullyRead; //!< How many registers an MCM read, in range [0-433];
  uint16_t mMissingRegisters;          //!< [0-433];  433 - the above number
  uint16_t mStartRegister;             //!< [0-433];  which register this mcm started on
  uint16_t mEndRegister;               //!< [0-433];  which register this mcm finished on
  uint8_t mParseStatus;                //!< 0x01 .. 0x4f bit pattern
  ClassDefNV(TrapConfigEventQCItem, 1);
};

class TrapConfigEventQC
{
  // This is an object that sent to qc for each timeframe that contains any config events.
  // accumulation and frequency is done on the qc side, this implies that qc receives all of these objects not a sampling.
  // in a single timeframe there can never be more than a single mcm sending data.
 public:
  TrapConfigEventQC() = default;
  TrapConfigEventQC(uint16_t mcmid) { mQCData.emplace_back(TrapConfigEventQCItem()); }
  ~TrapConfigEventQC() = default;
  // get and set methods of the underlying entries in the mQCData vector.
  uint32_t getReadRegisters(int mcmid) { return mQCData[mcmid].mRegistersSuccessfullyRead; }
  uint32_t getMissedRegisters(int mcmid) { return mQCData[mcmid].mMissingRegisters; }
  uint32_t getStartRegister(int mcmid) { return mQCData[mcmid].mStartRegister; }
  uint32_t getStopRegister(int mcmid) { return mQCData[mcmid].mEndRegister; }
  uint32_t getParseStatus(int mcmid) { return mQCData[mcmid].mParseStatus; }
  void setReadRegisters(int mcmid, int regread) { mQCData[mcmid].mRegistersSuccessfullyRead = regread; }
  void setMissedRegisters(int mcmid, int missed) { mQCData[mcmid].mMissingRegisters = missed; }
  void setStartRegister(int mcmid, int start) { mQCData[mcmid].mStartRegister = start; }
  void setStopRegister(int mcmid, int end) { mQCData[mcmid].mEndRegister = end; }
  void setParseStatus(int mcmid, int parsing) { mQCData[mcmid].mParseStatus = parsing; }

  void addMCM(int mcmid) { mQCData.emplace_back(TrapConfigEventQCItem()); }
  void clear() { mQCData.clear(); }

 private:
  std::vector<o2::trd::TrapConfigEventQCItem> mQCData; //!< one item per mcm for the given timeframe
  ClassDefNV(TrapConfigEventQC, 1);
};

} // namespace trd
} // namespace o2

#endif
