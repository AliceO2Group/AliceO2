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

#ifndef ALICEO2_FDD_DIGIT_H
#define ALICEO2_FDD_DIGIT_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFIT/Triggers.h"
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>
#include <iostream>
#include <tuple>

namespace o2
{
namespace fdd
{
class ChannelData;
using Triggers = o2::fit::Triggers;

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "FDDTRIGGERINPUT";
  o2::InteractionRecord mIntRecord; // bc/orbit of the intpur
  std::bitset<5> mInputs;           // pattern of inputs.
  DetTrigInput() = default;
  DetTrigInput(const o2::InteractionRecord& iRec, Bool_t isA, Bool_t isC, Bool_t isVrtx, Bool_t isCnt, Bool_t isSCnt)
    : mIntRecord(iRec),
      mInputs((isA << Triggers::bitA) |
              (isC << Triggers::bitC) |
              (isVrtx << Triggers::bitVertex) |
              (isCnt << Triggers::bitCen) |
              (isSCnt << Triggers::bitSCen))
  {
  }
  ClassDefNV(DetTrigInput, 1);
};

struct Digit {
  static constexpr char sChannelNameDPL[] = "DIGITSBC";
  static constexpr char sDigitName[] = "Digit";
  static constexpr char sDigitBranchName[] = "FDDDigit";
  o2::dataformats::RangeReference<int, int> ref{};
  Triggers mTriggers;               // pattern of triggers  in this BC
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  Digit() = default;
  Digit(int first, int ne, o2::InteractionRecord iRec, Triggers chTrig) : ref(first, ne), mIntRecord(iRec), mTriggers(chTrig) {}
  typedef DetTrigInput DetTrigInput_t;
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  o2::InteractionRecord getIntRecord() const { return mIntRecord; };
  void setIntRecord(const o2::InteractionRecord& intRec) { mIntRecord = intRec; }
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
  {
    return ref.getEntries() ? gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const ChannelData>();
  }
  DetTrigInput makeTrgInput() const { return DetTrigInput{mIntRecord, mTriggers.getOrA(), mTriggers.getOrC(), mTriggers.getVertex(), mTriggers.getCen(), mTriggers.getSCen()}; }
  void fillTrgInputVec(std::vector<DetTrigInput>& vecTrgInput) const
  {
    vecTrgInput.emplace_back(mIntRecord, mTriggers.getOrA(), mTriggers.getOrC(), mTriggers.getVertex(), mTriggers.getCen(), mTriggers.getSCen());
  }
  bool operator==(const Digit& other) const
  {
    return std::tie(ref, mTriggers, mIntRecord) == std::tie(other.ref, other.mTriggers, other.mIntRecord);
  }
  void printLog() const
  {
    LOG(info) << "______________DIGIT DATA____________";
    LOG(info) << "BC: " << mIntRecord.bc << "| ORBIT: " << mIntRecord.orbit;
    LOG(info) << "Ref first: " << ref.getFirstEntry() << "| Ref entries: " << ref.getEntries();
    mTriggers.printLog();
  }
  ClassDefNV(Digit, 4);
};
//For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  static constexpr char sChannelNameDPL[] = "DIGITSTRGEXT";
  static constexpr char sDigitName[] = "TriggersExt";
  static constexpr char sDigitBranchName[] = "FDDDigitTrgExt";
  TriggersExt(std::array<uint32_t, 20> triggerWords) : mTriggerWords(triggerWords) {}
  TriggersExt() = default;
  o2::InteractionRecord mIntRecord;
  void setTrgWord(uint32_t trgWord, std::size_t pos) { mTriggerWords[pos] = trgWord; }
  std::array<uint32_t, 20> mTriggerWords;
  void printLog() const
  {
    LOG(info) << "______________EXTENDED TRIGGERS____________";
    LOG(info) << "BC: " << mIntRecord.bc << "| ORBIT: " << mIntRecord.orbit;
    for (int i = 0; i < 20; i++) {
      LOG(info) << "N: " << i + 1 << " | TRG: " << mTriggerWords[i];
    }
  }
  ClassDefNV(TriggersExt, 1);
};

} // namespace fdd
} // namespace o2
#endif
