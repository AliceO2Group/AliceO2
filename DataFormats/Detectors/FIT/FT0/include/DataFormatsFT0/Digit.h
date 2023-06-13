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

// \file Digit.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author Alla.Maevskaya@cern.ch

#ifndef _FT0_DIGIT_H_
#define _FT0_DIGIT_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFIT/Triggers.h"
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>
#include <iostream>
#include <tuple>

namespace o2
{
namespace ft0
{
class ChannelData;
using Triggers = o2::fit::Triggers;

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "FT0TRIGGERINPUT";
  o2::InteractionRecord mIntRecord{}; // bc/orbit of the intpur
  std::bitset<5> mInputs{};           // pattern of inputs.
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
  bool isVertex() const { return mInputs.test(Triggers::bitVertex); }
  ClassDefNV(DetTrigInput, 1);
};

struct Digit {
  static constexpr char sChannelNameDPL[] = "DIGITSBC";
  static constexpr char sDigitName[] = "Digit";
  static constexpr char sDigitBranchName[] = "FT0DIGITSBC";
  o2::dataformats::RangeReference<int, int> ref{};
  Triggers mTriggers{};               // pattern of triggers  in this BC
  uint8_t mEventStatus = 0;           //Status of event from FT0, such as Pileup , etc
  o2::InteractionRecord mIntRecord{}; // Interaction record (orbit, bc)
  int mEventID = 0;
  enum EEventStatus {
    kPileup
  };
  Digit() = default;
  Digit(int first, int ne, const o2::InteractionRecord& iRec, const Triggers& chTrig, int event)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = chTrig;
    mEventID = event;
  }
  typedef DetTrigInput DetTrigInput_t;
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  Triggers getTriggers() const { return mTriggers; }
  int getEventID() const { return mEventID; }
  const o2::InteractionRecord& getIntRecord() const { return mIntRecord; };
  void setIntRecord(const o2::InteractionRecord& intRec) { mIntRecord = intRec; }
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  DetTrigInput makeTrgInput() const { return DetTrigInput{mIntRecord, mTriggers.getOrA(), mTriggers.getOrC(), mTriggers.getVertex(), mTriggers.getCen(), mTriggers.getSCen()}; }
  void fillTrgInputVec(std::vector<DetTrigInput>& vecTrgInput) const
  {
    vecTrgInput.emplace_back(mIntRecord, mTriggers.getOrA(), mTriggers.getOrC(), mTriggers.getVertex(), mTriggers.getCen(), mTriggers.getSCen());
  }
  void printStream(std::ostream& stream) const;
  void setTriggers(Triggers trig) { mTriggers = trig; };
  void setEventStatus(uint8_t stat) { mEventStatus = stat; };
  void setStatusFlag(EEventStatus bit, bool value) { mEventStatus |= (value << bit); };
  bool getStatusFlag(EEventStatus bit) const { return bool(mEventStatus << bit); }
  uint8_t getEventStatusWord() const { return mEventStatus; }
  bool operator==(const Digit& other) const
  {
    return std::tie(ref, mTriggers, mIntRecord) == std::tie(other.ref, other.mTriggers, other.mIntRecord);
  }
  void printLog() const;
  ClassDefNV(Digit, 7);
};

//For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  static constexpr char sChannelNameDPL[] = "DIGITSTRGEXT";
  static constexpr char sDigitName[] = "TriggersExt";
  static constexpr char sDigitBranchName[] = "FT0DIGITSTRGEXT";
  TriggersExt(std::array<uint32_t, 20> triggerWords) : mTriggerWords(triggerWords) {}
  TriggersExt() = default;
  o2::InteractionRecord mIntRecord{};
  void setTrgWord(uint32_t trgWord, std::size_t pos) { mTriggerWords[pos] = trgWord; }
  std::array<uint32_t, 20> mTriggerWords{};
  void printLog() const;
  ClassDefNV(TriggersExt, 2);
};

} // namespace ft0
} // namespace o2

#endif
