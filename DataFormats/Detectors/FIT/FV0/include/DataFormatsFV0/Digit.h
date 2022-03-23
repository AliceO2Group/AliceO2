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

/// \file Digit.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author ruben.shahoyan@cern.ch -> maciej.slupecki@cern.ch

#ifndef _FV0_DIGIT_H_
#define _FV0_DIGIT_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsFV0/ChannelData.h"
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>
#include <iostream>

#include <tuple>

namespace o2
{
namespace fv0
{
class ChannelData;

struct Triggers {
  enum { bitA,
         bitAIn = 1,
         bitC = 1, // alias of bitAIn - needed to be compatible with common-for-FIT raw reader
         bitAOut = 2,
         bitVertex = 2, // alias of bitAOut - needed to be compatible with common-for-FIT raw reader
         bitCen,
         bitSCen,
         bitLaser };
  uint8_t triggersignals = 0; // V0 trigger signals
  int8_t nChanA = 0;          // number of fired channels A side
  int8_t nChanC = 0;          // TODO: MS: unused in FV0
  int32_t amplA = -5000;      // sum amplitude A side
  int32_t amplC = -5000;      // TODO: MS: unused in FV0
  int16_t timeA = -5000;      // average time A side
  int16_t timeC = -5000;      // TODO: MS: unused in FV0
  Triggers() = default;
  Triggers(uint8_t signals, int8_t chanA, int32_t aamplA, int16_t atimeA)
  {
    triggersignals = signals;
    nChanA = chanA;
    amplA = aamplA;
    timeA = atimeA;
  }
  bool getOrA() const { return (triggersignals & (1 << bitA)) != 0; }
  bool getOrAIn() const { return (triggersignals & (1 << bitAIn)) != 0; }
  bool getOrAOut() const { return (triggersignals & (1 << bitAOut)) != 0; }
  bool getCen() const { return (triggersignals & (1 << bitCen)) != 0; }
  bool getSCen() const { return (triggersignals & (1 << bitSCen)) != 0; }
  bool getLaserBit() const { return (triggersignals & (1 << bitLaser)) != 0; }

  // TODO: MS: temporary aliases to keep DigitBlockFIT.h working (treat FV0 as FT0/FDD)
  bool getOrC() const { return getOrAIn(); }
  bool getVertex() const { return getOrAOut(); }

  void setTriggers(Bool_t isA, Bool_t isAIn, Bool_t isAOut, Bool_t isCnt, Bool_t isSCnt, int8_t chanA, int32_t aamplA,
                   int16_t atimeA, Bool_t isLaser = kFALSE)
  {
    triggersignals = (isA << bitA) | (isAIn << bitAIn) | (isAOut << bitAOut) | (isCnt << bitCen) | (isSCnt << bitSCen) | (isLaser << bitLaser);
    nChanA = chanA;
    amplA = aamplA;
    timeA = atimeA;
  }
  void cleanTriggers()
  {
    triggersignals = 0;
    nChanA = 0;
    amplA = -5000;
    timeA = -5000;
  }
  bool operator==(Triggers const& other) const
  {
    return std::tie(triggersignals, nChanA, amplA, timeA) ==
           std::tie(other.triggersignals, other.nChanA, other.amplA, other.timeA);
  }
  void printLog() const;
  ClassDefNV(Triggers, 2);
};

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "FV0TRIGGERINPUT";
  o2::InteractionRecord mIntRecord{}; // bc/orbit of the intpur
  std::bitset<5> mInputs{};           // pattern of inputs.
  DetTrigInput() = default;
  DetTrigInput(const o2::InteractionRecord& iRec, Bool_t isA, Bool_t isAIn, Bool_t isAOut, Bool_t isCnt, Bool_t isSCnt)
    : mIntRecord(iRec),
      mInputs((isA << Triggers::bitA) |
              (isAIn << Triggers::bitAIn) |
              (isAOut << Triggers::bitAOut) |
              (isCnt << Triggers::bitCen) |
              (isSCnt << Triggers::bitSCen))
  {
  }
  ClassDefNV(DetTrigInput, 1);
};

struct Digit {
  static constexpr char sChannelNameDPL[] = "DIGITSBC";
  static constexpr char sDigitName[] = "Digit";
  static constexpr char sDigitBranchName[] = "FV0DigitBC";
  /// we are going to refer to at most 48 channels, so 6 bits for the number of channels and 26 for the reference
  o2::dataformats::RangeRefComp<6> ref;
  Triggers mTriggers{}; // pattern of triggers  in this BC

  o2::InteractionRecord mIntRecord{}; // Interaction record (orbit, bc)
  int mEventID = 0;
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
  DetTrigInput makeTrgInput() const { return DetTrigInput{mIntRecord, mTriggers.getOrA(), mTriggers.getOrAIn(), mTriggers.getOrAOut(), mTriggers.getCen(), mTriggers.getSCen()}; }
  void fillTrgInputVec(std::vector<DetTrigInput>& vecTrgInput) const
  {
    vecTrgInput.emplace_back(mIntRecord, mTriggers.getOrA(), mTriggers.getOrAIn(), mTriggers.getOrAOut(), mTriggers.getCen(), mTriggers.getSCen());
  }
  void printStream(std::ostream& stream) const;
  void setTriggers(Triggers trig) { mTriggers = trig; };
  bool operator==(const Digit& other) const
  {
    return std::tie(ref, mTriggers, mIntRecord) == std::tie(other.ref, other.mTriggers, other.mIntRecord);
  }
  void printLog() const;
  ClassDefNV(Digit, 2);
};

// For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  static constexpr char sChannelNameDPL[] = "DIGITSTRGEXT";
  static constexpr char sDigitName[] = "TriggersExt";
  static constexpr char sDigitBranchName[] = "FV0DIGITSTRGEXT";
  TriggersExt(std::array<uint32_t, 20> triggerWords) : mTriggerWords(triggerWords) {}
  TriggersExt() = default;
  o2::InteractionRecord mIntRecord{};
  void setTrgWord(uint32_t trgWord, std::size_t pos) { mTriggerWords[pos] = trgWord; }
  std::array<uint32_t, 20> mTriggerWords{};
  void printLog() const;
  ClassDefNV(TriggersExt, 1);
};
} // namespace fv0
} // namespace o2

#endif
