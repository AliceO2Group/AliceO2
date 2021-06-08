// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

struct Triggers {
  enum { bitA,
         bitC,
         bitVertex,
         bitCen,
         bitSCen,
         bitLaser };
  uint8_t triggersignals = 0; // T0 trigger signals
  int8_t nChanA = 0;          // number of fired channels A side
  int8_t nChanC = 0;          // number of fired channels A side
  int32_t amplA = -5000;      // sum amplitude A side
  int32_t amplC = -5000;      // sum amplitude C side
  int16_t timeA = -5000;      // average time A side
  int16_t timeC = -5000;      // average time C side
  uint8_t eventFlags = 0;     // event conditions
  Triggers() = default;
  Triggers(uint8_t signals, int8_t chanA, int8_t chanC, int32_t aamplA, int32_t aamplC, int16_t atimeA, int16_t atimeC)
  {
    triggersignals = signals;
    nChanA = chanA;
    nChanC = chanC;
    amplA = aamplA;
    amplC = aamplC;
    timeA = atimeA;
    timeC = atimeC;
  }
  bool getOrA() const { return (triggersignals & (1 << bitA)) != 0; }
  bool getOrC() const { return (triggersignals & (1 << bitC)) != 0; }
  bool getVertex() const { return (triggersignals & (1 << bitVertex)) != 0; }
  bool getCen() const { return (triggersignals & (1 << bitCen)) != 0; }
  bool getSCen() const { return (triggersignals & (1 << bitSCen)) != 0; }
  bool getLaserBit() const { return (triggersignals & (1 << bitLaser)) != 0; }

  void setTriggers(Bool_t isA, Bool_t isC, Bool_t isVrtx, Bool_t isCnt, Bool_t isSCnt, int8_t chanA, int8_t chanC, int32_t aamplA,
                   int32_t aamplC, int16_t atimeA, int16_t atimeC, Bool_t isLaser = kFALSE)
  {
    triggersignals = (isA << bitA) | (isC << bitC) | (isVrtx << bitVertex) | (isCnt << bitCen) | (isSCnt << bitSCen) | (isLaser << bitLaser);
    nChanA = chanA;
    nChanC = chanC;
    amplA = aamplA;
    amplC = aamplC;
    timeA = atimeA;
    timeC = atimeC;
  }
  void cleanTriggers()
  {
    triggersignals = 0;
    nChanA = nChanC = 0;
    amplA = amplC = -5000;
    timeA = timeC = -5000;
  }
  bool operator==(Triggers const& other) const
  {
    return std::tie(triggersignals, nChanA, nChanC, amplA, amplC, timeA, timeC) ==
           std::tie(other.triggersignals, other.nChanA, other.nChanC, other.amplA, other.amplC, other.timeA, other.timeC);
  }
  void printLog() const;
  ClassDefNV(Triggers, 2);
};

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "FT0TRIGGERINPUT";
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
  static constexpr char sDigitBranchName[] = "FT0DIGITSBC";
  o2::dataformats::RangeReference<int, int> ref;
  Triggers mTriggers;               // pattern of triggers  in this BC
  uint8_t mEventStatus;             //Status of event from FT0, such as Pileup , etc
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  int mEventID;
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
  o2::InteractionRecord getIntRecord() const { return mIntRecord; };
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
  ClassDefNV(Digit, 6);
};

//For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  static constexpr char sChannelNameDPL[] = "DIGITSTRGEXT";
  static constexpr char sDigitName[] = "TriggersExt";
  static constexpr char sDigitBranchName[] = "FT0DIGITSTRGEXT";
  TriggersExt(std::array<uint32_t, 20> triggerWords) : mTriggerWords(triggerWords) {}
  TriggersExt() = default;
  o2::InteractionRecord mIntRecord;
  void setTrgWord(uint32_t trgWord, std::size_t pos) { mTriggerWords[pos] = trgWord; }
  std::array<uint32_t, 20> mTriggerWords;
  void printLog() const;
  ClassDefNV(TriggersExt, 2);
};
} // namespace ft0
} // namespace o2

#endif
