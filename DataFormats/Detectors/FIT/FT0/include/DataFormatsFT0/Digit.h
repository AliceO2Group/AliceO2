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
         bitSCen };
  uint8_t triggersignals = 0; // T0 trigger signals
  int8_t nChanA = 0;          // number of fired channels A side
  int8_t nChanC = 0;          // number of fired channels A side
  int32_t amplA = -1000;      // sum amplitude A side
  int32_t amplC = -1000;      // sum amplitude C side
  int16_t timeA = -1000;      // average time A side
  int16_t timeC = -1000;      // average time C side
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

  void setTriggers(Bool_t isA, Bool_t isC, Bool_t isVrtx, Bool_t isCnt, Bool_t isSCnt, int8_t chanA, int8_t chanC, int32_t aamplA,
                   int32_t aamplC, int16_t atimeA, int16_t atimeC)
  {
    triggersignals = (isA << bitA) | (isC << bitC) | (isVrtx << bitVertex) | (isCnt << bitCen) | (isSCnt << bitSCen);
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
    amplA = amplC = -1000;
    timeA = timeC = -1000;
  }

  ClassDefNV(Triggers, 1);
};

struct DetTrigInput {
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
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  Triggers getTriggers() const { return mTriggers; }
  int getEventID() const { return mEventID; }
  o2::InteractionRecord getIntRecord() { return mIntRecord; };
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;

  void printStream(std::ostream& stream) const;
  void setTriggers(Triggers trig) { mTriggers = trig; };
  void setEventStatus(uint8_t stat) { mEventStatus = stat; };
  void setStatusFlag(EEventStatus bit, bool value) { mEventStatus |= (value << bit); };
  bool getStatusFlag(EEventStatus bit) const { return bool(mEventStatus << bit); }
  uint8_t getEventStatusWord() const { return mEventStatus; }

  ClassDefNV(Digit, 5);
};

//For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  TriggersExt(uint32_t triggerWord)
  {
    mTriggerWord = triggerWord;
  }
  TriggersExt() = default;
  uint32_t mTriggerWord;
  ClassDefNV(TriggersExt, 1);
};

//For TCM extended mode (calibration mode)
struct DigitExt : Digit {
  DigitExt(int first, int ne, int firstExt, int neExt, const o2::InteractionRecord& iRec, const Triggers& chTrig, int event) : Digit(first, ne, iRec, chTrig, event)
  {
    refExt.setFirstEntry(firstExt);
    refExt.setEntries(neExt);
  }
  DigitExt() = default;
  o2::dataformats::RangeReference<int, int> refExt; //range reference to container with TriggerExt objects
  ClassDefNV(DigitExt, 1);
};
} // namespace ft0
} // namespace o2

#endif
