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
  uint8_t triggersignals; // T0 trigger signals
  int8_t nChanA;          // number of faired channels A side
  int8_t nChanC;          // number of faired channels A side
  int32_t amplA;          // sum amplitude A side
  int32_t amplC;          // sum amplitude C side
  int16_t timeA;          // average time A side
  int16_t timeC;          // average time C side
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
  bool getOrA() { return (triggersignals & (1 << bitA)) != 0; }
  bool getOrC() { return (triggersignals & (1 << bitC)) != 0; }
  bool getVertex() { return (triggersignals & (1 << bitVertex)) != 0; }
  bool getCen() { return (triggersignals & (1 << bitCen)) != 0; }
  bool getSCen() { return (triggersignals & (1 << bitSCen)) != 0; }

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
    nChanA = nChanC = -1;
    amplA = amplC = -1000;
    timeA = timeC = -1000;
  }

  Triggers getTriggers();

  ClassDefNV(Triggers, 1);
};

struct Digit {
  o2::dataformats::RangeReference<int, int> ref;
  Triggers mTriggers;               // pattern of triggers  in this BC
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  int mEventID;
  Digit() = default;
  Digit(int first, int ne, o2::InteractionRecord iRec, Triggers chTrig, int event)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = chTrig;
    mEventID = event;
  }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  Triggers getTriggers() { return mTriggers; }
  int getEventID() const { return mEventID; }
  o2::InteractionRecord getIntRecord() { return mIntRecord; };
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  enum { bit1TimeLostEvent,
         bit2TimeLostEvent,
         bitADCinGate,
         bitTimeInfoLate,
         bitAmpHigh,
         bitEventInTVDC,
         bitTimeInfoLost };
  // T0 channel flags
  uint8_t flags;
  void setFlags(bool is1TimeLostEvent, bool is2TimeLostEvent, bool isADCinGate, bool isTimeInfoLate, bool isAmpHigh, bool isEventInTVDC, bool isTimeInfoLost)
  {
    flags = (is1TimeLostEvent << bit1TimeLostEvent) | (is2TimeLostEvent << bit2TimeLostEvent) | (isADCinGate << bitADCinGate) | (isTimeInfoLate << bitTimeInfoLate) | (isAmpHigh << bitAmpHigh) | (isEventInTVDC << bitEventInTVDC) | (isTimeInfoLost << bitTimeInfoLost);
  };

  void printStream(std::ostream& stream) const;

  ClassDefNV(Digit, 3);
};
} // namespace ft0
} // namespace o2

#endif
