// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGIT_H
#define ALICEO2_FDD_DIGIT_H

#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFDD/ChannelData.h"
#include <iosfwd>
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>

namespace o2
{
namespace fdd
{

class ChannelData;

struct Triggers {
  enum { bitA,
         bitC,
         bitVertex,
         bitCen,
         bitSCen };
  uint8_t triggersignals = 0; // FDD trigger signals
  int8_t nChanA = -1;         // number of fired channels A side
  int8_t nChanC = -1;         // number of fired channels A side
  int32_t amplA = -1024;      // sum amplitude A side
  int32_t amplC = -1024;      // sum amplitude C side
  int16_t timeA = -1024;      // average time A side
  int16_t timeC = -1024;      // average time C side
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
  void cleanTriggers()
  {
    triggersignals = 0;
    nChanA = nChanC = -1;
    amplA = amplC = -1024;
    timeA = timeC = -1024;
  }
  Triggers getTriggers();

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
  o2::dataformats::RangeRefComp<5> ref;

  Triggers mTriggers;               // pattern of triggers  in this BC
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)

  Digit() = default;
  Digit(int first, int ne, o2::InteractionRecord iRec, Triggers chTrig) : ref(first, ne), mIntRecord(iRec), mTriggers(chTrig) {}

  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  o2::InteractionRecord getIntRecord() const { return mIntRecord; };
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
  {
    return ref.getEntries() ? gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const ChannelData>();
  }

  ClassDefNV(Digit, 3);
};
} // namespace fdd
} // namespace o2
#endif
