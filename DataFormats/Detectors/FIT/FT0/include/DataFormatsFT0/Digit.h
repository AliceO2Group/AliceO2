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

namespace o2
{
namespace ft0
{
class ChannelData;

struct Triggers {
  uint8_t  triggersignals;    // T0 trigger signals
  uint8_t nChanA;        // number of faired channels A side 
  uint8_t nChanC; // number of faired channels A side 
  uint16_t amplA; // sum amplitude A side 
  uint16_t amplC;// sum amplitude C side
  uint16_t timeA;// average time A side
  uint16_t timeC;// average time C side
  Triggers() = default;
  Triggers(  uint8_t signals, uint8_t chanA, uint8_t chanC, uint16_t aamplA,  uint16_t aamplC, uint16_t atimeA,  uint16_t atimeC)
  {
    triggersignals = signals ;
    nChanA = chanA;
    nChanC =chanC;
    aamplA = amplA;
    aamplC = amplC;
    timeA = atimeA;
    timeC = atimeC;
  }
  bool GetFT0Triggers(int i) {return (triggersignals&(1<<i)) != 0;}
  uint8_t GetFT0Triggers() {return triggersignals;}
  void SetFT0Triggers(bool tr[5])
  {
    triggersignals = 0;
    for (Int_t i = 0; i < 5; i++)
      triggersignals = triggersignals | (tr[i] ? (1 << i) : 0);   
  }
  void setTriggers(Bool_t isA, Bool_t isC, Bool_t isCnt, Bool_t isSCnt, Bool_t isVrtx)
  {
    triggersignals = triggersignals| (isA  ? (1 << 0) : 0);
    mIsC = isC;
    mIsCentral = isCnt;
    mIsSemiCentral = isSCnt;
    mIsVertex = isVrtx;
  }

  
  void SetTriggerWords(  uint8_t chanA, uint8_t chanC, uint16_t aamplA,  uint16_t aamplC, uint16_t atimeA,  uint16_t atimeC)
  {
    nChanA = chanA;
    nChanC = chanC;
    aamplA = amplA;
    aamplC = amplC;
    timeA = atimeA;
    timeC = atimeC;
  }

  ClassDefNV(Triggers, 1);
};

struct Digit {
  o2::dataformats::RangeRefComp<5> ref;

  Triggers mTriggers;               // pattern of triggers  in this BC
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)

  Digit() = default;
  Digit(int first, int ne, o2::InteractionRecord iRec, Triggers chTrig)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = chTrig;
  }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  o2::InteractionRecord getIntRecord() { return mIntRecord; };
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;

  ClassDefNV(Digit, 2);
};
} // namespace ft0
} // namespace o2

#endif
