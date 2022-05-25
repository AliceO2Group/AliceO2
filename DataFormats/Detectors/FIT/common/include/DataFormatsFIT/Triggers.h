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

// \file Triggers.h
/// \brief Class to describe triggers common for all FIT detectors
/// \author maciej.slupecki@cern.ch

#ifndef _FIT_TRIGGERS_H_
#define _FIT_TRIGGERS_H_

#include <Rtypes.h>
#include <tuple>
#include <iostream>
#include <string>

namespace o2
{
namespace fit
{

class Triggers
{
 public:
  enum { bitA = 0,
         bitC = 1,   // alias of bitAIn (FT0/FDD)
         bitAIn = 1, // alias of bitC (FV0)
         bitSCen = 2,
         bitAOut = 2, // alias of bitVertex (FV0)
         bitCen = 3,
         bitVertex = 4,            // alias of bitAOut (FT0/FDD)
         bitLaser = 5,             // indicates the laser was triggered in this BC
         bitOutputsAreBlocked = 6, // indicates that laser-induced pulses should arrive from detector to FEE in this BC (and trigger outputs are blocked)
         bitDataIsValid = 7 };
  static const int16_t DEFAULT_TIME = -5000; // for average of one side (A or C)
  static const int16_t DEFAULT_AMP = 0;
  static const int16_t DEFAULT_ZERO = 0;

  Triggers() = default;
  Triggers(uint8_t signals, uint8_t chanA, uint8_t chanC, int32_t aamplA, int32_t aamplC, int16_t atimeA, int16_t atimeC)
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
  bool getOrC() const { return (triggersignals & (1 << bitC)) != 0; }         // only used by FT0/FDD (same bit as OrAIn in FV0)
  bool getOrAIn() const { return (triggersignals & (1 << bitAIn)) != 0; }     // only used by FV0 (same bit as OrC in FT0/FDD)
  bool getVertex() const { return (triggersignals & (1 << bitVertex)) != 0; } // only used by FT0/FDD (same bit as OrAOut in FV0
  bool getOrAOut() const { return (triggersignals & (1 << bitAOut)) != 0; }   // only used by FV0 (same bit as Vertex in FT0/FDD)
  bool getCen() const { return (triggersignals & (1 << bitCen)) != 0; }
  bool getSCen() const { return (triggersignals & (1 << bitSCen)) != 0; }
  bool getLaser() const { return (triggersignals & (1 << bitLaser)) != 0; }
  bool getLaserBit() const { return getLaser(); } // TODO: remove after QC is modified
  bool getOutputsAreBlocked() const { return (triggersignals & (1 << bitOutputsAreBlocked)) != 0; }
  bool getDataIsValid() const { return (triggersignals & (1 << bitDataIsValid)) != 0; }

  int8_t getTriggersignals() const { return triggersignals; }
  uint8_t getNChanA() const { return nChanA; }
  uint8_t getNChanC() const { return nChanC; }
  int32_t getAmplA() const { return amplA; }
  int32_t getAmplC() const { return amplC; }
  int16_t getTimeA() const { return timeA; }
  int16_t getTimeC() const { return timeC; }

  void setTriggers(uint8_t trgsig, uint8_t chanA, uint8_t chanC, int32_t aamplA, int32_t aamplC, int16_t atimeA, int16_t atimeC)
  {
    triggersignals = trgsig;
    nChanA = chanA;
    nChanC = chanC;
    amplA = aamplA;
    amplC = aamplC;
    timeA = atimeA;
    timeC = atimeC;
  }

  void setTriggers(Bool_t isA, Bool_t isC, Bool_t isVrtx, Bool_t isCnt, Bool_t isSCnt, uint8_t chanA, uint8_t chanC, int32_t aamplA,
                   int32_t aamplC, int16_t atimeA, int16_t atimeC, Bool_t isLaser, Bool_t isOutputsAreBlocked, Bool_t isDataValid)
  {
    uint8_t trgsig = (isA << bitA) | (isC << bitC) | (isVrtx << bitVertex) | (isCnt << bitCen) | (isSCnt << bitSCen) | (isLaser << bitLaser) | (isOutputsAreBlocked << bitOutputsAreBlocked) | (isDataValid << bitDataIsValid);
    setTriggers(trgsig, chanA, chanC, aamplA, aamplC, atimeA, atimeC);
  }

  void cleanTriggers()
  {
    triggersignals = DEFAULT_ZERO;
    nChanA = nChanC = DEFAULT_ZERO;
    amplA = amplC = DEFAULT_AMP;
    timeA = timeC = DEFAULT_TIME;
  }

  bool operator==(Triggers const& other) const
  {
    return std::tie(triggersignals, nChanA, nChanC, amplA, amplC, timeA, timeC) ==
           std::tie(other.triggersignals, other.nChanA, other.nChanC, other.amplA, other.amplC, other.timeA, other.timeC);
  }

  std::string print() const;
  void print(std::ostream&) const;
  void printLog() const;

 public:                                 // TODO: change to 'private' after modifying QC to use the setters/getters
  uint8_t triggersignals = DEFAULT_ZERO; // FIT trigger signals
  uint8_t nChanA = DEFAULT_ZERO;         // number of fired channels A side
  uint8_t nChanC = DEFAULT_ZERO;         // number of fired channels A side
  int32_t amplA = DEFAULT_AMP;           // sum amplitude A side
  int32_t amplC = DEFAULT_AMP;           // sum amplitude C side
  int16_t timeA = DEFAULT_TIME;          // average time A side (shouldn't be used if nChanA == 0)
  int16_t timeC = DEFAULT_TIME;          // average time C side (shouldn't be used if nChanC == 0)

  ClassDefNV(Triggers, 5);
};

} // namespace fit
} // namespace o2

#endif
