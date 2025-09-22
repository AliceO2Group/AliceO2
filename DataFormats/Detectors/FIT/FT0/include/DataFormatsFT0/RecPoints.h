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

/// \file RecPoints.h
/// \brief Definition of the FIT RecPoints class

#ifndef ALICEO2_FT0_RECPOINTS_H
#define ALICEO2_FT0_RECPOINTS_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFT0/ChannelData.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFIT/EEventDataBit.h"
#include <array>
#include "Rtypes.h"
#include <TObject.h>
#include <gsl/span>
#include <string>
#include <utility>
#include <map>
namespace o2
{
namespace ft0
{

struct ChannelDataFloat {
  static constexpr int DUMMY_CHANNEL_ID = -1;
  static constexpr int DUMMY_CHAIN_QTC = -1;
  static constexpr float DUMMY_CFD_TIME = -20000;
  static constexpr float DUMMY_QTC_AMPL = -20000;

  int ChId = DUMMY_CHANNEL_ID;    // channel Id
  int ChainQTC = DUMMY_CHAIN_QTC; // QTC chain
  float CFDTime = DUMMY_CFD_TIME; // time in ps, 0 at the LHC clk center
  float QTCAmpl = DUMMY_QTC_AMPL; // Amplitude mV

  ChannelDataFloat() = default;
  ChannelDataFloat(int iPmt, float time, float charge, int chainQTC)
  {
    ChId = iPmt;
    CFDTime = time;
    QTCAmpl = charge;
    ChainQTC = chainQTC;
  }

  static void setFlag(fit::EEventDataBit bitFlag, int& chainQTC)
  {
    chainQTC = uint8_t(chainQTC) | 1u << uint8_t(bitFlag);
  }
  static void clearFlag(fit::EEventDataBit bitFlag, int& chainQTC)
  {
    chainQTC = uint8_t(chainQTC) & ~(1u << uint8_t(bitFlag));
  }
  void setFlag(int flag)
  {
    ChainQTC = flag;
  }
  void setFlag(fit::EEventDataBit bitFlag, bool value)
  {
    ChainQTC = uint8_t(ChainQTC) | uint8_t(value) << uint8_t(bitFlag);
  }
  [[nodiscard]] bool getFlag(fit::EEventDataBit bitFlag) const
  {
    return bool(uint8_t(ChainQTC) & (1u << uint8_t(bitFlag)));
  }
  [[nodiscard]] bool areAllFlagsGood() const
  {
    return (!getFlag(fit::EEventDataBit::kIsDoubleEvent) &&
            !getFlag(fit::EEventDataBit::kIsTimeInfoNOTvalid) &&
            getFlag(fit::EEventDataBit::kIsCFDinADCgate) &&
            !getFlag(fit::EEventDataBit::kIsTimeInfoLate) &&
            !getFlag(fit::EEventDataBit::kIsAmpHigh) &&
            getFlag(fit::EEventDataBit::kIsEventInTVDC) &&
            !getFlag(fit::EEventDataBit::kIsTimeInfoLost));
  }

  void print() const;
  [[nodiscard]] int getChannelId() const { return ChId; }
  [[nodiscard]] float getTime() const { return CFDTime; }
  [[nodiscard]] float getAmp() const { return QTCAmpl; }

  bool operator==(const ChannelDataFloat&) const = default;

  ClassDefNV(ChannelDataFloat, 1);
};

class RecPoints
{

 public:
  enum ETimeType { kTimeMean,
                   kTimeA,
                   kTimeC,
                   kTimeVertex };

  // Enum for trigger nits specified in rec-points and AOD data
  enum ETriggerBits { kOrA = 0,           // OrA time-trigger signal
                      kOrC = 1,           // OrC time-trigger signal
                      kSemiCentral = 2,   // Semi-central amplitude-trigger signal
                      kCentral = 3,       // Central amplitude-trigger signal
                      kVertex = 4,        // Vertex time-trigger signal
                      kIsActiveSideA = 5, // Side-A has at least one channel active
                      kIsActiveSideC = 6, // Side-C has at least one channel active
                      kIsFlangeEvent = 7  // Flange event at Side-C, at least one channel has time which corresponds to -82 cm area
  };
  static const inline std::map<unsigned int, std::string> sMapTriggerBits = {
    {ETriggerBits::kOrA, "OrA"},
    {ETriggerBits::kOrC, "OrC"},
    {ETriggerBits::kSemiCentral, "Semicentral"},
    {ETriggerBits::kCentral, "Central"},
    {ETriggerBits::kVertex, "Vertex"},
    {ETriggerBits::kIsActiveSideA, "IsActiveSideA"},
    {ETriggerBits::kIsActiveSideC, "IsActiveSideC"},
    {ETriggerBits::kIsFlangeEvent, "IsFlangeEvent"}};

  enum ETechnicalBits { kLaser = 0,             // indicates the laser was triggered in this BC
                        kOutputsAreBlocked = 1, // indicates that laser-induced pulses should arrive from detector to FEE in this BC (and trigger outputs are blocked)
                        kDataIsValid = 2,       // data is valid for processing
  };
  static const inline std::map<unsigned int, std::string> sMapTechnicalBits = {
    {ETechnicalBits::kLaser, "Laser"},
    {ETechnicalBits::kOutputsAreBlocked, "OutputsAreBlocked"},
    {ETechnicalBits::kDataIsValid, "DataIsValid"}};

  o2::dataformats::RangeReference<int, int> ref;
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  RecPoints() = default;
  RecPoints(const std::array<short, 4>& collisiontime,
            int first, int ne, o2::InteractionRecord iRec, o2::fit::Triggers chTrig)
    : mCollisionTime(collisiontime)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = chTrig;
  }
  RecPoints(int chDataFirstEntryPos,
            int chDataNEntries,
            const o2::InteractionRecord& ir,
            const std::array<short, 4>& arrTimes,
            const o2::fit::Triggers& digitTriggers,
            uint8_t extraTriggerWord) : mIntRecord(ir), mCollisionTime(arrTimes), mTriggers(digitTriggers)
  {
    ref.setFirstEntry(chDataFirstEntryPos);
    ref.setEntries(chDataNEntries);
    initRecPointTriggers(digitTriggers, extraTriggerWord);
  }

  ~RecPoints() = default;

  short getCollisionTime(int side) const { return mCollisionTime[side]; }
  short getCollisionTimeMean() const { return getCollisionTime(kTimeMean); }
  short getCollisionTimeA() const { return getCollisionTime(kTimeA); }
  short getCollisionTimeC() const { return getCollisionTime(kTimeC); }
  bool isValidTime(int side) const { return getCollisionTime(side) < o2::InteractionRecord::DummyTime; }
  void setCollisionTime(short time, int side) { mCollisionTime[side] = time; }

  short getVertex() const { return getCollisionTime(kTimeVertex); }
  void setVertex(short vertex) { mCollisionTime[kTimeVertex] = vertex; }

  o2::fit::Triggers getTrigger() const { return mTriggers; }
  void setTriggers(o2::fit::Triggers trig) { mTriggers = trig; }
  uint8_t getTechnicalWord() const { return mTechnicalWord; }
  static constexpr uint8_t makeExtraTrgWord(bool isActiveA = true, bool isActiveC = true, bool isFlangeEvent = true)
  {
    return (static_cast<uint8_t>(isActiveA) << kIsActiveSideA) |
           (static_cast<uint8_t>(isActiveC) << kIsActiveSideC) |
           (static_cast<uint8_t>(isFlangeEvent) << kIsFlangeEvent);
  }

  o2::InteractionRecord getInteractionRecord() const { return mIntRecord; };
  gsl::span<const ChannelDataFloat> getBunchChannelData(const gsl::span<const ChannelDataFloat> tfdata) const;
  short static constexpr sDummyCollissionTime = 32767;

  void print() const;
  bool operator==(const RecPoints&) const = default;

 private:
  void initRecPointTriggers(const o2::fit::Triggers& digitTriggers, uint8_t extraTrgWord = 0)
  {
    const auto digitTriggerWord = digitTriggers.getTriggersignals();
    const auto trgAndTechWordPair = o2::fit::Triggers::parseDigitTriggerWord(digitTriggerWord, true);
    mTriggers.setTriggers(trgAndTechWordPair.first | extraTrgWord);
    mTechnicalWord = trgAndTechWordPair.second;
  }

  std::array<short, 4> mCollisionTime = {sDummyCollissionTime,
                                         sDummyCollissionTime,
                                         sDummyCollissionTime,
                                         sDummyCollissionTime};
  o2::fit::Triggers mTriggers; // pattern of triggers  in this BC
  uint8_t mTechnicalWord{0};   // field for keeping ETechnicalBits
  ClassDefNV(RecPoints, 4);
};
} // namespace ft0
} // namespace o2
#endif
