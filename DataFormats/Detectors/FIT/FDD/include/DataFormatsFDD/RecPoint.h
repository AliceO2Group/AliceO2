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

/// \file  RecPoint.h
/// \brief Definition of the FDD RecPoint class

#ifndef ALICEO2_FDD_RECPOINT_H
#define ALICEO2_FDD_RECPOINT_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFDD/ChannelData.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFIT/ChannelDataBit.h"

namespace o2
{
namespace fdd
{
struct ChannelDataFloat {
  static constexpr int DUMMY_CHANNEL_ID = -1;
  static constexpr int DUMMY_CHAIN_QTC = -1;
  static constexpr double DUMMY_CFD_TIME = -20000;
  static constexpr double DUMMY_QTC_AMPL = -20000;

  int mPMNumber = DUMMY_CHANNEL_ID;   // channel Id
  int adcId = DUMMY_CHAIN_QTC;        // QTC chain
  double mTime = DUMMY_CFD_TIME;      // time in ps, 0 at the LHC clk center
  double mChargeADC = DUMMY_QTC_AMPL; // charge [channels]

  ChannelDataFloat() = default;
  ChannelDataFloat(int Channel, double Time, double Charge, int AdcId)
  {
    mPMNumber = Channel;
    mTime = Time;
    mChargeADC = Charge;
    adcId = AdcId;
  }

  static void setFlag(fit::ChannelDataBit bitFlag, int& adcId)
  {
    adcId = uint8_t(adcId) | 1u << uint8_t(bitFlag);
  }
  static void clearFlag(fit::ChannelDataBit bitFlag, int& adcId)
  {
    adcId = uint8_t(adcId) & ~(1u << uint8_t(bitFlag));
  }
  void setFlag(int flag)
  {
    adcId = flag;
  }
  void setFlag(fit::ChannelDataBit bitFlag, bool value)
  {
    adcId = uint8_t(adcId) | uint8_t(value) << uint8_t(bitFlag);
  }
  [[nodiscard]] bool getFlag(fit::ChannelDataBit bitFlag) const
  {
    return bool(uint8_t(adcId) & (1u << uint8_t(bitFlag)));
  }
  [[nodiscard]] bool areAllFlagsGood() const
  {
    return (!getFlag(fit::ChannelDataBit::kIsDoubleEvent) &&
            !getFlag(fit::ChannelDataBit::kIsTimeInfoNOTvalid) &&
            getFlag(fit::ChannelDataBit::kIsCFDinADCgate) &&
            !getFlag(fit::ChannelDataBit::kIsTimeInfoLate) &&
            !getFlag(fit::ChannelDataBit::kIsAmpHigh) &&
            getFlag(fit::ChannelDataBit::kIsEventInTVDC) &&
            !getFlag(fit::ChannelDataBit::kIsTimeInfoLost));
  }

  void print() const;
  [[nodiscard]] int getChannelId() const { return mPMNumber; }
  [[nodiscard]] double getTime() const { return mTime; }
  [[nodiscard]] double getAmp() const { return mChargeADC; }
  bool operator==(const ChannelDataFloat&) const = default;

  ClassDefNV(ChannelDataFloat, 1);
};

class RecPoint
{
 public:
  enum TimeTypeIndex : int { TimeA,
                             TimeC };
  RecPoint() = default;
  RecPoint(const std::array<int, 2>& collisiontime, int first, int ne,
           o2::InteractionRecord iRec, o2::fdd::Triggers triggers)
    : mCollisionTimePs(collisiontime)
  {
    mRef.setFirstEntry(first);
    mRef.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = triggers;
  }
  ~RecPoint() = default;

  float getCollisionTime(TimeTypeIndex type) const { return mCollisionTimePs[type]; }
  float getCollisionTimeA() const { return getCollisionTime(TimeA); }
  float getCollisionTimeC() const { return getCollisionTime(TimeC); }
  bool isValidTime(TimeTypeIndex type) const { return getCollisionTime(type) < sDummyCollissionTime; }
  void setCollisionTime(Float_t time, TimeTypeIndex type) { mCollisionTimePs[type] = time; }

  const o2::fdd::Triggers getTrigger() const { return mTriggers; }
  o2::InteractionRecord getInteractionRecord() const { return mIntRecord; };
  gsl::span<const ChannelDataFloat> getBunchChannelData(const gsl::span<const ChannelDataFloat> tfdata) const
  {
    // extract the span of channel data for this bunch from the whole TF data
    return mRef.getEntries() ? gsl::span<const ChannelDataFloat>(tfdata).subspan(mRef.getFirstEntry(), mRef.getEntries()) : gsl::span<const ChannelDataFloat>();
  }
  short static constexpr sDummyCollissionTime = 32767;
  int getFirstEntry() const { return mRef.getFirstEntry(); }
  int getEntriesInCurrentBC() const { return mRef.getEntries(); }

  void print() const;
  bool operator==(const RecPoint&) const = default;

 private:
  o2::dataformats::RangeReference<int, int> mRef;
  o2::InteractionRecord mIntRecord;
  o2::fdd::Triggers mTriggers;                                                        // pattern of triggers  in this BC
  std::array<int, 2> mCollisionTimePs = {sDummyCollissionTime, sDummyCollissionTime}; // in picoseconds

  ClassDefNV(RecPoint, 3);
};
} // namespace fdd
} // namespace o2
#endif
