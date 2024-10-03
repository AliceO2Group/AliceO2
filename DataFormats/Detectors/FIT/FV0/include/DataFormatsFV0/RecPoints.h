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
/// \brief Definition of the FV0 RecPoints class

#ifndef ALICEO2_FV0_RECPOINTS_H
#define ALICEO2_FV0_RECPOINTS_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsFV0/Digit.h"
#include <array>
#include <gsl/span>

namespace o2
{
namespace fv0
{
struct ChannelDataFloat {

  int channel = -1;       // channel Id
  double time = -20000;   // time in ns, 0 at the LHC clk center
  double charge = -20000; // charge [channels]
  int adcId = -1;         // QTC chain

  ChannelDataFloat() = default;
  ChannelDataFloat(int Channel, double Time, double Charge, int AdcId)
  {
    channel = Channel;
    time = Time;
    charge = Charge;
    adcId = AdcId;
  }

  void print() const;
  bool operator==(const ChannelDataFloat&) const = default;

  ClassDefNV(ChannelDataFloat, 1);
};

class RecPoints
{

 public:
  enum TimeTypeIndex : int { TimeFirst,
                             TimeGlobalMean,
                             TimeSelectedMean };
  RecPoints() = default;
  RecPoints(const std::array<short, 3>& collisiontime, int first, int ne,
            o2::InteractionRecord iRec, o2::fit::Triggers triggers)
    : mCollisionTimePs(collisiontime)
  {
    mRef.setFirstEntry(first);
    mRef.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = triggers;
  }
  ~RecPoints() = default;

  float getCollisionTime(TimeTypeIndex type) const { return mCollisionTimePs[type]; }
  float getCollisionFirstTime() const { return getCollisionTime(TimeFirst); }
  float getCollisionGlobalMeanTime() const { return getCollisionTime(TimeGlobalMean); }
  float getCollisionSelectedMeanTime() const { return getCollisionTime(TimeSelectedMean); }
  bool isValidTime(TimeTypeIndex type) const { return getCollisionTime(type) < sDummyCollissionTime; }
  void setCollisionTime(Float_t time, TimeTypeIndex type) { mCollisionTimePs[type] = time; }

  o2::fit::Triggers getTrigger() const { return mTriggers; }
  o2::InteractionRecord getInteractionRecord() const { return mIntRecord; };
  gsl::span<const ChannelDataFloat> getBunchChannelData(const gsl::span<const ChannelDataFloat> tfdata) const;
  short static constexpr sDummyCollissionTime = 32767;

  void print() const;
  bool operator==(const RecPoints&) const = default;

 private:
  o2::dataformats::RangeReference<int, int> mRef;
  o2::InteractionRecord mIntRecord;
  o2::fit::Triggers mTriggers;                                                                                // pattern of triggers  in this BC
  std::array<short, 3> mCollisionTimePs = {sDummyCollissionTime, sDummyCollissionTime, sDummyCollissionTime}; // in picoseconds

  ClassDefNV(RecPoints, 1);
};
} // namespace fv0
} // namespace o2
#endif
