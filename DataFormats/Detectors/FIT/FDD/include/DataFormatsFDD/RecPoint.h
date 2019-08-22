// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecPoint.h
/// \brief Definition of the FDD RecPoint class
#ifndef ALICEO2_FDD_RECPOINT_H
#define ALICEO2_FDD_RECPOINT_H

#include "CommonDataFormat/InteractionRecord.h"
#include <DataFormatsFDD/Digit.h>

namespace o2
{
namespace fdd
{

class RecPoint
{
 public:
  RecPoint() = default;
  RecPoint(Double_t timeA, Double_t timeC,
           std::vector<o2::fdd::ChannelData> channeldata)
    : mMeanTimeFDA(timeA),
      mMeanTimeFDC(timeC),
      mChannelData(std::move(channeldata))
  {
  }
  ~RecPoint() = default;

  Double_t GetMeanTimeFDA() const { return mMeanTimeFDA; }
  Double_t GetMeanTimeFDC() const { return mMeanTimeFDC; }

  void SetMeanTimeFDA(Double_t time) { mMeanTimeFDA = time; }
  void SetMeanTimeFDC(Double_t time) { mMeanTimeFDC = time; }

  float GetTimeFromDigit() const { return mEventTime; }
  void SetTimeFromDigit(Float_t time) { mEventTime = time; }

  const std::vector<o2::fdd::ChannelData>& GetChannelData() const { return mChannelData; }
  void SetChannelData(const std::vector<o2::fdd::ChannelData>& ChannelData) { mChannelData = ChannelData; }
  void SetChannelData(std::vector<o2::fdd::ChannelData>&& ChannelData) { mChannelData = std::move(ChannelData); }

  void SetInteractionRecord(const o2::InteractionRecord& intRecord) { mIntRecord = intRecord; }
  void SetInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& GetInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& GetInteractionRecord() { return mIntRecord; }
  uint32_t GetOrbit() const { return mIntRecord.orbit; }
  uint16_t GetBC() const { return mIntRecord.bc; }

 private:
  Double_t mMeanTimeFDA = o2::InteractionRecord::DummyTime;
  Double_t mMeanTimeFDC = o2::InteractionRecord::DummyTime;
  std::vector<o2::fdd::ChannelData> mChannelData;

  Double_t mEventTime = o2::InteractionRecord::DummyTime; //event time from Fair for continuous
  o2::InteractionRecord mIntRecord;                       // Interaction record (orbit, bc) from digits

  ClassDefNV(RecPoint, 1);
};
} // namespace fdd
} // namespace o2
#endif
