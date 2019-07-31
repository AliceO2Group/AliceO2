// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecPoints.h
/// \brief Definition of the FIT RecPoints class
#ifndef ALICEO2_FIT_RECPOINTS_H
#define ALICEO2_FIT_RECPOINTS_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include <array>
#include "Rtypes.h"
#include <TObject.h>
#include <DataFormatsFT0/Digit.h>

namespace o2
{
namespace ft0
{

class RecPoints
{
 public:
  enum : int { TimeMean,
               TimeA,
               TimeC };

  RecPoints() = default;
  RecPoints(const std::array<Float_t, 3>& collisiontime,
            Float_t vertex,
            std::vector<o2::ft0::ChannelData> timeamp)
    : mCollisionTime(collisiontime),
      mVertex(vertex),
      mTimeAmp(std::move(timeamp))
  {
  }
  ~RecPoints() = default;

  void fillFromDigits(const o2::ft0::Digit& digit);
  float getCollisionTime(int side) const { return mCollisionTime[side]; }
  float getCollisionTimeMean() const { return getCollisionTime(TimeMean); }
  float getCollisionTimeA() const { return getCollisionTime(TimeC); }
  float getCollisionTimeC() const { return getCollisionTime(TimeA); }
  bool isValidTime(int side) const { return getCollisionTime(side) < o2::InteractionRecord::DummyTime; }
  void setCollisionTime(Float_t time, int side) { mCollisionTime[side] = time; }

  float getTimeFromDigit() const { return mEventTime; }
  void setTimFromDigit(Float_t time) { mEventTime = time; }

  Float_t getVertex(Float_t vertex) const { return mVertex; }
  void setVertex(Float_t vertex) { mVertex = vertex; }

  void SetMgrEventTime(Double_t time) { mEventTime = time; }

  const std::vector<o2::ft0::ChannelData>& getChDgData() const { return mTimeAmp; }
  void setChDgData(const std::vector<o2::ft0::ChannelData>& TimeAmp) { mTimeAmp = TimeAmp; }
  void setChDgData(std::vector<o2::ft0::ChannelData>&& TimeAmp) { mTimeAmp = std::move(TimeAmp); }

  void setInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord() { return mIntRecord; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }

 private:
  std::array<Float_t, 3> mCollisionTime = { 2 * o2::InteractionRecord::DummyTime,
                                            2 * o2::InteractionRecord::DummyTime,
                                            2 * o2::InteractionRecord::DummyTime };
  Float_t mVertex = 0;
  Double_t mEventTime = 2 * o2::InteractionRecord::DummyTime; //event time from Fair for continuous
  std::vector<o2::ft0::ChannelData> mTimeAmp;
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc) from digits

  ClassDefNV(RecPoints, 1);
};
} // namespace ft0
} // namespace o2
#endif
