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
#include <array>
#include "Rtypes.h"
#include <TObject.h>
#include <gsl/span>

namespace o2
{
namespace ft0
{
struct ChannelDataFloat {

  int ChId = -1;           //channel Id
  int ChainQTC = -1;       //QTC chain
  double CFDTime = -20000; //time in ps, 0 at the LHC clk center
  double QTCAmpl = -20000; // Amplitude mV

  ChannelDataFloat() = default;
  ChannelDataFloat(int iPmt, double time, double charge, int chainQTC)
  {
    ChId = iPmt;
    CFDTime = time;
    QTCAmpl = charge;
    ChainQTC = chainQTC;
  }

  void print() const;

  ClassDefNV(ChannelDataFloat, 1);
};

class RecPoints
{

 public:
  enum : int { TimeMean,
               TimeA,
               TimeC,
               Vertex };

  o2::dataformats::RangeReference<int, int> ref;
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  RecPoints() = default;
  RecPoints(const std::array<Float_t, 4>& collisiontime,
            int first, int ne, o2::InteractionRecord iRec, o2::ft0::Triggers chTrig)
    : mCollisionTime(collisiontime)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = chTrig;
  }
  ~RecPoints() = default;

  void print() const;

  o2::ft0::Triggers mTriggers; // pattern of triggers  in this BC

  float getCollisionTime(int side) const { return mCollisionTime[side]; }
  float getCollisionTimeMean() const { return getCollisionTime(TimeMean); }
  float getCollisionTimeA() const { return getCollisionTime(TimeA); }
  float getCollisionTimeC() const { return getCollisionTime(TimeC); }
  bool isValidTime(int side) const { return getCollisionTime(side) < o2::InteractionRecord::DummyTime; }
  void setCollisionTime(Float_t time, int side) { mCollisionTime[side] = time; }

  Float_t getVertex(Float_t vertex) const { return getCollisionTime(Vertex); }
  void setVertex(Float_t vertex) { mCollisionTime[Vertex] = vertex; }

  o2::ft0::Triggers getTrigger() const { return mTriggers; }

  o2::InteractionRecord getInteractionRecord() const { return mIntRecord; };

  void SetMgrEventTime(Double_t time) { mTimeStamp = time; }

  gsl::span<const ChannelDataFloat> getBunchChannelData(const gsl::span<const ChannelDataFloat> tfdata) const;

 private:
  std::array<Float_t, 4> mCollisionTime = {2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime};
  Double_t mTimeStamp = 2 * o2::InteractionRecord::DummyTime; //event time from Fair for continuous

  ClassDefNV(RecPoints, 2);
};
} // namespace ft0
} // namespace o2
#endif
