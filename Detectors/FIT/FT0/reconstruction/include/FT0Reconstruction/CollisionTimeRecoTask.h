// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FIT collision time reconstruction task
#ifndef ALICEO2_FIT_COLLISIONTIMERECOTASK_H
#define ALICEO2_FIT_COLLISIONTIMERECOTASK_H

#include <vector>
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace ft0
{
class CollisionTimeRecoTask
{

 public:
  enum : int { TimeMean,
               TimeA,
               TimeC };
  CollisionTimeRecoTask() = default;
  ~CollisionTimeRecoTask() = default;
  void Process(const std::vector<o2::ft0::Digit>& digitsBC,
               const std::vector<o2::ft0::ChannelData>& digitsCh,
               RecPoints& recPoints) const;
  void FinishTask();
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  std::array<Float_t, 3> mCollisionTime = {2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime};
  Float_t mVertex = 0;
  Double_t mTimeStamp = 2 * o2::InteractionRecord::DummyTime; //event time from Fair for continuous
 
 private:
  ClassDefNV(CollisionTimeRecoTask, 1);
};
} // namespace ft0
} // namespace o2
#endif
