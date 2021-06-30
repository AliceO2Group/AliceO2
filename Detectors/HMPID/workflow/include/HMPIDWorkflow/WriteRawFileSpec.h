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

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_WRITERAWFROMDIGITS_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_WRITERAWFROMDIGITS_H_

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "HMPIDBase/Common.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDSimulation/HmpidCoder2.h"

namespace o2
{
namespace hmpid
{

class WriteRawFileTask : public framework::Task
{
 public:
  WriteRawFileTask() = default;
  ~WriteRawFileTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  //     static bool eventEquipPadsComparision(o2::hmpid::Digit d1, o2::hmpid::Digit d2);
  std::string mBaseFileName = "";
  std::vector<o2::hmpid::Digit> mDigits;
  std::vector<o2::hmpid::Trigger> mEvents;
  bool mSkipEmpty = false;
  bool mFixedPacketLenght = false;
  bool mOrderTheEvents = true;
  long mDigitsReceived;
  long mFramesReceived;
  bool mIsTheStremClosed = false;
  HmpidCoder2* mCod;

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getWriteRawFileSpec(std::string inputSpec = "HMP/DIGITS");

} // end namespace hmpid
} // end namespace o2

#endif
