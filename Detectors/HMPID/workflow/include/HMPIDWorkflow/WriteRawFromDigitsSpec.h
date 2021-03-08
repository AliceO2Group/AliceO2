// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_WRITERAWFROMDIGITS_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_WRITERAWFROMDIGITS_H_

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "HMPIDBase/Common.h"
#include "HMPIDBase/Digit.h"
#include "HMPIDSimulation/HmpidCoder.h"

namespace o2
{
namespace hmpid
{

class WriteRawFromDigitsTask : public framework::Task
{
 public:
  WriteRawFromDigitsTask() = default;
  ~WriteRawFromDigitsTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  //     static bool eventEquipPadsComparision(o2::hmpid::Digit d1, o2::hmpid::Digit d2);
  std::string mBaseFileName = "";
  std::vector<o2::hmpid::Digit> mDigits;
  bool mSkipEmpty = false;
  bool mFixedPacketLenght = false;
  bool mOrderTheEvents = true;
  long mDigitsReceived;
  long mFramesReceived;
  bool mIsTheStremClosed = false;
  HmpidCoder* mCod;

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getWriteRawFromDigitsSpec(std::string inputSpec = "HMP/DIGITS");

} // end namespace hmpid
} // end namespace o2

#endif
