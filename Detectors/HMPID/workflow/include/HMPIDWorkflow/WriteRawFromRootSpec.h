// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_WRITERAWFROMROOT_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_WRITERAWFROMROOT_H_

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"

#include "HMPIDBase/Common.h"
#include "HMPIDBase/Digit.h"
#include "HMPIDSimulation/HmpidCoder2.h"

namespace o2
{
namespace hmpid
{

class WriteRawFromRootTask : public framework::Task
{
 public:
  WriteRawFromRootTask() = default;
  ~WriteRawFromRootTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  void readRootFile();
  std::string mBaseFileName = "";
  std::string mBaseRootFileName = "";
  bool mSkipEmpty = false;
  bool mDumpDigits = false;
  bool mPerFlpFile = false;

  std::vector<o2::hmpid::Digit> mDigits;
  long mDigitsReceived;
  int mEventsReceived;
  HmpidCoder2* mCod;
  ExecutionTimer mExTimer;
  TTree* mDigTree;
};

o2::framework::DataProcessorSpec getWriteRawFromRootSpec(std::string inputSpec = "HMP/DIGITS");

} // end namespace hmpid
} // end namespace o2

#endif
