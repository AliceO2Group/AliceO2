// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DATADECODERSPEC_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DATADECODERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Common.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"

namespace o2
{
namespace hmpid
{

class DataDecoderTask2 : public framework::Task
{
 public:
  DataDecoderTask2() = default;
  ~DataDecoderTask2() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void decodeTF(framework::ProcessingContext& pc);
  void decodeReadout(framework::ProcessingContext& pc);
  void decodeRawFile(framework::ProcessingContext& pc);
  void endOfStream(framework::EndOfStreamContext& ec) override;
  void orderTriggers();

 private:
  HmpidDecoder2* mDeco;
  long mTotalDigits;
  long mTotalFrames;
  std::string mRootStatFile;
  bool mFastAlgorithm;

  ExecutionTimer mExTimer;
  std::vector<o2::hmpid::Trigger> mTriggers;
};

o2::framework::DataProcessorSpec getDecodingSpec2(bool askSTFDist);
} // end namespace hmpid
} // end namespace o2

#endif
