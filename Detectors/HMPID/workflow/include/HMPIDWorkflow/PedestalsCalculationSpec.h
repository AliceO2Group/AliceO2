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

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_PEDESTALS_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_PEDESTALS_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "CCDB/CcdbApi.h"

#include "HMPIDBase/Common.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"

namespace o2
{
namespace hmpid
{

class PedestalsCalculationTask : public framework::Task
{
 public:
  PedestalsCalculationTask() = default;
  ~PedestalsCalculationTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void decodeTF(framework::ProcessingContext& pc);
  void decodeReadout(framework::ProcessingContext& pc);
  void decodeRawFile(framework::ProcessingContext& pc);
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  void recordPedInCcdb();

 private:
  HmpidDecoder2* mDeco;
  long mTotalDigits;
  long mTotalFrames;
  std::string mPedestalsBasePath;
  float mSigmaCut;
  std::string mPedestalTag;

  o2::ccdb::CcdbApi mDBapi;
  std::map<std::string, std::string> mDbMetadata; // can be empty
  bool mWriteToDB;
  bool mFastAlgorithm;

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getPedestalsCalculationSpec(std::string inputSpec = "TF:HMP/RAWDATA");
//o2::framework::DataProcessorSpec getDecodingSpec();
} // end namespace hmpid
} // end namespace o2

#endif
