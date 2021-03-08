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

#include "CCDB/CcdbApi.h"

#include "HMPIDBase/Common.h"
#include "HMPIDReconstruction/HmpidDecodeRawMem.h"

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
  HmpidDecodeRawMem* mDeco;
  long mTotalDigits;
  long mTotalFrames;
  std::string mPedestalsBasePath;
  float mSigmaCut;
  std::string mPedestalTag;

  o2::ccdb::CcdbApi mDBapi;
  std::map<std::string, std::string> mDbMetadata; // can be empty
  bool mWriteToDB;

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getPedestalsCalculationSpec(std::string inputSpec = "TF:HMP/RAWDATA");
//o2::framework::DataProcessorSpec getDecodingSpec();
} // end namespace hmpid
} // end namespace o2

#endif
