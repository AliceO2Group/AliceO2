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

/// \file DigitsToClustersSpec.h
/// \brief Implementation of clusterization for HMPID; read upstream/from file write upstream/to file

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DIGITSTOCLUSTERSPEC_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DIGITSTOCLUSTERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include "HMPIDBase/Common.h"
#include "HMPIDReconstruction/Clusterer.h"
#include "DataFormatsHMP/Cluster.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"

#include "TFile.h"
#include "TTree.h"

namespace o2
{
namespace hmpid
{

class DigitsToClustersTask : public framework::Task
{
 public:
  DigitsToClustersTask() = default;
  ~DigitsToClustersTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  bool mReadFile = false;
  std::string mSigmaCutPar;
  float mSigmaCut[7] = {4, 4, 4, 4, 4, 4, 4};

  std::unique_ptr<TFile> mFile; ///< input file containin the tree
  std::unique_ptr<TTree> mTree; ///< input tree

  std::unique_ptr<o2::hmpid::Clusterer> mRec; // ef: changed to smart-pointer
  long mDigitsReceived;
  long mClustersReceived;

  void initFileIn(const std::string& fileName);

  ExecutionTimer mExTimer;
  void strToFloatsSplit(std::string s, std::string delimiter, float* res,
                        int maxElem = 7);
};

// ef : read from stream by default:
o2::framework::DataProcessorSpec
  getDigitsToClustersSpec();

} // end namespace hmpid
} // end namespace o2

#endif
