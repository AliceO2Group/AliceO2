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

#ifndef O2_HYPERTRACKING_SPEC_H
#define O2_HYPERTRACKING_SPEC_H

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

#include "TStopwatch.h"

#include "StrangenessTracking/HyperTracker.h"

namespace o2
{
namespace strangeness_tracking
{
class HypertrackerSpec : public framework::Task
{
 public:
  using ITSCluster = o2::BaseCluster<float>;

  HypertrackerSpec(bool isMC = false);
  ~HypertrackerSpec() override = default;

  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  bool mIsMC = false;
  bool mRecreateV0 = true;
  TStopwatch mTimer;
  HyperTracker mTracker;
  std::unique_ptr<parameters::GRPObject> mGRP = nullptr;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
};

o2::framework::DataProcessorSpec getHyperTrackerSpec();
o2::framework::WorkflowSpec getWorkflow(bool upstreamClusters = false, bool upstreamV0s = false);

} // namespace strangeness_tracking
} // namespace o2
#endif