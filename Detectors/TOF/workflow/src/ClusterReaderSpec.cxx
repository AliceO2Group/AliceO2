// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "TOFWorkflow/ClusterReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void ClusterReader::init(InitContext& ic)
{
  LOG(INFO) << "Init Cluster reader!";
  auto filename = ic.options().get<std::string>("tof-cluster-infile");
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    mState = 0;
    return;
  }
  mState = 1;
}

void ClusterReader::run(ProcessingContext& pc)
{
  if (mState != 1) {
    return;
  }

  std::unique_ptr<TTree> treeClu((TTree*)mFile->Get("o2sim"));

  if (treeClu) {
    treeClu->SetBranchAddress("TOFCluster", &mPclusters);

    if (mUseMC) {
      treeClu->SetBranchAddress("TOFClusterMCTruth", &mPlabels);
    }

    treeClu->GetEntry(0);

    // add clusters loaded in the output snapshot
    pc.outputs().snapshot(Output{ "TOF", "CLUSTERS", 0, Lifetime::Timeframe }, mClusters);
    if (mUseMC)
      pc.outputs().snapshot(Output{ "TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe }, mLabels);

    static o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::CONTINUOUS;

    LOG(INFO) << "TOF: Sending ROMode= " << roMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{ "TOF", "ROMode", 0, Lifetime::Timeframe }, roMode);
  } else {
    LOG(ERROR) << "Cannot read the TOF clusters !";
    return;
  }

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getClusterReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TOF", "CLUSTERS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("TOF", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-cluster-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{ adaptFromTask<ClusterReader>(useMC) },
    Options{
      { "tof-cluster-infile", VariantType::String, "tofclusters.root", { "Name of the input file" } } }
  };
}

} // namespace tof
} // namespace o2
