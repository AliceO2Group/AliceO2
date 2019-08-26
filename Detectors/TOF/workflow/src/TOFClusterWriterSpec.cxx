// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TODClusterWriterSpec.cxx

#include "TOFWorkflow/TOFClusterWriterSpec.h"
#include "Framework/ControlService.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "DataFormatsTOF/Cluster.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
template <typename T>
TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
{
  if (auto br = tree.GetBranch(brname.c_str())) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname.c_str(), ptr);
}

void ClusterWriter::init(InitContext& ic)
{
  // get the option from the init context
  mOutFileName = ic.options().get<std::string>("tof-cluster-outfile");
  mOutTreeName = ic.options().get<std::string>("treename");
}

void ClusterWriter::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  TFile outf(mOutFileName.c_str(), "recreate");
  if (outf.IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << mOutFileName;
  }
  TTree tree(mOutTreeName.c_str(), "Tree of TOF clusters");
  auto indata = pc.inputs().get<std::vector<o2::tof::Cluster>>("tofclusters");
  LOG(INFO) << "RECEIVED CLUSTERS SIZE " << indata.size();

  auto br = getOrMakeBranch(tree, "TOFCluster", &indata);
  br->Fill();

  if (mUseMC) {
    auto labeldata = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabels");
    LOG(INFO) << "TOF GOT " << labeldata->getNElements() << " LABELS ";
    auto labeldataraw = labeldata.get();
    // connect this to a particular branch

    auto labelbr = getOrMakeBranch(tree, "TOFClusterMCTruth", &labeldataraw);
    labelbr->Fill();
  }

  tree.SetEntries(1);
  tree.Write();
  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getTOFClusterWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofclusters", "TOF", "CLUSTERS", 0, Lifetime::Timeframe);
  if (useMC)
    inputs.emplace_back("tofclusterlabels", "TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFClusterWriter",
    inputs,
    {}, // no output
    AlgorithmSpec{adaptFromTask<ClusterWriter>(useMC)},
    Options{
      {"tof-cluster-outfile", VariantType::String, "tofclusters.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}
} // namespace tof
} // namespace o2
