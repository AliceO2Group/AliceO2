// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   T0RecPointWriterSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "FITWorkflow/T0RecPointWriterSpec.h"
#include "DataFormatsFITT0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace t0
{

void T0RecPointWriter::init(InitContext& ic)
{
}

void T0RecPointWriter::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }
  // no MC infor treatment at the moment
  auto recPoints = pc.inputs().get<const std::vector<o2::t0::RecPoints>>("recpoints");
  auto recPointsPtr = &recPoints;
  LOG(INFO) << "T0RecPointWriter pulled " << recPoints.size() << " RecPoints";

  TFile flOut(mOutputFileName.c_str(), "recreate");
  if (flOut.IsZombie()) {
    LOG(FATAL) << "Failed to create T0 RecPoints output file " << mOutputFileName;
  }
  TTree tree(mOutputTreeName.c_str(), "Tree with T0 RecPoints");
  tree.Branch(mRPOutputBranchName.c_str(), &recPointsPtr);
  tree.Fill();
  tree.Write();

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getT0RecPointWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputSpec;
  inputSpec.emplace_back("recpoints", o2::header::gDataOriginT0, "RECPOINTS", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "t0-recpoint-writer",
    inputSpec,
    Outputs{},
    AlgorithmSpec{ adaptFromTask<T0RecPointWriter>(useMC) },
    Options{
      { "t0-recpoint-outfile", VariantType::String, "o2reco_t0.root", { "Name of the output file" } },
      { "t0-recpoint-tree-name", VariantType::String, "o2sim", { "Name of the T0 recpoints tree" } },
      { "t0-recpoint-branch-name", VariantType::String, "T0Cluster", { "Name of the T0 recpoints branch" } },
    }
  };
}

} // namespace t0
} // namespace o2
