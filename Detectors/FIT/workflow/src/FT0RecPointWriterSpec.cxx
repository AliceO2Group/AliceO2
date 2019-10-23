// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0RecPointWriterSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "FITWorkflow/FT0RecPointWriterSpec.h"
#include "DataFormatsFT0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

void FT0RecPointWriter::init(InitContext& ic)
{
}

void FT0RecPointWriter::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }
  // no MC infor treatment at the moment
  auto recPoints = pc.inputs().get<const std::vector<o2::ft0::RecPoints>>("recpoints");
  auto recPointsPtr = &recPoints;
  LOG(INFO) << "FT0RecPointWriter pulled " << recPoints.size() << " RecPoints";

  TFile flOut(mOutputFileName.c_str(), "recreate");
  if (flOut.IsZombie()) {
    LOG(FATAL) << "Failed to create FT0 RecPoints output file " << mOutputFileName;
  }
  TTree tree(mOutputTreeName.c_str(), "Tree with FT0 RecPoints");
  tree.Branch(mRPOutputBranchName.c_str(), &recPointsPtr);
  tree.Fill();
  tree.Write();

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getFT0RecPointWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputSpec;
  inputSpec.emplace_back("recpoints", o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "ft0-recpoint-writer",
    inputSpec,
    Outputs{},
    AlgorithmSpec{adaptFromTask<FT0RecPointWriter>(useMC)},
    Options{
      {"ft0-recpoint-outfile", VariantType::String, "o2reco_ft0.root", {"Name of the output file"}},
      {"ft0-recpoint-tree-name", VariantType::String, "o2sim", {"Name of the FT0 recpoints tree"}},
      {"ft0-recpoint-branch-name", VariantType::String, "FT0Cluster", {"Name of the FT0 recpoints branch"}},
    }};
}

} // namespace ft0
} // namespace o2
