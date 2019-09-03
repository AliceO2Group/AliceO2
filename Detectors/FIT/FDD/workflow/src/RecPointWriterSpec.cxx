// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecPointWriterSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "FDDWorkflow/RecPointWriterSpec.h"
#include "DataFormatsFDD/RecPoint.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{

void FDDRecPointWriter::init(InitContext& ic)
{
}

void FDDRecPointWriter::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }
  // no MC infor treatment at the moment
  auto recPoints = pc.inputs().get<const std::vector<o2::fdd::RecPoint>>("recpoints");
  auto recPointsPtr = &recPoints;
  LOG(INFO) << "FDDRecPointWriter pulled " << recPoints.size() << " RecPoints";

  TFile flOut(mOutputFileName.c_str(), "recreate");
  if (flOut.IsZombie()) {
    LOG(FATAL) << "Failed to create FDD RecPoints output file " << mOutputFileName;
  }
  TTree tree(mOutputTreeName.c_str(), "Tree with FDD RecPoints");
  tree.Branch(mRPOutputBranchName.c_str(), &recPointsPtr);
  tree.Fill();
  tree.Write();

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getFDDRecPointWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputSpec;
  inputSpec.emplace_back("recpoints", o2::header::gDataOriginFDD, "RECPOINTS", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "fdd-recpoint-writer",
    inputSpec,
    Outputs{},
    AlgorithmSpec{adaptFromTask<FDDRecPointWriter>(useMC)},
    Options{
      {"fdd-recpoint-outfile", VariantType::String, "o2reco_fdd.root", {"Name of the output file"}},
      {"fdd-recpoint-tree-name", VariantType::String, "o2sim", {"Name of the FDD recpoints tree"}},
      {"fdd-recpoint-branch-name", VariantType::String, "FDDCluster", {"Name of the FDD recpoints branch"}},
    }};
}

} // namespace fdd
} // namespace o2
