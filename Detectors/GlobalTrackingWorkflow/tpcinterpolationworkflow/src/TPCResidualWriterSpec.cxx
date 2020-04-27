// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TPCResidualWriterSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "SpacePoints/TrackInterpolation.h"
#include "TPCInterpolationWorkflow/TPCResidualWriterSpec.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

template <typename T>
TBranch* getOrMakeBranch(TTree* tree, const char* brname, T* ptr)
{
  if (auto br = tree->GetBranch(brname)) {
    // return branch if it already exists
    br->SetAddress(static_cast<void*>(ptr));
    return br;
  }
  // otherwise make it
  return tree->Branch(brname, ptr);
}

void ResidualWriterTPC::init(InitContext& ic)
{
  mOutFileName = ic.options().get<std::string>("residuals-outfile");
  mFile = std::make_unique<TFile>(mOutFileName.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    throw std::runtime_error(o2::utils::concat_string("Failed to open TPC SCD residuals output file ", mOutFileName));
  }
  mTree = std::make_unique<TTree>(mTreeName.c_str(), "Tree of TPC residuals and reference tracks");
}

void ResidualWriterTPC::run(ProcessingContext& pc)
{

  auto tracks = std::move(pc.inputs().get<const std::vector<TrackData>>("tracks"));
  auto residuals = std::move(pc.inputs().get<const std::vector<TPCClusterResiduals>>("residuals"));

  auto tracksPtr = &tracks;
  auto residualsPtr = &residuals;

  LOG(INFO) << "ResidualWriterTPC pulled " << tracks.size() << " reference tracks and " << residuals.size() << " TPC cluster residuals";

  getOrMakeBranch(mTree.get(), mOutTracksBranchName.c_str(), &tracksPtr);
  getOrMakeBranch(mTree.get(), mOutResidualsBranchName.c_str(), &residualsPtr);

  if (mUseMC) {
    // TODO
  }

  mTree->Fill();
}

void ResidualWriterTPC::endOfStream(EndOfStreamContext& ec)
{
  LOG(INFO) << "Finalizing TPC SCD interpolation residuals writing";
  mTree->Write();
  mTree.release()->Delete();
  mFile->Close();
}

DataProcessorSpec getTPCResidualWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tracks", "GLO", "TPCINT_TRK", 0, Lifetime::Timeframe);
  inputs.emplace_back("residuals", "GLO", "TPCINT_RES", 0, Lifetime::Timeframe);
  if (useMC) {
    // TODO
  }
  return DataProcessorSpec{
    "tpc-residuals-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<ResidualWriterTPC>(useMC)},
    Options{
      {"residuals-outfile", VariantType::String, "o2residuals_tpc.root", {"Name of the output file"}}}};
}

} // namespace tpc
} // namespace o2
