// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace its
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

template <typename T>
TBranch* getOrMakeBranch(TTree* tree, const char* brname, T* ptr)
{
  if (auto br = tree->GetBranch(brname)) {
    br->SetAddress(static_cast<void*>(ptr));
    return br;
  }
  return tree->Branch(brname, ptr); // otherwise make it
}

void TrackWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("its-track-outfile");
  mFile = std::make_unique<TFile>(filename.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    throw std::runtime_error(o2::utils::concat_string("failed to open ITS tracks output file ", filename));
  }
  mTree = std::make_unique<TTree>("o2sim", "Tree with ITS tracks");
}

void TrackWriter::run(ProcessingContext& pc)
{
  auto tracks = pc.inputs().get<const std::vector<o2::its::TrackITS>>("tracks");
  auto clusIdx = pc.inputs().get<gsl::span<int>>("trackClIdx");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto vertices = pc.inputs().get<const std::vector<Vertex>>("vertices");
  auto verticesROF = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("verticesROF");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* plabels = nullptr;
  std::vector<int> clusIdxOut;
  clusIdxOut.reserve(clusIdx.size());
  std::copy(clusIdx.begin(), clusIdx.end(), std::back_inserter(clusIdxOut));

  LOG(INFO) << "ITSTrackWriter pulled " << tracks.size() << " tracks, in " << rofs.size() << " RO frames";

  auto tracksPtr = &tracks;
  getOrMakeBranch(mTree.get(), "ITSTrack", &tracksPtr);
  auto clusIdxOutPtr = &clusIdxOut;
  getOrMakeBranch(mTree.get(), "ITSTrackClusIdx", &clusIdxOutPtr);
  auto verticesPtr = &vertices;
  getOrMakeBranch(mTree.get(), "Vertices", &verticesPtr);
  auto verticesROFPtr = &verticesROF;
  getOrMakeBranch(mTree.get(), "VerticesROF", &verticesROFPtr);
  auto rofsPtr = &rofs;
  getOrMakeBranch(mTree.get(), "ITSTracksROF", &rofsPtr); // write ROFrecords vector to a tree

  std::vector<o2::itsmft::MC2ROFRecord> mc2rofs, *mc2rofsPtr = &mc2rofs;
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    plabels = labels.get();
    getOrMakeBranch(mTree.get(), "ITSTrackMCTruth", &plabels);

    const auto m2rvec = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
    mc2rofs.reserve(m2rvec.size());
    std::copy(m2rvec.begin(), m2rvec.end(), std::back_inserter(mc2rofs));
    getOrMakeBranch(mTree.get(), "ITSTracksMC2ROF", &mc2rofsPtr);
  }
  mTree->Fill();
}

void TrackWriter::endOfStream(EndOfStreamContext& ec)
{
  LOG(INFO) << "Finalizing ITS tracks writing";
  mTree->Write();
  mTree.release()->Delete();
  mFile->Close();
}

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tracks", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("vertices", "ITS", "VERTICES", 0, Lifetime::Timeframe);
  inputs.emplace_back("verticesROF", "ITS", "VERTICESROF", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("labels", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-track-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<TrackWriter>(useMC)},
    Options{
      {"its-track-outfile", VariantType::String, "o2trac_its.root", {"Name of the output file"}}}};
}

} // namespace its
} // namespace o2
