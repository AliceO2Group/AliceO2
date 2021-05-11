// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackCosmicsReaderSpec.cxx

#include <vector>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "GlobalTrackingWorkflowReaders/TrackCosmicsReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/SerializationMethods.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;
using namespace o2::globaltracking;

namespace o2
{
namespace globaltracking
{
void TrackCosmicsReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("cosmics-infile"));
  connectTree(mFileName);
}

void TrackCosmicsReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mTracks.size() << " Cosmic Tracks at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "COSMICTRC", 0, Lifetime::Timeframe}, mTracks);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe}, mLabels);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TrackCosmicsReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("cosmics"));
  assert(mTree);
  mTree->SetBranchAddress("tracks", &mTracksPtr);
  if (mUseMC) {
    mTree->SetBranchAddress("MCTruth", &mLabelsPtr);
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTrackCosmicsReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "COSMICTRC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "cosmic-track-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TrackCosmicsReader>(useMC)},
    Options{
      {"cosmics-infile", VariantType::String, "cosmics.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace globaltracking
} // namespace o2
