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

/// @file   CalibClusReaderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "TOFWorkflowIO/CalibClusReaderSpec.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void CalibClusReader::init(InitContext& ic)
{
  LOG(debug) << "Init Cluster reader!";
  auto filename = ic.options().get<std::string>("tof-calclus-infile");
  connectTree(filename);
}

void CalibClusReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(debug) << "Pushing " << mPclusInfos->size() << " TOF clusters calib info at entry " << ent;
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOCALCLUS", 0}, mClusInfos);

  if (mIsCosmics) {
    LOG(debug) << "Pushing " << mPcosmicInfo->size() << " TOF cosmics info at entry " << ent;
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOCOSMICS", 0}, mCosmicInfo);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOTRACKCOS", 0}, mCosmicTrack);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "INFOTRACKSIZE", 0}, mCosmicTrackSize);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void CalibClusReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("o2sim"));
  assert(mTree);
  mTree->SetBranchAddress("TOFClusterCalInfo", &mPclusInfos);
  if (mIsCosmics) {
    mTree->SetBranchAddress("TOFCosmics", &mPcosmicInfo);
    mTree->SetBranchAddress("TOFTracks", &mPcosmicTrack);
    mTree->SetBranchAddress("TOFTracksSize", &mPcosmicTrackSize);
  }
  LOG(debug) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getCalibClusReaderSpec(bool isCosmics)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "INFOCALCLUS", 0, Lifetime::Timeframe);
  if (isCosmics) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOCOSMICS", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOTRACKCOS", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "INFOTRACKSIZE", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tof-calclus-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CalibClusReader>(isCosmics)},
    Options{
      {"tof-calclus-infile", VariantType::String, "tofclusCalInfo.root", {"Name of the input file"}}}};
}

} // namespace tof
} // namespace o2
