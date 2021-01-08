// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TRDTrackletReaderSpec.cxx

#include "TRDWorkflow/TRDTrackletReaderSpec.h"

#include "Headers/DataHeader.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDTrackletReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(INFO) << "Init TRD tracklet reader!";
  mInFileName = ic.options().get<std::string>("trd-tracklet-infile");
  mInTreeName = ic.options().get<std::string>("treename");
  connectTree(mInFileName);
}

void TRDTrackletReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mInTreeName.c_str()));
  assert(mTree);
  mTree->SetBranchAddress("Tracklet", &mTrackletsPtr);
  mTree->SetBranchAddress("TrackTrg", &mTriggerRecordsPtr);
  if (mUseMC) {
    LOG(FATAL) << "MC information not yet included for TRD tracklets";
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

void TRDTrackletReader::run(ProcessingContext& pc)
{
  auto currEntry = mTree->GetReadEntry() + 1;
  assert(currEntry < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(currEntry);
  LOG(INFO) << "Pushing " << mTriggerRecords.size() << " TRD trigger records at entry " << currEntry;
  LOG(INFO) << "Pushing " << mTracklets.size() << " TRD tracklets for these trigger records";

  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe}, mTracklets);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe}, mTriggerRecords);
  if (mUseMC) {
    LOG(FATAL) << "MC information not yet included for TRD tracklets";
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTRDTrackletReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(FATAL) << "MC information not yet included for TRD tracklets";
  }

  return DataProcessorSpec{
    "TRDTrackletReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackletReader>(useMC)},
    Options{
      {"trd-tracklet-infile", VariantType::String, "trdtracklets.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}

} // namespace trd
} // namespace o2
