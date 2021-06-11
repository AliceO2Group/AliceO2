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

#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "fairlogger/Logger.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDTrackletReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(INFO) << "Init TRD tracklet reader!";
  mInFileNameTrklt = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                   ic.options().get<std::string>("trd-tracklet-infile"));
  mInTreeNameTrklt = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                   ic.options().get<std::string>("treename"));
  connectTree();
  if (mUseTrackletTransform) {
    connectTreeCTracklet();
  }
}

void TRDTrackletReader::connectTreeCTracklet()
{
  mTreeCTrklt.reset(nullptr); // in case it was already loaded
  mFileCTrklt.reset(TFile::Open("trdcalibratedtracklets.root"));
  assert(mFileCTrklt && !mFileCTrklt->IsZombie());
  mTreeCTrklt.reset((TTree*)mFileCTrklt->Get("ctracklets"));
  assert(mTreeCTrklt);
  mTreeCTrklt->SetBranchAddress("CTracklets", &mTrackletsCalPtr);
  LOG(INFO) << "Loaded tree from trdcalibratedtracklets.root with " << mTreeCTrklt->GetEntries() << " entries";
}

void TRDTrackletReader::connectTree()
{
  mTreeTrklt.reset(nullptr); // in case it was already loaded
  mFileTrklt.reset(TFile::Open(mInFileNameTrklt.c_str()));
  assert(mFileTrklt && !mFileTrklt->IsZombie());
  mTreeTrklt.reset((TTree*)mFileTrklt->Get(mInTreeNameTrklt.c_str()));
  assert(mTreeTrklt);
  mTreeTrklt->SetBranchAddress("Tracklet", &mTrackletsPtr);
  mTreeTrklt->SetBranchAddress("TrackTrg", &mTriggerRecordsPtr);
  if (mUseMC) {
    mTreeTrklt->SetBranchAddress("TRKLabels", &mLabelsPtr);
  }
  LOG(INFO) << "Loaded tree from " << mInFileNameTrklt << " with " << mTreeTrklt->GetEntries() << " entries";
}

void TRDTrackletReader::run(ProcessingContext& pc)
{
  auto currEntry = mTreeTrklt->GetReadEntry() + 1;
  assert(currEntry < mTreeTrklt->GetEntries()); // this should not happen
  mTreeTrklt->GetEntry(currEntry);
  LOG(INFO) << "Pushing " << mTriggerRecords.size() << " TRD trigger records at entry " << currEntry;
  LOG(INFO) << "Pushing " << mTracklets.size() << " uncalibrated TRD tracklets for these trigger records";
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe}, mTracklets);
  if (mUseTrackletTransform) {
    assert(mTreeTrklt->GetEntries() == mTreeCTrklt->GetEntries());
    mTreeCTrklt->GetEntry(currEntry);
    LOG(INFO) << "Pushing " << mTrackletsCal.size() << " calibrated TRD tracklets for these trigger records";
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe}, mTrackletsCal);
  }

  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe}, mTriggerRecords);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"TRD", "TRKLABELS", 0, Lifetime::Timeframe}, mLabels);
  }

  if (mTreeTrklt->GetReadEntry() + 1 >= mTreeTrklt->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTRDTrackletReaderSpec(bool useMC, bool useCalibratedTracklets)
{
  std::vector<OutputSpec> outputs;
  if (useCalibratedTracklets) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "CTRACKLETS", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TRD", "TRKLABELS", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TRDTrackletReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackletReader>(useMC, useCalibratedTracklets)},
    Options{
      {"trd-tracklet-infile", VariantType::String, "trdtracklets.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}

} // namespace trd
} // namespace o2
