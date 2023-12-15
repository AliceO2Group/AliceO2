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

/// @file   TRDTrackletReaderSpec.cxx

#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "fairlogger/Logger.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDTrackletReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(info) << "Init TRD tracklet reader!";
  mInFileNameTrklt = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("trd-tracklet-infile"));
  mInFileNameCTrklt = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("trd-calib-tracklet-infile"));
  mInTreeNameTrklt = ic.options().get<std::string>("treename");
  connectTree();
  if (mUseTrackletTransform) {
    connectTreeCTracklet();
  }
}

void TRDTrackletReader::connectTreeCTracklet()
{
  mTreeCTrklt.reset(nullptr); // in case it was already loaded
  mFileCTrklt.reset(TFile::Open(mInFileNameCTrklt.c_str()));
  assert(mFileCTrklt && !mFileCTrklt->IsZombie());
  mTreeCTrklt.reset((TTree*)mFileCTrklt->Get("ctracklets"));
  assert(mTreeCTrklt);
  mTreeCTrklt->SetBranchAddress("CTracklets", &mTrackletsCalPtr);
  mTreeCTrklt->SetBranchAddress("TrigRecMask", &mTrigRecMaskPtr);
  LOG(info) << "Loaded tree from trdcalibratedtracklets.root with " << mTreeCTrklt->GetEntries() << " entries";
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
  LOG(info) << "Loaded tree from " << mInFileNameTrklt << " with " << mTreeTrklt->GetEntries() << " entries";
}

void TRDTrackletReader::run(ProcessingContext& pc)
{
  auto currEntry = mTreeTrklt->GetReadEntry() + 1;
  assert(currEntry < mTreeTrklt->GetEntries()); // this should not happen
  mTreeTrklt->GetEntry(currEntry);
  LOG(info) << "Pushing " << mTriggerRecords.size() << " TRD trigger records at entry " << currEntry;
  LOG(info) << "Pushing " << mTracklets.size() << " uncalibrated TRD tracklets for these trigger records";
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0}, mTracklets);
  if (mUseTrackletTransform) {
    assert(mTreeTrklt->GetEntries() == mTreeCTrklt->GetEntries());
    mTreeCTrklt->GetEntry(currEntry);
    LOG(info) << "Pushing " << mTrackletsCal.size() << " calibrated TRD tracklets for these trigger records";
    LOG(info) << "Pushing " << mTrigRecMask.size() << " flags for the given TRD trigger records";
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "CTRACKLETS", 0}, mTrackletsCal);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRIGRECMASK", 0}, mTrigRecMask);
  }

  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0}, mTriggerRecords);
  if (mUseMC) {
    LOG(info) << "Pushing " << mLabels.getNElements() << " TRD tracklet labels";
    pc.outputs().snapshot(Output{"TRD", "TRKLABELS", 0}, mLabels);
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
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRIGRECMASK", 0, Lifetime::Timeframe);
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
      {"trd-tracklet-infile", VariantType::String, "trdtracklets.root", {"Name of the tracklets input file"}},
      {"trd-calib-tracklet-infile", VariantType::String, "trdcalibratedtracklets.root", {"Name of the calibrated tracklets input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "o2sim", {"Name of top-level TTree"}},
    }};
}

} // namespace trd
} // namespace o2
