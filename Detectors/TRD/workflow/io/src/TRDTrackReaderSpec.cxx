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

/// @file  TRDTrackReaderSpec.cxx

#include "TRDWorkflowIO/TRDTrackReaderSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/SerializationMethods.h"
#include "CommonUtils/StringUtils.h"
#include "ReconstructionDataFormats/MatchingType.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

void TRDTrackReader::init(InitContext& ic)
{

  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("track-infile"));

  connectTree(mFileName);
}

void TRDTrackReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mTracks.size() << " tracks and " << mTrigRec.size() << " trigger records at entry " << ent;
  if (mUseMC) {
    if (mLabelsTrd.size() != mLabelsMatch.size()) {
      LOG(error) << "The number of labels for matches and for TRD tracks is different. " << mLabelsTrd.size() << " TRD labels vs. " << mLabelsMatch.size() << " match labels";
    }
    LOG(info) << "Pushing " << mLabelsTrd.size() << " MC labels at entry " << ent;
  }

  if (mMode == Mode::TPCTRD) {
    uint32_t ss = o2::globaltracking::getSubSpec(mSubSpecStrict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCH_TPC", ss}, mTracks);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRGREC_TPC", ss}, mTrigRec);
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC", ss}, mLabelsMatch);
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC_TRD", ss}, mLabelsTrd);
    }
  } else if (mMode == Mode::ITSTPCTRD) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MATCH_ITSTPC", 0}, mTracks);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0}, mTrigRec);
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0}, mLabelsMatch);
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0}, mLabelsTrd);
    }
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TRDTrackReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("tracksTRD"));
  assert(mTree);
  mTree->SetBranchAddress("tracks", &mTracksPtr);
  mTree->SetBranchAddress("trgrec", &mTrigRecPtr);
  if (mUseMC) {
    mTree->SetBranchAddress("labels", &mLabelsMatchPtr);
    mTree->SetBranchAddress("labelsTRD", &mLabelsTrdPtr);
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTRDTPCTrackReaderSpec(bool useMC, bool subSpecStrict)
{
  std::vector<OutputSpec> outputs;
  uint32_t sspec = o2::globaltracking::getSubSpec(subSpecStrict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  outputs.emplace_back(o2::header::gDataOriginTRD, "MATCH_TPC", sspec, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRGREC_TPC", sspec, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC", sspec, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC_TRD", sspec, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "tpctrd-track-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackReader>(useMC, TRDTrackReader::Mode::TPCTRD, subSpecStrict)},
    Options{
      {"track-infile", VariantType::String, "trdmatches_tpc.root", {"Name of the input file for TPC-TRD matches"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

DataProcessorSpec getTRDGlobalTrackReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "MATCH_ITSTPC", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "itstpctrd-track-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackReader>(useMC, TRDTrackReader::Mode::ITSTPCTRD)},
    Options{
      {"track-infile", VariantType::String, "trdmatches_itstpc.root", {"Name of the input file for ITS-TPC-TRD matches"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace trd
} // namespace o2
