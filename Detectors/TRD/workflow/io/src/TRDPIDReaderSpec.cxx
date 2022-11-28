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

/// @file  TRDPIDReaderSpec.cxx

#include "TRDWorkflowIO/TRDPIDReaderSpec.h"
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

void TRDPIDReader::init(InitContext& ic)
{

  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("pid-infile"));

  connectTree(mFileName);
}

void TRDPIDReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mPID.size() << " PIDs and " << mTrigRec.size() << " trigger records at entry " << ent;
  if (mUseMC) {
    if (mLabelsTrd.size() != mLabelsMatch.size()) {
      LOG(error) << "The number of labels for matches and for TRD tracks is different. " << mLabelsTrd.size() << " TRD labels vs. " << mLabelsMatch.size() << " match labels";
    }
    LOG(info) << "Pushing " << mLabelsTrd.size() << " MC labels at entry " << ent;
  }

  switch (mMode) {
    case Mode::TPCTRD:
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRDPID_TPC", 0, Lifetime::Timeframe}, mPID);
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRGREC_TPC", 0, Lifetime::Timeframe}, mTrigRec);
      if (mUseMC) {
        pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC", 0, Lifetime::Timeframe}, mLabelsMatch);
        pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC_TRD", 0, Lifetime::Timeframe}, mLabelsTrd);
      }
      break;

    case Mode::ITSTPCTRD:
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRDPID_ITSTPC", 0, Lifetime::Timeframe}, mPID);
      pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0, Lifetime::Timeframe}, mTrigRec);
      if (mUseMC) {
        pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe}, mLabelsMatch);
        pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe}, mLabelsTrd);
      }
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TRDPIDReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("pidTRD"));
  assert(mTree);
  mTree->SetBranchAddress("pid", &mPIDPtr);
  mTree->SetBranchAddress("trgrec", &mTrigRecPtr);
  if (mUseMC) {
    mTree->SetBranchAddress("labels", &mLabelsMatchPtr);
    mTree->SetBranchAddress("labelsTRD", &mLabelsTrdPtr);
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTRDPIDTPCReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRDPID_TPC", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRGREC_TPC", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC_TRD", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "trdpid-tpc-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDPIDReader>(useMC, TRDPIDReader::Mode::TPCTRD)},
    Options{
      {"pid-infile", VariantType::String, "trdpid_tpc.root", {"Name of the input file for TPC-TRD pid"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

DataProcessorSpec getTRDPIDGlobalReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRDPID_ITSTPC", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "trdpid-itstpc-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDPIDReader>(useMC, TRDPIDReader::Mode::ITSTPCTRD)},
    Options{
      {"pid-infile", VariantType::String, "trdpid_itstpc.root", {"Name of the input file for ITS-TPC-TRD pid"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace trd
} // namespace o2
