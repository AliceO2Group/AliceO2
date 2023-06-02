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

/// @file   HMPMatchedReaderSpec.cxx

#include <vector>

#include "TTree.h"
#include "TFile.h"

#include "HMPIDWorkflow/HMPMatchedReaderSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/DataHeader.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/MatchInfoHMP.h"
#include "ReconstructionDataFormats/MatchingType.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/NameConf.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

using namespace o2::framework;

namespace o2
{
namespace hmpid
{
// static constexpr o2::header::DataDescription ddMatchInfo[4] = {"MTC_TPC", "MTC_ITSTPC", "MTC_TPCTRD", "MTC_ITSTPCTRD"};
// static constexpr o2::header::DataDescription ddMCMatchTOF[4] = {"MCMTC_TPC", "MCMTC_ITSTPC", "MCMTC_TPCTRD", "MCMTC_ITSTPCTRD"};
void HMPMatchedReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(debug) << "Init HMPID matching info reader!";
  mInFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("hmpid-matched-infile"));
  mInTreeName = ic.options().get<std::string>("treename");
  connectTree(mInFileName);
}

void HMPMatchedReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mInTreeName.c_str()));
  assert(mTree);
  mTree->SetBranchAddress("HMPMatchInfo", &mMatchesPtr);

  if (mUseMC) {
    mTree->SetBranchAddress("MatchHMPMCTruth", &mLabelHMPPtr);
  }
  LOG(debug) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

void HMPMatchedReader::run(ProcessingContext& pc)
{
  auto currEntry = mTree->GetReadEntry() + 1;
  assert(currEntry < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(currEntry);
  LOG(debug) << "Pushing " << mMatches.size() << " HMP matchings at entry " << currEntry;

  // uint32_t tpcMatchSS = o2::globaltracking::getSubSpec(mSubSpecStrict && (!mMode) ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  pc.outputs().snapshot(Output{o2::header::gDataOriginHMP, "CONSTRAINED", 0, Lifetime::Timeframe}, mMatches);
  /*if (mReadTracks && (!mMode)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "TOFTRACKS_TPC", tpcMatchSS, Lifetime::Timeframe}, mTracks);
  }*/
  if (mUseMC) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginHMP, "CONSTRAINED_MC", 0, Lifetime::Timeframe}, mLabelHMP);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getHMPMatchedReaderSpec(bool useMC, int mode)
{
  /*const char* match_name[4] = {"TOFMatchedReader_TPC", "TOFMatchedReader_ITSTPC", "TOFMatchedReader_TPCTRD", "TOFMatchedReader_ITSTPCTRD"};
  const char* match_name_strict[4] = {"TOFMatchedReader_TPC_str", "TOFMatchedReader_ITSTPC_str", "TOFMatchedReader_TPCTRD_str", "TOFMatchedReader_ITSTPCTRD_str"};
  const char* file_name[4] = {"o2match_tof_tpc.root", "o2match_tof_itstpc.root", "o2match_tof_tpctrd.root", "o2match_tof_itstpctrd.root"};
  const char* taskName = match_name[mode];
  const char* fileName = file_name[mode];
  if (subSpecStrict) {
    taskName = match_name_strict[mode];
  }
 */

  std::vector<OutputSpec> outputs;
  // uint32_t tpcMatchSS = o2::globaltracking::getSubSpec(subSpecStrict && (!mode) ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  outputs.emplace_back(o2::header::gDataOriginHMP, "CONSTRAINED", 0, Lifetime::Timeframe);
  /* if (!mode && readTracks) {
     outputs.emplace_back(o2::header::gDataOriginTOF, "TOFTRACKS_TPC", tpcMatchSS, Lifetime::Timeframe);
   }*/
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginHMP, "CONSTRAINED_MC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "HMPMatchedReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<HMPMatchedReader>(useMC)},
    Options{
      {"hmp-matched-infile", VariantType::String, "o2match_hmpid.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "matchHMP", {"Name of top-level TTree"}},
    }};
}
} // namespace hmpid
} // namespace o2
