// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFMatchedReaderSpec.cxx

#include <vector>

#include "TTree.h"
#include "TFile.h"

#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/DataHeader.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "CommonUtils/StringUtils.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
o2::header::DataDescription ddMatchInfo{"MATCHINFOS"}, ddMatchInfo_tpc{"MATCHINFOS_TPC"},
  ddMCMatchTOF{"MCMATCHTOF"}, ddMCMatchTOF_tpc{"MCMATCHTOF_TPC"};

void TOFMatchedReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(INFO) << "Init TOF matching info reader!";
  mInFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("tof-matched-infile"));
  mInTreeName = ic.options().get<std::string>("treename");
  connectTree(mInFileName);
}

void TOFMatchedReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mInTreeName.c_str()));
  assert(mTree);
  mTree->SetBranchAddress("TOFMatchInfo", &mMatchesPtr);
  if (mReadTracks) {
    if (!mTree->GetBranch("TPCTOFTracks")) {
      throw std::runtime_error("TPC-TOF tracks are requested but not found in the tree");
    }
    mTree->SetBranchAddress("TPCTOFTracks", &mTracksPtr);
  }
  if (mUseMC) {
    mTree->SetBranchAddress("MatchTOFMCTruth", &mLabelTOFPtr);
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

void TOFMatchedReader::run(ProcessingContext& pc)
{
  auto currEntry = mTree->GetReadEntry() + 1;
  assert(currEntry < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(currEntry);
  LOG(INFO) << "Pushing " << mMatches.size() << " TOF matchings at entry " << currEntry;

  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, mTPCMatch ? ddMatchInfo_tpc : ddMatchInfo, 0, Lifetime::Timeframe}, mMatches);
  if (mReadTracks && mTPCMatch) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "TOFTRACKS_TPC", 0, Lifetime::Timeframe}, mTracks);
  }
  if (mUseMC) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, mTPCMatch ? ddMCMatchTOF_tpc : ddMCMatchTOF, 0, Lifetime::Timeframe}, mLabelTOF);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTOFMatchedReaderSpec(bool useMC, bool tpcmatch, bool readTracks)
{
  std::vector<OutputSpec> outputs;

  outputs.emplace_back(o2::header::gDataOriginTOF, tpcmatch ? ddMatchInfo_tpc : ddMatchInfo, 0, Lifetime::Timeframe);
  if (tpcmatch && readTracks) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "TOFTRACKS_TPC", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, tpcmatch ? ddMCMatchTOF_tpc : ddMCMatchTOF, 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    o2::utils::Str::concat_string("TOFMatchedReader_", tpcmatch ? "tpc" : "glo"),
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TOFMatchedReader>(useMC, tpcmatch, readTracks)},
    Options{
      {"tof-matched-infile", VariantType::String, tpcmatch ? "o2match_toftpc.root" : "o2match_tof.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "matchTOF", {"Name of top-level TTree"}},
    }};
}
} // namespace tof
} // namespace o2
