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
void HMPMatchedReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(debug) << "Init HMPID matching info reader!";
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

  pc.outputs().snapshot(Output{o2::header::gDataOriginHMP, "MATCHES", 0, Lifetime::Timeframe}, mMatches);
  if (mUseMC) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginHMP, "MCLABELS", 0, Lifetime::Timeframe}, mLabelHMP);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getHMPMatchedReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginHMP, "MATCHES", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginHMP, "MCLABELS", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "HMPMatchedReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<HMPMatchedReader>(useMC)},
    /*Options{
      {"hmp-matched-infile", VariantType::String, "o2match_hmp.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "matchHMP", {"Name of top-level TTree"}},
    }*/
  };
}
} // namespace hmpid
} // namespace o2
