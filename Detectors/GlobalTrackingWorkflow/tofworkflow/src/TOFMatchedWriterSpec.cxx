// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFMatchedWriterSpec.cxx

#include "TOFWorkflow/TOFMatchedWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

template <typename T>
TBranch* getOrMakeBranch(TTree* tree, const char* brname, T* ptr)
{
  if (auto br = tree->GetBranch(brname)) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree->Branch(brname, ptr);
}

void TOFMatchedWriter::init(InitContext& ic)
{
  // get the option from the init context
  mOutFileName = ic.options().get<std::string>("tof-matched-outfile");
  mOutTreeName = ic.options().get<std::string>("treename");
  mFile = std::make_unique<TFile>(mOutFileName.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    throw std::runtime_error(o2::utils::concat_string("failed to open TOF macthes output file ", mOutFileName));
  }
  mTree = std::make_unique<TTree>(mOutTreeName.c_str(), "Tree of TOF matching infos");
}

void TOFMatchedWriter::run(ProcessingContext& pc)
{
  auto indata = pc.inputs().get<MatchOutputType>("tofmatching");
  LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();
  auto indataPtr = &indata;
  getOrMakeBranch(mTree.get(), "TOFMatchInfo", &indataPtr);
  std::vector<o2::MCCompLabel> labeltof, *labeltofPtr = &labeltof;
  std::vector<o2::MCCompLabel> labeltpc, *labeltpcPtr = &labeltpc;
  std::vector<o2::MCCompLabel> labelits, *labelitsPtr = &labelits;
  if (mUseMC) {
    labeltof = std::move(pc.inputs().get<std::vector<o2::MCCompLabel>>("matchtoflabels"));
    labeltpc = std::move(pc.inputs().get<std::vector<o2::MCCompLabel>>("matchtpclabels")); // RS why do we need to repead ITS/TPC labels ?
    labelits = std::move(pc.inputs().get<std::vector<o2::MCCompLabel>>("matchitslabels")); // They can be extracted from TPC-ITS matches
    LOG(INFO) << "TOF LABELS GOT " << labeltof.size() << " LABELS ";
    LOG(INFO) << "TPC LABELS GOT " << labeltpc.size() << " LABELS ";
    LOG(INFO) << "ITS LABELS GOT " << labelits.size() << " LABELS ";
    // connect this to particular branches
    getOrMakeBranch(mTree.get(), "MatchTOFMCTruth", &labeltofPtr);
    getOrMakeBranch(mTree.get(), "MatchTPCMCTruth", &labeltpcPtr);
    getOrMakeBranch(mTree.get(), "MatchITSMCTruth", &labelitsPtr);
  }
  mTree->Fill();
}

void TOFMatchedWriter::endOfStream(EndOfStreamContext& ec)
{
  LOG(INFO) << "Finalizing TOF matching info writing";
  mTree->Write();
  mTree.release()->Delete();
  mFile->Close();
}

DataProcessorSpec getTOFMatchedWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofmatching", o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("matchtoflabels", o2::header::gDataOriginTOF, "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchtpclabels", o2::header::gDataOriginTOF, "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchitslabels", o2::header::gDataOriginTOF, "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TOFMatchedWriter",
    inputs,
    {}, // no output
    AlgorithmSpec{adaptFromTask<TOFMatchedWriter>(useMC)},
    Options{
      {"tof-matched-outfile", VariantType::String, "o2match_tof.root", {"Name of the input file"}},
      {"treename", VariantType::String, "matchTOF", {"Name of top-level TTree"}},
    }};
}
} // namespace tof
} // namespace o2
