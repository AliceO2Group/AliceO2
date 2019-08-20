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
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

template <typename T>
TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
{
  if (auto br = tree.GetBranch(brname.c_str())) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname.c_str(), ptr);
}

void TOFMatchedWriter::init(InitContext& ic)
{
  // get the option from the init context
  mOutFileName = ic.options().get<std::string>("tof-matched-outfile");
  mOutTreeName = ic.options().get<std::string>("treename");
  
}

void TOFMatchedWriter::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  TFile outf(mOutFileName.c_str(), "recreate");
  if (outf.IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << mOutFileName;
  }
  TTree tree(mOutTreeName.c_str(), "Tree of TOF matching infos");
  auto indata = pc.inputs().get<MatchOutputType>("tofmatching");
  LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();

  auto br = getOrMakeBranch(tree, "TOFMatchInfo", &indata);
  br->Fill();

  if (mUseMC) {
    auto labeltof = pc.inputs().get<std::vector<o2::MCCompLabel>>("matchtoflabels");
    auto labeltpc = pc.inputs().get<std::vector<o2::MCCompLabel>>("matchtpclabels");
    auto labelits = pc.inputs().get<std::vector<o2::MCCompLabel>>("matchitslabels");

    LOG(INFO) << "TOF LABELS GOT " << labeltof.size() << " LABELS ";
    LOG(INFO) << "TPC LABELS GOT " << labeltpc.size() << " LABELS ";
    LOG(INFO) << "ITS LABELS GOT " << labelits.size() << " LABELS ";
    // connect this to particular branches    
    auto labeltofbr = getOrMakeBranch(tree, "MatchTOFMCTruth", &labeltof);
    auto labeltpcbr = getOrMakeBranch(tree, "MatchTPCMCTruth", &labeltpc);
    auto labelitsbr = getOrMakeBranch(tree, "MatchITSMCTruth", &labelits);
    labeltofbr->Fill();
    labeltpcbr->Fill();
    labelitsbr->Fill();
  }

  tree.SetEntries(1);
  tree.Write();
  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getTOFMatchedWriterSpec(bool useMC)
{
 std::vector<InputSpec> inputs;
  inputs.emplace_back("tofmatching", "TOF", "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("matchtoflabels", "TOF", "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchtpclabels", "TOF", "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchitslabels", "TOF", "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
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
