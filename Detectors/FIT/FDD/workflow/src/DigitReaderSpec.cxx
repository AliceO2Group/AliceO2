// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "FDDWorkflow/DigitReaderSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include <vector>

using namespace o2::framework;
using namespace o2::fdd;

namespace o2
{
namespace fdd
{

DigitReader::DigitReader(bool useMC)
{
  mUseMC = useMC;
}

void DigitReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("fdd-digits-infile"));
}

void DigitReader::run(ProcessingContext& pc)
{
  std::vector<o2::fdd::Digit>* digitsBC = nullptr;
  std::vector<o2::fdd::ChannelData>* digitsCh = nullptr;
  std::vector<o2::fdd::DetTrigInput>* digitsTrig = nullptr;
  o2::dataformats::IOMCTruthContainerView* mcTruthRootBuffer = nullptr;

  { // load data from files
    TFile digFile(mInputFileName.c_str(), "read");
    if (digFile.IsZombie()) {
      LOG(FATAL) << "Failed to open FDD digits file " << mInputFileName;
    }
    TTree* digTree = (TTree*)digFile.Get(mDigitTreeName.c_str());
    if (!digTree) {
      LOG(FATAL) << "Failed to load FDD digits tree " << mDigitTreeName << " from " << mInputFileName;
    }
    LOG(INFO) << "Loaded FDD digits tree " << mDigitTreeName << " from " << mInputFileName;

    digTree->SetBranchAddress(mDigitBCBranchName.c_str(), &digitsBC);

    digTree->SetBranchAddress(mTriggerBranchName.c_str(), &digitsTrig);
    if (mUseMC) {
      if (digTree->GetBranch(mDigitChBranchName.c_str())) {
        digTree->SetBranchAddress(mDigitChBranchName.c_str(), &digitsCh);
      }
      if (digTree->GetBranch(mDigitMCTruthBranchName.c_str())) {
        digTree->SetBranchAddress(mDigitMCTruthBranchName.c_str(), &mcTruthRootBuffer);
        LOG(INFO) << "Will use MC-truth from " << mDigitMCTruthBranchName;
      } else {
        LOG(INFO) << "MC-truth is missing";
        mUseMC = false;
      }
    }
    digTree->GetEntry(0);
    delete digTree;
    digFile.Close();
  }

  LOG(INFO) << "FDD DigitReader pushes " << digitsBC->size() << " digits";
  pc.outputs().snapshot(Output{mOrigin, "DIGITSBC", 0, Lifetime::Timeframe}, *digitsBC);
  pc.outputs().snapshot(Output{mOrigin, "DIGITSCH", 0, Lifetime::Timeframe}, *digitsCh);

  if (mUseMC) {
    // TODO: To be replaced with sending ConstMCTruthContainer as soon as reco workflow supports it
    pc.outputs().snapshot(Output{mOrigin, "TRIGGERINPUT", 0, Lifetime::Timeframe}, *digitsTrig);

    std::vector<char> flatbuffer;
    mcTruthRootBuffer->copyandflatten(flatbuffer);
    o2::dataformats::MCTruthContainer<o2::fdd::MCLabel> mcTruth;
    mcTruth.restore_from(flatbuffer.data(), flatbuffer.size());
    pc.outputs().snapshot(Output{mOrigin, "DIGITLBL", 0, Lifetime::Timeframe}, mcTruth);
  }

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getFDDDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITSBC", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back(o2::header::gDataOriginFDD, "TRIGGERINPUT", 0, Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITLBL", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "fdd-digit-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<DigitReader>()},
    Options{
      {"fdd-digits-infile", VariantType::String, "fdddigits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace fdd
} // namespace o2
