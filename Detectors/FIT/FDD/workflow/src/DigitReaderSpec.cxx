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

/// @file   DigitReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "FDDWorkflow/DigitReaderSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "CommonUtils/NameConf.h"
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

  mFile.reset(TFile::Open(mInputFileName.c_str()));
  if (!mFile->IsOpen()) {
    LOG(error) << "Cannot open the " << mInputFileName.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2sim"));
  if (!mTree) {
    LOG(error) << "Did not find o2sim tree in " << mInputFileName.c_str();
    throw std::runtime_error("Did not fine o2sim file in FDD digits tree");
  }
}

void DigitReader::run(ProcessingContext& pc)
{
  std::vector<o2::fdd::Digit>* digitsBC = nullptr;
  std::vector<o2::fdd::ChannelData>* digitsCh = nullptr;
  std::vector<o2::fdd::DetTrigInput>* digitsTrig = nullptr;
  o2::dataformats::IOMCTruthContainerView* mcTruthRootBuffer = nullptr;

  mTree->SetBranchAddress(mDigitBCBranchName.c_str(), &digitsBC);
  mTree->SetBranchAddress(mDigitChBranchName.c_str(), &digitsCh);
  if (mTree->GetBranch(mTriggerBranchName.c_str())) {
    mTree->SetBranchAddress(mTriggerBranchName.c_str(), &digitsTrig);
  }

  if (mUseMC) {
    if (mTree->GetBranch(mDigitMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mDigitMCTruthBranchName.c_str(), &mcTruthRootBuffer);
      LOG(info) << "Will use MC-truth from " << mDigitMCTruthBranchName;
    } else {
      LOG(info) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  auto ent = mTree->GetReadEntry() + 1;
  // A timeframe holds no collision at all whenever the interaction rate is low enough, and
  // the tree then has no entry to read. Publish empty containers instead of reading past the
  // end and pushing branch addresses that GetEntry has not filled, so that the consumers
  // downstream still see the timeframe. (This used to be an assert, which is compiled out of
  // every production build since ENABLE_CASSERT defaults to OFF.)
  const bool noEntry = ent >= mTree->GetEntries();
  if (noEntry) {
    LOG(info) << "no entry to read, publishing empty output";
  } else {
    mTree->GetEntry(ent);
  }

  static const std::vector<o2::fdd::Digit> noDigitsBC;
  static const std::vector<o2::fdd::ChannelData> noDigitsCh;
  static const std::vector<o2::fdd::DetTrigInput> noDigitsTrig;
  const auto& digitsBCOut = noEntry ? noDigitsBC : *digitsBC;
  const auto& digitsChOut = noEntry ? noDigitsCh : *digitsCh;
  LOG(info) << "FDD DigitReader pushes " << digitsBCOut.size() << " digits";
  pc.outputs().snapshot(Output{mOrigin, "DIGITSBC", 0}, digitsBCOut);
  pc.outputs().snapshot(Output{mOrigin, "DIGITSCH", 0}, digitsChOut);

  if (mUseMC) {
    // TODO: To be replaced with sending ConstMCTruthContainer as soon as reco workflow supports it
    pc.outputs().snapshot(Output{mOrigin, "TRIGGERINPUT", 0}, noEntry ? noDigitsTrig : *digitsTrig);

    o2::dataformats::MCTruthContainer<o2::fdd::MCLabel> mcTruth;
    if (!noEntry) {
      std::vector<char> flatbuffer;
      mcTruthRootBuffer->copyandflatten(flatbuffer);
      mcTruth.restore_from(flatbuffer.data(), flatbuffer.size());
    }
    pc.outputs().snapshot(Output{mOrigin, "DIGITLBL", 0}, mcTruth);
  }
  if (noEntry || mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
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
