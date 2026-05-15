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

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "ITS3Workflow/DigitReaderSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include <cassert>
#include <stdexcept>

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2::its3
{

ITS3DigitReader::ITS3DigitReader(bool useMC, bool doStag, bool useCalib) : mUseMC(useMC), mDoStaggering(doStag), mUseCalib(useCalib), mDetNameLC(mDetName = "IT3"), mDigTreeName("o2sim")
{
  mDigBranchName = mDetName + mDigBranchName;
  mDigROFBranchName = mDetName + mDigROFBranchName;
  mDigMCTruthBranchName = mDetName + mDigMCTruthBranchName;

  std::transform(mDetNameLC.begin(), mDetNameLC.end(), mDetNameLC.begin(), ::tolower);
}

void ITS3DigitReader::init(InitContext& ic)
{
  mFileName = ic.options().get<std::string>((mDetNameLC + "-digit-infile").c_str());
  connectTree(mFileName);
}

void ITS3DigitReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen

  mTree->GetEntry(ent);
  for (uint32_t iLayer = 0; iLayer < (mDoStaggering ? NLayers : 1); ++iLayer) {
    if (!mDigROFRec[iLayer] || !mDigits[iLayer]) {
      throw std::runtime_error("ITS3 digit reader requires all 7 layer branches to be present and populated in every entry");
    }
    LOG(info) << mDetName << "DigitReader pushes " << mDigROFRec[iLayer]->size() << " ROFRecords, " << mDigits[iLayer]->size() << " digits at entry " << ent << " on layer " << iLayer;
    pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", iLayer}, *mDigROFRec[iLayer]);
    pc.outputs().snapshot(Output{mOrigin, "DIGITS", iLayer}, *mDigits[iLayer]);
    if (mUseMC) {
      if (!mPLabels[iLayer]) {
        throw std::runtime_error("ITS3 digit reader requires MC truth branches for all 7 layers to be present and populated in every entry");
      }
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", iLayer});
      mPLabels[iLayer]->copyandflatten(sharedlabels);
      delete mPLabels[iLayer];
      mPLabels[iLayer] = nullptr;
    }
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

template <typename Ptr>
void ITS3DigitReader::setBranchAddress(const std::string& base, Ptr& addr, int layer)
{
  const auto name = getBranchName(base, layer);
  if (Int_t ret = mTree->SetBranchAddress(name.c_str(), &addr); ret != 0) {
    LOGP(fatal, "failed to set branch address for {} ret={}", name, ret);
  }
}

void ITS3DigitReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mDigTreeName.c_str()));
  assert(mTree);
  for (int iLayer = 0; iLayer < (mDoStaggering ? NLayers : 1); ++iLayer) {
    const auto rofBranchName = getBranchName(mDigROFBranchName, iLayer);
    const auto digBranchName = getBranchName(mDigBranchName, iLayer);
    if (!mTree->GetBranch(rofBranchName.c_str()) || !mTree->GetBranch(digBranchName.c_str())) {
      throw std::runtime_error("ITS3 digit reader requires all branches in the input file");
    }
    setBranchAddress(mDigROFBranchName, mDigROFRec[iLayer], iLayer);
    setBranchAddress(mDigBranchName, mDigits[iLayer], iLayer);
    if (mUseMC) {
      if (!mTree->GetBranch(getBranchName(mDigMCTruthBranchName, iLayer).c_str())) {
        throw std::runtime_error("ITS3 digit reader requires MC truth branches for all 7 layers in the input file");
      }
      if (!mPLabels[iLayer]) {
        setBranchAddress(mDigMCTruthBranchName, mPLabels[iLayer], iLayer);
      }
    }
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getITS3DigitReaderSpec(bool useMC, bool doStag, bool useCalib, std::string defname)
{
  std::vector<OutputSpec> outputSpec;
  for (uint32_t iLayer = 0; iLayer < (doStag ? ITS3DigitReader::NLayers : 1); ++iLayer) {
    outputSpec.emplace_back("IT3", "DIGITS", iLayer, Lifetime::Timeframe);
    outputSpec.emplace_back("IT3", "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (useMC) {
      outputSpec.emplace_back("IT3", "DIGITSMCTR", iLayer, Lifetime::Timeframe);
    }
  }

  return DataProcessorSpec{
    .name = "its3-digit-reader",
    .inputs = Inputs{},
    .outputs = outputSpec,
    .algorithm = AlgorithmSpec{adaptFromTask<ITS3DigitReader>(useMC, doStag, useCalib)},
    .options = Options{
      {"it3-digit-infile", VariantType::String, defname, {"Name of the input digit file"}}}};
}

} // namespace o2::its3
