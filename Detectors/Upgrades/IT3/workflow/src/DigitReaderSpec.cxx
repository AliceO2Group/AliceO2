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

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "ITS3Workflow/DigitReaderSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include <cassert>

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its3
{

DigitReader::DigitReader(o2::detectors::DetID id, bool useMC, bool useCalib)
{
  assert(id == o2::detectors::DetID::IT3);
  mDetNameLC = mDetName = id.getName();
  mDigTreeName = "o2sim";

  mDigitBranchName = mDetName + mDigitBranchName;
  mDigROFBranchName = mDetName + mDigROFBranchName;
  mCalibBranchName = mDetName + mCalibBranchName;

  mDigtMCTruthBranchName = mDetName + mDigtMCTruthBranchName;
  mDigtMC2ROFBranchName = mDetName + mDigtMC2ROFBranchName;

  mUseMC = useMC;
  mUseCalib = useCalib;
  std::transform(mDetNameLC.begin(), mDetNameLC.end(), mDetNameLC.begin(), ::tolower);
}

void DigitReader::init(InitContext& ic)
{
  mFileName = ic.options().get<std::string>((mDetNameLC + "-digit-infile").c_str());
  connectTree(mFileName);
}

void DigitReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen

  o2::dataformats::IOMCTruthContainerView* plabels = nullptr;
  if (mUseMC) {
    mTree->SetBranchAddress(mDigtMCTruthBranchName.c_str(), &plabels);
  }
  mTree->GetEntry(ent);
  LOG(INFO) << mDetName << "DigitReader pushes " << mDigROFRec.size() << " ROFRecords, "
            << mDigits.size() << " digits at entry " << ent;

  // This is a very ugly way of providing DataDescription, which anyway does not need to contain detector name.
  // To be fixed once the names-definition class is ready
  pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", 0, Lifetime::Timeframe}, mDigROFRec);
  pc.outputs().snapshot(Output{mOrigin, "DIGITS", 0, Lifetime::Timeframe}, mDigits);
  if (mUseCalib) {
    pc.outputs().snapshot(Output{mOrigin, "GBTCALIB", 0, Lifetime::Timeframe}, mCalib);
  }

  if (mUseMC) {
    auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", 0, Lifetime::Timeframe});
    plabels->copyandflatten(sharedlabels);
    delete plabels;
    pc.outputs().snapshot(Output{mOrigin, "DIGITSMC2ROF", 0, Lifetime::Timeframe}, mDigMC2ROFs);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void DigitReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mDigTreeName.c_str()));
  assert(mTree);

  mTree->SetBranchAddress(mDigROFBranchName.c_str(), &mDigROFRecPtr);
  mTree->SetBranchAddress(mDigitBranchName.c_str(), &mDigitsPtr);
  if (mUseCalib) {
    if (!mTree->GetBranch(mCalibBranchName.c_str())) {
      throw std::runtime_error("GBT calibration data requested but not found in the tree");
    }
    mTree->SetBranchAddress(mCalibBranchName.c_str(), &mCalibPtr);
  }
  if (mUseMC) {
    if (!mTree->GetBranch(mDigtMC2ROFBranchName.c_str()) || !mTree->GetBranch(mDigtMCTruthBranchName.c_str())) {
      throw std::runtime_error("MC data requested but not found in the tree");
    }
    mTree->SetBranchAddress(mDigtMC2ROFBranchName.c_str(), &mDigMC2ROFsPtr);
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getITS3DigitReaderSpec(bool useMC, bool useCalib, std::string defname)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("ITS", "DIGITS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "DIGITSROF", 0, Lifetime::Timeframe);
  if (useCalib) {
    outputSpec.emplace_back("ITS", "GBTCALIB", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputSpec.emplace_back("ITS", "DIGITSMCTR", 0, Lifetime::Timeframe);
    outputSpec.emplace_back("ITS", "DIGITSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-digit-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<ITS3DigitReader>(useMC, useCalib)},
    Options{
      {"its-digit-infile", VariantType::String, defname, {"Name of the input digit file"}}}};
}

} // namespace its3
} // namespace o2
