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

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "TRKWorkflow/DigitReaderSpec.h"
#include "TRKBase/AlmiraParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <string>

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace trk
{

DigitReader::DigitReader(o2::detectors::DetID id, bool useMC, bool useCalib)
{
  assert(id == o2::detectors::DetID::TRK || id == o2::detectors::DetID::FT3);
  mLayers = id == o2::detectors::DetID::TRK ? static_cast<int>(o2::trk::AlmiraParam::kNLayers) : o2::trk::constants::MLOTDisks::nLayers;
  mDetNameLC = mDetName = id.getName();
  mDigTreeName = "o2sim";

  mDigits.resize(mLayers, nullptr);
  mDigROFRec.resize(mLayers, nullptr);
  mPLabels.resize(mLayers, nullptr);

  mDigitBranchName = mDetName + mDigitBranchName;
  mDigROFBranchName = mDetName + mDigROFBranchName;
  mCalibBranchName = mDetName + mCalibBranchName;

  mDigtMCTruthBranchName = mDetName + mDigtMCTruthBranchName;

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

  static const std::vector<o2::trkft3::ROFRecord> noDigROFRec;
  static const std::vector<o2::trkft3::Digit> noDigits;
  for (int iLayer = 0; iLayer < mLayers; ++iLayer) {
    const auto& digROFRec = noEntry ? noDigROFRec : *mDigROFRec[iLayer];
    const auto& digits = noEntry ? noDigits : *mDigits[iLayer];
    LOG(info) << mDetName << "DigitReader on layer " << iLayer << " pushes " << digROFRec.size() << " ROFRecords, "
              << digits.size() << " digits at entry " << ent;

    pc.outputs().snapshot(Output{mOrigin, "DIGITSROF", static_cast<o2::framework::DataAllocator::SubSpecificationType>(iLayer)}, digROFRec);
    pc.outputs().snapshot(Output{mOrigin, "DIGITS", static_cast<o2::framework::DataAllocator::SubSpecificationType>(iLayer)}, digits);

    if (mUseMC) {
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{mOrigin, "DIGITSMCTR", static_cast<o2::framework::DataAllocator::SubSpecificationType>(iLayer)});
      if (noEntry) {
        o2::dataformats::MCTruthContainer<o2::MCCompLabel> noLabels;
        noLabels.flatten_to(sharedlabels);
      } else {
        mPLabels[iLayer]->copyandflatten(sharedlabels);
        delete mPLabels[iLayer];
        mPLabels[iLayer] = nullptr;
      }
    }
  }

  if (mUseCalib) {
    pc.outputs().snapshot(Output{mOrigin, "GBTCALIB", 0}, mCalib);
  }

  if (noEntry || mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
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

  for (int iLayer = 0; iLayer < mLayers; ++iLayer) {
    setBranchAddress(mDigROFBranchName, mDigROFRec[iLayer], iLayer);
    setBranchAddress(mDigitBranchName, mDigits[iLayer], iLayer);
    if (mUseMC) {
      const auto mctruthBranch = getBranchName(mDigtMCTruthBranchName, iLayer);
      if (!mTree->GetBranch(mctruthBranch.c_str())) {
        throw std::runtime_error("MC data requested but missing branch(es) at layer " + std::to_string(iLayer) +
                                 ": " + mctruthBranch);
      }
      setBranchAddress(mDigtMCTruthBranchName, mPLabels[iLayer], iLayer);
    }
  }

  if (mUseCalib) {
    if (!mTree->GetBranch(mCalibBranchName.c_str())) {
      throw std::runtime_error("GBT calibration data requested but not found in the tree");
    }
    setBranchAddress(mCalibBranchName, mCalibPtr);
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

std::string DigitReader::getBranchName(const std::string& base, int index) const
{
  if (index >= 0) {
    return base + "_" + std::to_string(index);
  }
  return base;
}

template <typename Ptr>
void DigitReader::setBranchAddress(const std::string& base, Ptr& addr, int layer)
{
  const auto name = getBranchName(base, layer);
  if (Int_t ret = mTree->SetBranchAddress(name.c_str(), &addr); ret != 0) {
    LOGP(fatal, "failed to set branch address for {} ret={}", name, ret);
  }
}

namespace
{
template <int DetID, typename Reader>
DataProcessorSpec getDigitReaderSpecT(bool useMC, bool useCalib, std::string defname)
{
  constexpr auto origin = DetID == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3;
  std::vector<OutputSpec> outputSpec;
  constexpr int NLayers = DetID == o2::detectors::DetID::TRK ? static_cast<int>(o2::trk::AlmiraParam::kNLayers) : o2::trk::constants::MLOTDisks::nLayers;
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    outputSpec.emplace_back(origin, "DIGITS", iLayer, Lifetime::Timeframe);
    outputSpec.emplace_back(origin, "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (useMC) {
      outputSpec.emplace_back(origin, "DIGITSMCTR", iLayer, Lifetime::Timeframe);
    }
  }
  if (useCalib) {
    outputSpec.emplace_back(origin, "GBTCALIB", 0, Lifetime::Timeframe);
  }

  std::string detNameLC = o2::detectors::DetID(DetID).getName();
  std::transform(detNameLC.begin(), detNameLC.end(), detNameLC.begin(), ::tolower);
  return DataProcessorSpec{
    .name = detNameLC + "-digit-reader",
    .inputs = Inputs{},
    .outputs = outputSpec,
    .algorithm = AlgorithmSpec{adaptFromTask<Reader>(useMC, useCalib)},
    .options = Options{
      {detNameLC + "-digit-infile", VariantType::String, defname, {"Name of the input digit file"}}}};
}
} // namespace

DataProcessorSpec getTRKDigitReaderSpec(bool useMC, bool useCalib, std::string defname)
{
  return getDigitReaderSpecT<o2::detectors::DetID::TRK, TRKDigitReader>(useMC, useCalib, std::move(defname));
}

DataProcessorSpec getFT3DigitReaderSpec(bool useMC, bool useCalib, std::string defname)
{
  return getDigitReaderSpecT<o2::detectors::DetID::FT3, FT3DigitReader>(useMC, useCalib, std::move(defname));
}

} // namespace trk
} // namespace o2
