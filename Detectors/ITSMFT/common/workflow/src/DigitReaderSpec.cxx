// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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

#include <string>
#include <vector>
#include <format>

#include <TTree.h>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsITSMFT/PhysTrigger.h"
#include "CommonUtils/NameConf.h"
#include "CommonDataFormat/IRFrame.h"
#include "CommonUtils/IRFrameSelector.h"
#include "CCDB/BasicCCDBManager.h"
#include <cassert>

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace itsmft
{

template <int N>
DigitReader<N>::DigitReader(bool useMC, bool doStag, bool useCalib, bool triggerOut) : mUseMC(useMC), mDoStaggering(doStag), mUseCalib(useCalib), mTriggerOut(triggerOut), mDetNameLC(mDetName = ID.getName()), mDigTreeName("o2sim")
{
  mDigitBranchName = mDetName + mDigitBranchName;
  mDigitROFBranchName = mDetName + mDigitROFBranchName;
  mCalibBranchName = mDetName + mCalibBranchName;

  mDigitMCTruthBranchName = mDetName + mDigitMCTruthBranchName;

  std::transform(mDetNameLC.begin(), mDetNameLC.end(), mDetNameLC.begin(), ::tolower);

  if (mDoStaggering) {
    mLayers = DPLAlpideParam<N>::getNLayers();
    mDigits.resize(mLayers, nullptr);
    mDigROFRec.resize(mLayers, nullptr);
    mPLabels.resize(mLayers, nullptr);
  }
}

template <int N>
void DigitReader<N>::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>((mDetNameLC + "-digit-infile").c_str()));
  if (ic.options().hasOption("ignore-irframes") && !ic.options().get<bool>("ignore-irframes")) {
    mUseIRFrames = true;
  }
  connectTree(mFileName);
}

template <int N>
void DigitReader<N>::run(ProcessingContext& pc)
{
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  const auto& alpideParam = o2::itsmft::DPLAlpideParam<N>::Instance();
  if (tinfo.globalRunNumberChanged && mUseIRFrames) { // new run is starting: 1st call
    // TODO: we have to find a way define CCDBInput for IRFrames mode only using DPL fetcher
    auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
    auto rlim = ccdb.getRunDuration(tinfo.runNumber);
    long ts = (rlim.first + rlim.second) / 2;
    if constexpr (N == o2::detectors::DetID::ITS) {
      ccdb.getForTimeStamp<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>>("ITS/Config/AlpideParam", ts);
      mROFBiasInBC = alpideParam.roFrameBiasInBC;
      mROFLengthInBC = alpideParam.roFrameLengthInBC;
      mNRUs = o2::itsmft::ChipMappingITS::getNRUs();
    } else {
      ccdb.getForTimeStamp<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>>("MFT/Config/AlpideParam", ts);
      mROFBiasInBC = alpideParam.roFrameBiasInBC;
      mROFLengthInBC = alpideParam.roFrameLengthInBC;
      mNRUs = o2::itsmft::ChipMappingMFT::getNRUs();
    }
  }
  gsl::span<const o2::dataformats::IRFrame> irFrames{};
  if (mUseIRFrames) {
    irFrames = pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("driverInfo");
  }

  auto ent = mTree->GetReadEntry();
  if (!mUseIRFrames) {
    ent++;
    assert(ent < mTree->GetEntries()); // this should not happen
    mTree->GetEntry(ent);
    for (uint32_t iLayer = 0; iLayer < mLayers; ++iLayer) {
      LOG(info) << mDetName << "DigitReader" << ((mDoStaggering) ? std::format(": {}", iLayer) : "") << " pushes " << mDigROFRec[iLayer]->size() << " ROFRecords, " << mDigits[iLayer]->size() << " digits at entry " << ent;
      pc.outputs().snapshot(Output{Origin, "DIGITSROF", iLayer}, *mDigROFRec[iLayer]);
      pc.outputs().snapshot(Output{Origin, "DIGITS", iLayer}, *mDigits[iLayer]);
      if (mUseMC) {
        auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{Origin, "DIGITSMCTR", iLayer});
        mPLabels[iLayer]->copyandflatten(sharedlabels);
        delete mPLabels[iLayer];
        mPLabels[iLayer] = nullptr;
        // read dummy MC2ROF vector to keep writer/readers backward compatible
        static std::vector<o2::itsmft::MC2ROFRecord> dummyMC2ROF;
        pc.outputs().snapshot(Output{Origin, "DIGITSMC2ROF", iLayer}, dummyMC2ROF);
      }
    }
    if (mUseCalib) {
      pc.outputs().snapshot(Output{Origin, "GBTCALIB", 0}, mCalib);
    }
    if (mTriggerOut) {
      std::vector<o2::itsmft::PhysTrigger> dummyTrig;
      pc.outputs().snapshot(Output{Origin, "PHYSTRIG", 0}, dummyTrig);
    }
    if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  } else { // need to select particulars IRs range, presumably from the same tree entry
           // TODO implement for staggering
    std::vector<o2::itsmft::Digit> digitsSel;
    std::vector<o2::itsmft::GBTCalibData> calibSel;
    std::vector<o2::itsmft::ROFRecord> digROFRecSel;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> digitLabelsSel;

    if (irFrames.size()) { // we assume the IRFrames are in the increasing order
      if (ent < 0) {
        ent++;
      }
      o2::utils::IRFrameSelector irfSel;
      irfSel.setSelectedIRFrames(irFrames, 0, 0, -mROFBiasInBC, true);
      const auto irMin = irfSel.getIRFrames().front().getMin(); // use processed IRframes for rough comparisons (possible shift!)
      const auto irMax = irfSel.getIRFrames().back().getMax();
      LOGP(info, "Selecting IRFrame {}-{}", irMin.asString(), irMax.asString());
      while (ent < mTree->GetEntries()) {
        // do we need to read a new entry?
        if (ent > mTree->GetReadEntry()) {
          if (mUseMC) {
            delete mPLabels[0];
            mPLabels[0] = nullptr;
            mConstLabels[0].clear();
            mTree->SetBranchAddress(mDigitMCTruthBranchName.c_str(), &mPLabels[0]);
          }
          mTree->GetEntry(ent);
          if (mUseMC) {
            mPLabels[0]->copyandflatten(mConstLabels[0]);
            delete mPLabels[0];
            mPLabels[0] = nullptr;
          }
        }
        std::vector<int> rofOld2New;
        rofOld2New.resize(mDigROFRec[0]->size(), -1);

        if (mDigROFRec[0]->front().getBCData() <= irMax && (mDigROFRec[0]->back().getBCData() + mROFLengthInBC - 1) >= irMin) { // there is an overlap
          for (int irof = 0; irof < (int)mDigROFRec[0]->size(); irof++) {
            const auto& rof = mDigROFRec[0]->at(irof);
            if (irfSel.check({rof.getBCData(), rof.getBCData() + mROFLengthInBC - 1}) != -1) {
              rofOld2New[irof] = (int)digROFRecSel.size();
              LOGP(debug, "Adding selected ROF {}", rof.getBCData().asString());
              digROFRecSel.push_back(rof);
              int offs = digitsSel.size();
              digROFRecSel.back().setFirstEntry(offs);
              std::copy(mDigits[0]->begin() + rof.getFirstEntry(), mDigits[0]->begin() + rof.getFirstEntry() + rof.getNEntries(), std::back_inserter(digitsSel));
              for (int id = 0; id < rof.getNEntries(); id++) { // copy MC info
                digitLabelsSel.addElements(id + offs, mConstLabels[0].getLabels(id + rof.getFirstEntry()));
              }
              if (mCalib.size() >= size_t((irof + 1) * mNRUs)) {
                std::copy(mCalib.begin() + irof * mNRUs, mCalib.begin() + (irof + 1) * mNRUs, std::back_inserter(calibSel));
              }
            }
          }
        }
        if (mDigROFRec[0]->back().getBCData() + mROFLengthInBC - 1 < irMax) { // need to check the next entry
          ent++;
          continue;
        }
        break; // push collected data
      }
    }
    pc.outputs().snapshot(Output{Origin, "DIGITSROF", 0}, digROFRecSel);
    pc.outputs().snapshot(Output{Origin, "DIGITS", 0}, digitsSel);
    if (mUseCalib) {
      pc.outputs().snapshot(Output{Origin, "GBTCALIB", 0}, calibSel);
    }
    if (mTriggerOut) {
      std::vector<o2::itsmft::PhysTrigger> dummyTrig;
      pc.outputs().snapshot(Output{Origin, "PHYSTRIG", 0}, dummyTrig);
    }
    if (mUseMC) {
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{Origin, "DIGITSMCTR", 0});
      digitLabelsSel.flatten_to(sharedlabels);
    }

    if (!irFrames.size() || irFrames.back().isLast()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }
}

template <int N>
void DigitReader<N>::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mDigTreeName.c_str()));
  assert(mTree);
  for (uint32_t iLayer = 0; iLayer < mLayers; ++iLayer) {
    setBranchAddress(mDigitROFBranchName, mDigROFRec[iLayer], iLayer);
    setBranchAddress(mDigitBranchName, mDigits[iLayer], iLayer);
    if (mUseMC) {
      if (!mTree->GetBranch(getBranchName(mDigitMCTruthBranchName, iLayer).c_str())) {
        throw std::runtime_error("MC data requested but not found in the tree");
      }
      if (!mPLabels[iLayer]) {
        setBranchAddress(mDigitMCTruthBranchName, mPLabels[iLayer], iLayer);
      }
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

template <int N>
std::string DigitReader<N>::getBranchName(const std::string& base, int index)
{
  if (mDoStaggering) {
    return base + "_" + std::to_string(index);
  }
  return base;
}

template <int N>
template <typename Ptr>
void DigitReader<N>::setBranchAddress(const std::string& base, Ptr& addr, int layer)
{
  const auto name = getBranchName(base, layer);
  if (Int_t ret = mTree->SetBranchAddress(name.c_str(), &addr); ret != 0) {
    LOGP(fatal, "failed to set branch address for {} ret={}", name, ret);
  }
}

namespace
{
template <int N>
std::vector<OutputSpec> makeOutChannels(bool mctruth, bool doStag, bool useCalib)
{
  constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  std::vector<OutputSpec> outputs;
  int nLayers = doStag ? o2::itsmft::DPLAlpideParam<N>::getNLayers() : 1;
  for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
    outputs.emplace_back(Origin, "DIGITS", iLayer, Lifetime::Timeframe);
    outputs.emplace_back(Origin, "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back(Origin, "DIGITSMC2ROF", iLayer, Lifetime::Timeframe);
      outputs.emplace_back(Origin, "DIGITSMCTR", iLayer, Lifetime::Timeframe);
    }
  }
  if (useCalib) {
    outputs.emplace_back(Origin, "GBTCALIB", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(Origin, "PHYSTRIG", 0, Lifetime::Timeframe);
  return outputs;
}
} // namespace

DataProcessorSpec getITSDigitReaderSpec(bool useMC, bool doStag, bool useCalib, bool useTriggers, std::string defname)
{
  return DataProcessorSpec{
    .name = "its-digit-reader",
    .inputs = Inputs{},
    .outputs = makeOutChannels<o2::detectors::DetID::ITS>(useMC, doStag, useCalib),
    .algorithm = AlgorithmSpec{adaptFromTask<ITSDigitReader>(useMC, doStag, useCalib, useTriggers)},
    .options = Options{
      {"its-digit-infile", VariantType::String, defname, {"Name of the input digit file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

DataProcessorSpec getMFTDigitReaderSpec(bool useMC, bool doStag, bool useCalib, bool useTriggers, std::string defname)
{
  return DataProcessorSpec{
    .name = "mft-digit-reader",
    .inputs = Inputs{},
    .outputs = makeOutChannels<o2::detectors::DetID::MFT>(useMC, doStag, useCalib),
    .algorithm = AlgorithmSpec{adaptFromTask<MFTDigitReader>(useMC, doStag, useCalib, useTriggers)},
    .options = Options{
      {"mft-digit-infile", VariantType::String, defname, {"Name of the input digit file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace itsmft
} // namespace o2
