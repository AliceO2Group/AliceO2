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

/// @file   ClustererSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "ITS3Workflow/ClustererSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "DataFormatsITS3/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

using namespace o2::framework;

namespace o2
{
namespace its3
{

void ClustererDPL::init(InitContext& ic)
{
  mClusterer = std::make_unique<o2::its3::Clusterer>();
  mClusterer->setNChips(o2::itsmft::ChipMappingITS::getNChips(o2::itsmft::ChipMappingITS::MB) + o2::itsmft::ChipMappingITS::getNChips(o2::itsmft::ChipMappingITS::OB) + 6); // FIXME
  mUseClusterDictionary = !ic.options().get<bool>("ignore-cluster-dictionary");
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mState = 1;
  mClusterer->print();
}

void ClustererDPL::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  gsl::span<const o2::itsmft::MC2ROFRecord> mc2rofs;
  gsl::span<const char> labelbuffer;
  if (mUseMC) {
    labelbuffer = pc.inputs().get<gsl::span<char>>("labels");
    mc2rofs = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
  }
  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);

  LOG(info) << "ITS3Clusterer pulled " << digits.size() << " digits, in "
            << rofs.size() << " RO frames";
  LOG(info) << "ITS3Clusterer pulled " << labels.getNElements() << " labels ";

  o2::itsmft::DigitPixelReader reader;
  reader.setSquashingDepth(mClusterer->getMaxROFDepthToSquash());
  reader.setSquashingDist(mClusterer->getMaxRowColDiffToMask()); // Sharing same parameter/logic with masking
  reader.setMaxBCSeparationToSquash(mClusterer->getMaxBCSeparationToSquash());
  reader.setDigits(digits);
  reader.setROFRecords(rofs);
  if (mUseMC) {
    reader.setMC2ROFRecords(mc2rofs);
    reader.setDigitsMCTruth(labels.getIndexedSize() > 0 ? &labels : nullptr);
  }
  reader.init();
  auto orig = o2::header::gDataOriginIT3;
  std::vector<o2::its3::CompClusterExt> clusCompVec;
  std::vector<o2::itsmft::ROFRecord> clusROFVec;
  std::vector<unsigned char> clusPattVec;

  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }
  mClusterer->process(mNThreads, reader, &clusCompVec, mPatterns ? &clusPattVec : nullptr, &clusROFVec, clusterLabels.get());
  pc.outputs().snapshot(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe}, clusCompVec);
  pc.outputs().snapshot(Output{orig, "CLUSTERSROF", 0, Lifetime::Timeframe}, clusROFVec);
  pc.outputs().snapshot(Output{orig, "PATTERNS", 0, Lifetime::Timeframe}, clusPattVec);

  if (mUseMC) {
    pc.outputs().snapshot(Output{orig, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, *clusterLabels.get()); // at the moment requires snapshot
    std::vector<o2::itsmft::MC2ROFRecord> clusterMC2ROframes(mc2rofs.size());
    for (int i = mc2rofs.size(); i--;) {
      clusterMC2ROframes[i] = mc2rofs[i]; // Simply, replicate it from digits ?
    }
    pc.outputs().snapshot(Output{orig, "CLUSTERSMC2ROF", 0, Lifetime::Timeframe}, clusterMC2ROframes);
  }

  // TODO: in principle, after masking "overflow" pixels the MC2ROFRecord maxROF supposed to change, nominally to minROF
  // -> consider recalculationg maxROF
  LOG(info) << "ITSClusterer pushed " << clusCompVec.size() << " clusters, in " << clusROFVec.size() << " RO frames";
}

///_______________________________________
void ClustererDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alppar");
    pc.inputs().get<o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>*>("cluspar");
    mClusterer->setContinuousReadOut(o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS));
    // settings for the fired pixel overflow masking
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    const auto& clParams = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance();
    if (clParams.maxBCDiffToMaskBias > 0 && clParams.maxBCDiffToSquashBias > 0) {
      LOGP(fatal, "maxBCDiffToMaskBias = {} and maxBCDiffToMaskBias = {} cannot be set at the same time. Either set masking or squashing with a BCDiff > 0", clParams.maxBCDiffToMaskBias, clParams.maxBCDiffToSquashBias);
    }
    auto nbc = clParams.maxBCDiffToMaskBias;
    nbc += mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS);
    mClusterer->setMaxBCSeparationToMask(nbc);
    mClusterer->setMaxRowColDiffToMask(clParams.maxRowColDiffToMask);
    // Squasher
    int rofBC = mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS); // ROF length in BC
    mClusterer->setMaxBCSeparationToSquash(rofBC + clParams.maxBCDiffToSquashBias);
    int nROFsToSquash = 0; // squashing disabled if no reset due to maxSOTMUS>0.
    if (clParams.maxSOTMUS > 0 && rofBC > 0) {
      nROFsToSquash = 2 + int(clParams.maxSOTMUS / (rofBC * o2::constants::lhc::LHCBunchSpacingMUS)); // use squashing
    }
    mClusterer->setMaxROFDepthToSquash(clParams.maxBCDiffToSquashBias > 0 ? nROFsToSquash : 0);
    mClusterer->print();
  }
  // we may have other params which need to be queried regularly
}

///_______________________________________
void ClustererDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("IT3", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated" << (!mUseClusterDictionary ? " but its using is disabled" : "");
    if (mUseClusterDictionary) {
      mClusterer->setDictionary((const o2::its3::TopologyDictionary*)obj);
    }
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSPARAM", 0)) {
    LOG(info) << "Cluster param updated";
    const auto& par = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
}

DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "IT3", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "IT3", "DIGITSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("cldict", "IT3", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("IT3/Calib/ClusterDictionary"));
  inputs.emplace_back("cluspar", "ITS", "CLUSPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/ClustererParam"));
  inputs.emplace_back("alppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              inputs,
                                                              true);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("IT3", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("IT3", "PATTERNS", 0, Lifetime::Timeframe);
  outputs.emplace_back("IT3", "CLUSTERSROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "IT3", "DIGITSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "IT3", "DIGITSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("IT3", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("IT3", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its3-clusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ClustererDPL>(ggRequest, useMC)},
    Options{
      {"ignore-cluster-dictionary", VariantType::Bool, false, {"do not use cluster dictionary, always store explicit patterns"}}}};
}

///_______________________________________
void ClustererDPL::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  mClusterer->print();
}

} // namespace its3
} // namespace o2
