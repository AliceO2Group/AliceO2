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

#include "ITSMFTWorkflow/ClustererSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "Framework/InputRecordWalker.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSMFTReconstruction/ClustererParam.h"

namespace o2::itsmft
{

template <int N>
void ClustererDPL<N>::init(InitContext& ic)
{
  mClusterer = std::make_unique<o2::itsmft::Clusterer>();
  mClusterer->setNChips((N == o2::detectors::DetID::ITS) ? o2::itsmft::ChipMappingITS::getNChips() : o2::itsmft::ChipMappingMFT::getNChips());
  mUseClusterDictionary = !ic.options().get<bool>("ignore-cluster-dictionary");
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
  mDetName = Origin.as<std::string>();

  // prepare data filter
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    mFilter.emplace_back("digits", Origin, "DIGITS", iLayer, Lifetime::Timeframe);
    mFilter.emplace_back("ROframe", Origin, "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (mUseMC) {
      mFilter.emplace_back("labels", Origin, "DIGITSMCTR", iLayer, Lifetime::Timeframe);
      mFilter.emplace_back("MC2ROframes", Origin, "DIGITSMC2ROF", iLayer, Lifetime::Timeframe);
    }
  }
}

template <int N>
void ClustererDPL<N>::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);

  // filter input and compose
  std::array<gsl::span<const o2::itsmft::Digit>, NLayers> digits;
  std::array<gsl::span<const o2::itsmft::ROFRecord>, NLayers> rofs;
  std::array<gsl::span<const char>, NLayers> labelsbuffer;
  std::array<gsl::span<const o2::itsmft::MC2ROFRecord>, NLayers> mc2rofs;
  for (const DataRef& ref : InputRecordWalker{pc.inputs(), mFilter}) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (DataRefUtils::match(ref, {"digits", ConcreteDataTypeMatcher{Origin, "DIGITS"}})) {
      digits[dh->subSpecification] = pc.inputs().get<gsl::span<o2::itsmft::Digit>>(ref);
    }
    if (DataRefUtils::match(ref, {"ROframe", ConcreteDataTypeMatcher{Origin, "DIGITSROF"}})) {
      rofs[dh->subSpecification] = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>(ref);
    }
    if (DataRefUtils::match(ref, {"labels", ConcreteDataTypeMatcher{Origin, "DIGITSMCTR"}})) {
      labelsbuffer[dh->subSpecification] = pc.inputs().get<gsl::span<char>>(ref);
    }
    if (DataRefUtils::match(ref, {"MC2ROframes", ConcreteDataTypeMatcher{Origin, "DIGITSMC2ROF"}})) {
      mc2rofs[dh->subSpecification] = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>(ref);
    }
  }

  // query the first orbit in this TF
  const auto firstTForbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
  const o2::InteractionRecord firstIR(0, firstTForbit);
  const auto& par = DPLAlpideParam<N>::Instance();

  // process received inputs
  uint64_t nClusters{0};
  TStopwatch sw;
  o2::itsmft::DigitPixelReader reader;
  for (uint32_t iLayer{0}; iLayer < NLayers; ++iLayer) {
    int layer = (DPLAlpideParam<N>::supportsStaggering()) ? iLayer : -1;
    sw.Start();
    LOG(info) << mDetName << "Clusterer:" << layer << " pulled " << digits[iLayer].size() << " digits, in " << rofs[iLayer].size() << " RO frames";

    mClusterer->setMaxROFDepthToSquash(mClusterer->getMaxROFDepthToSquash(layer));
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelsbuffer[iLayer]);
    reader.setSquashingDepth(mClusterer->getMaxROFDepthToSquash(layer));
    reader.setSquashingDist(mClusterer->getMaxRowColDiffToMask()); // Sharing same parameter/logic with masking
    reader.setMaxBCSeparationToSquash(mClusterer->getMaxBCSeparationToSquash(layer));
    reader.setDigits(digits[iLayer]);
    reader.setROFRecords(rofs[iLayer]);
    if (mUseMC) {
      reader.setMC2ROFRecords(mc2rofs[iLayer]);
      LOG(info) << mDetName << "Clusterer:" << layer << " pulled " << labels.getNElements() << " labels ";
      reader.setDigitsMCTruth(labels.getIndexedSize() > 0 ? &labels : nullptr);
    }
    reader.init();
    std::vector<o2::itsmft::CompClusterExt> clusCompVec;
    std::vector<o2::itsmft::ROFRecord> clusROFVec;
    std::vector<unsigned char> clusPattVec;

    std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
    if (mUseMC) {
      clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
    }
    mClusterer->process(mNThreads, reader, &clusCompVec, &clusPattVec, &clusROFVec, clusterLabels.get());

    // ensure that the rof output is continuous
    size_t nROFs = clusROFVec.size();
    const int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / par.getROFLengthInBC(iLayer);
    const int nROFsTF = nROFsPerOrbit * o2::base::GRPGeomHelper::getNHBFPerTF();
    if (nROFsTF != clusROFVec.size()) {
      // it can happen that in the digitization rofs without contributing hits are skipped
      // however downstream consumers of the clusters cannot know apriori the time structure
      // the cluster rofs do not account for the bias so it will start always at BC=0
      // if we receive more cluster rofs then there supposed to be, do not throw away this data
      // the clusterer should be blind to this!
      const size_t nROFsLayer = std::max((size_t)nROFsTF, clusROFVec.size());
      std::vector<o2::itsmft::ROFRecord> expClusRofVec(nROFsLayer);
      for (int iROF{0}; iROF < nROFsLayer; ++iROF) {
        auto& rof = expClusRofVec[iROF];
        int orb = iROF * par.getROFLengthInBC(iLayer) / o2::constants::lhc::LHCMaxBunches + firstTForbit;
        int bc = iROF * par.getROFLengthInBC(iLayer) % o2::constants::lhc::LHCMaxBunches;
        o2::InteractionRecord ir(bc, orb);
        rof.setBCData(ir);
        rof.setROFrame(iROF);
        rof.setNEntries(0);
        rof.setFirstEntry(-1);
      }
      uint32_t prevEntry{0};
      for (const auto& rof : clusROFVec) {
        const auto& ir = rof.getBCData();
        const auto irToFirst = ir - firstIR;
        const int irROF = irToFirst.toLong() / par.getROFLengthInBC(iLayer);
        auto& expROF = expClusRofVec[irROF];
        expROF.setFirstEntry(rof.getFirstEntry());
        expROF.setNEntries(rof.getNEntries());
        if (expROF.getBCData() != rof.getBCData()) {
          LOGP(fatal, "detected mismatch between expected ROF:{} and received ROF:{}", expROF.asString(), rof.asString());
        }
      }
      int prevFirst{0};
      for (auto& rof : expClusRofVec) {
        if (rof.getFirstEntry() < 0) {
          rof.setFirstEntry(prevFirst);
        }
        prevFirst = rof.getFirstEntry();
      }
      nROFs = expClusRofVec.size();
      pc.outputs().snapshot(Output{Origin, "CLUSTERSROF", iLayer}, expClusRofVec);
    } else {
      pc.outputs().snapshot(Output{Origin, "CLUSTERSROF", iLayer}, clusROFVec);
    }
    pc.outputs().snapshot(Output{Origin, "COMPCLUSTERS", iLayer}, clusCompVec);
    pc.outputs().snapshot(Output{Origin, "PATTERNS", iLayer}, clusPattVec);

    nClusters += clusCompVec.size();

    if (mUseMC) {
      pc.outputs().snapshot(Output{Origin, "CLUSTERSMCTR", iLayer}, *clusterLabels); // at the moment requires snapshot
      std::vector<o2::itsmft::MC2ROFRecord> clusterMC2ROframes(mc2rofs[iLayer].size());
      for (int i = mc2rofs[iLayer].size(); i--;) {
        clusterMC2ROframes[i] = mc2rofs[iLayer][i]; // Simply, replicate it from digits ?
      }
      pc.outputs().snapshot(Output{Origin, "CLUSTERSMC2ROF", iLayer}, clusterMC2ROframes);
    }
    reader.reset();

    // TODO: in principle, after masking "overflow" pixels the MC2ROFRecord maxROF supposed to change, nominally to minROF
    // -> consider recalculationg maxROF
    sw.Stop();
    LOG(info) << mDetName << "Clusterer:" << layer << " pushed " << clusCompVec.size() << " clusters, in " << nROFs << " RO frames in " << sw.RealTime() << " s";
  }

  LOG(info) << mDetName << "Clusterer produced " << nClusters << " clusters";
}

///_______________________________________
template <int N>
void ClustererDPL<N>::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<N>*>("alppar");
    pc.inputs().get<o2::itsmft::ClustererParam<N>*>("cluspar");
    mClusterer->setContinuousReadOut(o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(N));
    // settings for the fired pixel overflow masking
    const auto& alpParams = o2::itsmft::DPLAlpideParam<N>::Instance();
    const auto& clParams = o2::itsmft::ClustererParam<N>::Instance();
    if (clParams.maxBCDiffToMaskBias > 0 && clParams.maxBCDiffToSquashBias > 0) {
      LOGP(fatal, "maxBCDiffToMaskBias = {} and maxBCDiffToSquashBias = {} cannot be set at the same time. Either set masking or squashing with a BCDiff > 0", clParams.maxBCDiffToMaskBias, clParams.maxBCDiffToSquashBias);
    }
    mClusterer->setDropHugeClusters(clParams.dropHugeClusters);
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
    mClusterer->setMaxROFDepthToSquash(nROFsToSquash);
    if constexpr (DPLAlpideParam<N>::supportsStaggering()) {
      if (mClusterer->isContinuousReadOut()) {
        for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
          mClusterer->addMaxBCSeparationToSquash(alpParams.getROFLengthInBC(iLayer) + clParams.getMaxBCDiffToSquashBias(iLayer));
          mClusterer->addMaxROFDepthToSquash((clParams.getMaxBCDiffToSquashBias(iLayer) > 0) ? 2 + int(clParams.maxSOTMUS / (alpParams.getROFLengthInBC(iLayer) * o2::constants::lhc::LHCBunchSpacingMUS)) : 0);
        }
      }
    }
    mClusterer->print(false);
  }
  // we may have other params which need to be queried regularly
}

///_______________________________________
template <int N>
void ClustererDPL<N>::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher(Origin, "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated" << (!mUseClusterDictionary ? " but its using is disabled" : "");
    if (mUseClusterDictionary) {
      mClusterer->setDictionary((const TopologyDictionary*)obj);
    }
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher(Origin, "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<N>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher(Origin, "CLUSPARAM", 0)) {
    LOG(info) << "Cluster param updated";
    const auto& par = o2::itsmft::ClustererParam<N>::Instance();
    par.printKeyValues();
    return;
  }
}

namespace
{
template <int N>
DataProcessorSpec getClustererSpec(bool useMC)
{
  constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  std::vector<InputSpec> inputs;
  constexpr uint32_t nLayers = (DPLAlpideParam<N>::supportsStaggering()) ? DPLAlpideParam<N>::getNLayers() : 1;
  for (uint32_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    inputs.emplace_back("digits", Origin, "DIGITS", iLayer, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", Origin, "DIGITSROF", iLayer, Lifetime::Timeframe);
    if (useMC) {
      inputs.emplace_back("labels", Origin, "DIGITSMCTR", iLayer, Lifetime::Timeframe);
      inputs.emplace_back("MC2ROframes", Origin, "DIGITSMC2ROF", iLayer, Lifetime::Timeframe);
    }
  }
  inputs.emplace_back("cldict", Origin, "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec(Origin.as<std::string>() + "/Calib/ClusterDictionary"));
  inputs.emplace_back("cluspar", Origin, "CLUSPARAM", 0, Lifetime::Condition, ccdbParamSpec(Origin.as<std::string>() + "/Config/ClustererParam"));
  inputs.emplace_back("alppar", Origin, "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec(Origin.as<std::string>() + "/Config/AlpideParam"));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              inputs,
                                                              true);
  std::vector<OutputSpec> outputs;
  for (uint32_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    outputs.emplace_back(Origin, "COMPCLUSTERS", iLayer, Lifetime::Timeframe);
    outputs.emplace_back(Origin, "PATTERNS", iLayer, Lifetime::Timeframe);
    outputs.emplace_back(Origin, "CLUSTERSROF", iLayer, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(Origin, "CLUSTERSMCTR", iLayer, Lifetime::Timeframe);
      outputs.emplace_back(Origin, "CLUSTERSMC2ROF", iLayer, Lifetime::Timeframe);
    }
  }
  return DataProcessorSpec{
    .name = (N == o2::detectors::DetID::ITS) ? "its-clusterer" : "mft-clusterer",
    .inputs = inputs,
    .outputs = outputs,
    .algorithm = AlgorithmSpec{adaptFromTask<ClustererDPL<N>>(ggRequest, useMC)},
    .options = Options{
      {"ignore-cluster-dictionary", VariantType::Bool, false, {"do not use cluster dictionary, always store explicit patterns"}},
      {"nthreads", VariantType::Int, 1, {"Number of clustering threads"}}}};
}
} // namespace

framework::DataProcessorSpec getITSClustererSpec(bool useMC)
{
  return getClustererSpec<o2::detectors::DetID::ITS>(useMC);
}

framework::DataProcessorSpec getMFTClustererSpec(bool useMC)
{
  return getClustererSpec<o2::detectors::DetID::MFT>(useMC);
}

} // namespace o2::itsmft
