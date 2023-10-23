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

/// \file   TPCCalibPadGainTracksSpec.h
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#ifndef O2_CALIBRATION_TPCCALIBPADGAINTRACKSSPEC_H
#define O2_CALIBRATION_TPCCALIBPADGAINTRACKSSPEC_H

#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "TPCCalibration/CalibPadGainTracks.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsParameters/GRPObject.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "Framework/CCDBParamSpec.h"
#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include <random>

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2
{
namespace tpc
{

class TPCCalibPadGainTracksDevice : public o2::framework::Task
{
 public:
  TPCCalibPadGainTracksDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, const uint32_t publishAfterTFs, const bool debug, const bool useLastExtractedMapAsReference, const std::string polynomialsFile, const bool disablePolynomialsCCDB) : mPublishAfter(publishAfterTFs), mDebug(debug), mUseLastExtractedMapAsReference(useLastExtractedMapAsReference), mDisablePolynomialsCCDB(disablePolynomialsCCDB), mCCDBRequest(req)
  {
    if (!polynomialsFile.empty()) {
      LOGP(info, "Loading polynomials from file {}", polynomialsFile);
      mPadGainTracks.loadPolTopologyCorrectionFromFile(polynomialsFile.data());
      mDisablePolynomialsCCDB = true;
    }
  }

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mTPCCorrMapsLoader.init(ic);
    // setting up the histogram ranges
    const auto nBins = ic.options().get<int>("nBins");
    auto reldEdxMin = ic.options().get<float>("reldEdxMin");
    auto reldEdxMax = ic.options().get<float>("reldEdxMax");
    const auto underflowBin = ic.options().get<bool>("underflowBin");
    const auto overflowBin = ic.options().get<bool>("overflowBin");
    const auto donotnormalize = ic.options().get<bool>("do-not-normalize");
    const auto disableLogTransform = ic.options().get<bool>("disable-log-transform");
    mPadGainTracks.setLogTransformQ(!disableLogTransform);
    float mindEdx = ic.options().get<float>("mindEdx");
    float maxdEdx = ic.options().get<float>("maxdEdx");
    if (!disableLogTransform) {
      if (reldEdxMin > -1) {
        reldEdxMin = std::log(1 + reldEdxMin);
      } else {
        LOGP(warn, "reldEdxMin (={}) is smaller than -1!", reldEdxMin);
      }
      if (reldEdxMax > 0) {
        reldEdxMax = std::log(1 + reldEdxMax);
      } else {
        LOGP(warn, "reldEdxMax (={}) is smaller than 0!", reldEdxMax);
      }
    }
    mPadGainTracks.init(nBins, reldEdxMin, reldEdxMax, underflowBin, overflowBin);
    mPadGainTracks.setdEdxMin(mindEdx);
    mPadGainTracks.setdEdxMax(maxdEdx);
    mPadGainTracks.doNotNomalize(donotnormalize);

    const auto propagateTrack = ic.options().get<bool>("do-not-propagateTrack");
    mPadGainTracks.setPropagateTrack(!propagateTrack);

    const auto dedxRegionType = ic.options().get<int>("dedxRegionType");
    mPadGainTracks.setdEdxRegion(static_cast<CalibPadGainTracks::DEdxRegion>(dedxRegionType));

    const auto dedxType = ic.options().get<int>("dedxType");
    mPadGainTracks.setMode(static_cast<CalibPadGainTracks::DEdxType>(dedxType));

    const auto chargeType = ic.options().get<int>("chargeType");
    assert(chargeType == 0 || chargeType == 1);
    mPadGainTracks.setChargeType(static_cast<ChargeType>(chargeType));

    mUseEveryNthTF = ic.options().get<int>("useEveryNthTF");
    if (mUseEveryNthTF <= 0) {
      mUseEveryNthTF = 1;
    }

    if (mPublishAfter > 1) {
      std::mt19937 rng(std::time(nullptr));
      std::uniform_int_distribution<std::mt19937::result_type> dist(1, mPublishAfter);
      mFirstTFSend = dist(rng);
      LOGP(info, "Publishing first data after {} processed TFs", mFirstTFSend);
    }

    mMaxTracksPerTF = ic.options().get<int>("maxTracksPerTF");

    const std::string gainMapFile = ic.options().get<std::string>("gainMapFile");
    if (!gainMapFile.empty()) {
      LOGP(info, "Loading GainMap from file {}", gainMapFile);
      mPadGainTracks.setRefGainMap(gainMapFile.data(), "GainMap");
    }

    const auto etaMax = ic.options().get<float>("etaMax");
    mPadGainTracks.setMaxEta(etaMax);

    const auto minClusters = ic.options().get<int>("minClusters");
    mPadGainTracks.setMinNClusters(minClusters);

    const auto momMin = ic.options().get<float>("momMin");
    const auto momMax = ic.options().get<float>("momMax");
    LOGP(info, "Using particle tracks with {} GeV/c < p < {} GeV/c ", momMin, momMax);
    mPadGainTracks.setMomentumRange(momMin, momMax);
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    LOGP(info, "finaliseCCDB");
    if (matcher == ConcreteDataMatcher(gDataOriginTPC, "RESIDUALGAINMAP", 0)) {
      if (!mUsingDefaultGainMapForFirstIter) {
        LOGP(info, "Updating reference gain map from previous iteration from CCDB");
        const auto* gainMapResidual = static_cast<std::unordered_map<string, o2::tpc::CalDet<float>>*>(obj);
        mPadGainTracks.setRefGainMap(gainMapResidual->at("GainMap"));
      } else {
        // just skip for the first time asking for an object -> not gain map will be used as reference
        LOGP(info, "Skipping loading reference gain map for first iteration from CCDB");
        mUsingDefaultGainMapForFirstIter = false;
      }
    } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "TOPOLOGYGAIN", 0)) {
      LOGP(info, "Updating Q topology correction from CCDB");
      const auto* topologyCorr = static_cast<o2::tpc::CalibdEdxTrackTopologyPolContainer*>(obj);
      mPadGainTracks.setPolTopologyCorrectionFromContainer(*topologyCorr);
    } else if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    } else if (mTPCCorrMapsLoader.accountCCDBInputs(matcher, obj)) {
    } else if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      const auto field = (5.00668f / 30000.f) * o2::base::GRPGeomHelper::instance().getGRPMagField()->getL3Current();
      LOGP(info, "Setting magnetic field to {} kG", field);
      mPadGainTracks.setField(field);
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto currentTF = processing_helpers::getCurrentTF(pc);
    if (mTFCounter++ % mUseEveryNthTF) {
      LOGP(info, "Skipping TF {}", currentTF);
      return;
    }
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);

    auto tracks = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
    auto clRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
    const auto& clusters = getWorkflowTPCInput(pc);
    const auto nTracks = tracks.size();
    if (nTracks == 0) {
      return;
    }
    LOGP(detail, "Processing TF {} with {} tracks by considering {} tracks", currentTF, nTracks, mMaxTracksPerTF);

    if (!mDisablePolynomialsCCDB) {
      pc.inputs().get<o2::tpc::CalibdEdxTrackTopologyPolContainer*>("tpctopologygain");
    }

    if (mUseLastExtractedMapAsReference) {
      LOGP(info, "fetching residual gain map");
      pc.inputs().get<std::unordered_map<std::string, o2::tpc::CalDet<float>>*>("tpcresidualgainmap");
    }
    mTPCVDriftHelper.extractCCDBInputs(pc);
    mTPCCorrMapsLoader.extractCCDBInputs(pc);
    bool updateMaps = false;
    if (mTPCCorrMapsLoader.isUpdated()) {
      mPadGainTracks.setTPCCorrMaps(&mTPCCorrMapsLoader);
      mTPCCorrMapsLoader.acknowledgeUpdate();
      updateMaps = true;
    }
    if (mTPCVDriftHelper.isUpdated()) {
      LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
           mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
           mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
           mTPCVDriftHelper.getSourceName());
      mPadGainTracks.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
      mTPCVDriftHelper.acknowledgeUpdate();
      updateMaps = true;
    }
    if (updateMaps) {
      mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
    }

    mPadGainTracks.setMembers(&tracks, &clRefs, clusters->clusterIndex);
    mPadGainTracks.processTracks(mMaxTracksPerTF);
    ++mProcessedTFs;
    if ((mFirstTFSend == mProcessedTFs) || (mPublishAfter && (mProcessedTFs % mPublishAfter) == 0)) {
      LOGP(info, "Publishing after {} TFs", mProcessedTFs);
      mProcessedTFs = 0;
      mFirstTFSend = 0; // set to zero in order to only trigger once
      if (mDebug) {
        mPadGainTracks.dumpToFile(fmt::format("calPadGain_TF{}.root", currentTF).data());
      }
      sendOutput(pc.outputs());
    }
  }

 private:
  const uint32_t mPublishAfter{0};                        ///< number of TFs after which to dump the calibration
  const bool mDebug{false};                               ///< create debug output
  const bool mUseLastExtractedMapAsReference{false};      ///< using the last extracted gain map as the reference map which will be applied
  bool mDisablePolynomialsCCDB{false};                    ///< do not load the polynomials from the CCDB
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest; ///< for accessing the b-field
  uint32_t mProcessedTFs{0};                              ///< counter to keep track of the processed TFs
  uint32_t mTFCounter{0};                                 ///< counter to keep track of the TFs
  CalibPadGainTracks mPadGainTracks{false};               ///< class for creating the pad-by-pad gain map
  bool mUsingDefaultGainMapForFirstIter{true};            ///< using no reference gain map for the first iteration
  unsigned int mUseEveryNthTF{1};                         ///< process every Nth TF only
  unsigned int mFirstTFSend{1};                           ///< first TF for which the data will be send (initialized randomly)
  int mMaxTracksPerTF{-1};                                ///< max number of tracks processed per TF
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};

  void sendOutput(DataAllocator& output)
  {
    output.snapshot(Output{"TPC", "TRACKGAINHISTOS", 0}, *mPadGainTracks.getHistos().get());
    mPadGainTracks.resetHistos();
  }
};

DataProcessorSpec getTPCCalibPadGainTracksSpec(const uint32_t publishAfterTFs, const bool debug, const bool useLastExtractedMapAsReference, const std::string polynomialsFile, bool disablePolynomialsCCDB, bool requestCTPLumi)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trackTPC", gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe);

  if (!polynomialsFile.empty()) {
    disablePolynomialsCCDB = true;
  }

  if (!disablePolynomialsCCDB) {
    inputs.emplace_back("tpctopologygain", gDataOriginTPC, "TOPOLOGYGAIN", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalTopologyGain)));
  }

  if (useLastExtractedMapAsReference) {
    inputs.emplace_back("tpcresidualgainmap", gDataOriginTPC, "RESIDUALGAINMAP", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalPadGainResidual)));
  }

  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
  Options opts{
    {"nBins", VariantType::Int, 20, {"Number of bins per histogram"}},
    {"reldEdxMin", VariantType::Int, 0, {"Minimum x coordinate of the histogram for Q/(dE/dx)"}},
    {"reldEdxMax", VariantType::Int, 3, {"Maximum x coordinate of the histogram for Q/(dE/dx)"}},
    {"underflowBin", VariantType::Bool, false, {"Using under flow bin"}},
    {"overflowBin", VariantType::Bool, true, {"Using overflow flow bin"}},
    {"momMin", VariantType::Float, 0.3f, {"minimum momentum of the tracks which are used for the pad-by-pad gain map"}},
    {"momMax", VariantType::Float, 1.f, {"maximum momentum of the tracks which are used for the pad-by-pad gain map"}},
    {"etaMax", VariantType::Float, 1.f, {"maximum eta of the tracks which are used for the pad-by-pad gain map"}},
    {"disable-log-transform", VariantType::Bool, false, {"Disable the transformation of q/dedx -> log(1 + q/dedx)"}},
    {"do-not-normalize", VariantType::Bool, false, {"Do not normalize the cluster charge to the dE/dx"}},
    {"mindEdx", VariantType::Float, 10.f, {"Minimum accepted dE/dx value"}},
    {"maxdEdx", VariantType::Float, 500.f, {"Maximum accepted dE/dx value (-1=accept all dE/dx)"}},
    {"minClusters", VariantType::Int, 50, {"minimum number of clusters of tracks which are used for the pad-by-pad gain map"}},
    {"gainMapFile", VariantType::String, "", {"file to reference gain map, which will be used for correcting the cluster charge"}},
    {"dedxRegionType", VariantType::Int, 2, {"using the dE/dx per chamber (0), stack (1) or per sector (2)"}},
    {"dedxType", VariantType::Int, 0, {"recalculating the dE/dx (0), using it from tracking (1)"}},
    {"chargeType", VariantType::Int, 0, {"Using qMax (0) or qTot (1) for the dE/dx and the pad-by-pad histograms"}},
    {"do-not-propagateTrack", VariantType::Bool, false, {"Performing a refit for obtaining track parameters instead of propagating."}},
    {"useEveryNthTF", VariantType::Int, 10, {"Using only a fraction of the data: 1: Use every TF, 10: Use only every tenth TF."}},
    {"maxTracksPerTF", VariantType::Int, 10000, {"Maximum number of processed tracks per TF (-1 for processing all tracks)"}},
  };
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(inputs, opts, requestCTPLumi);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                false,                          // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                true,                           // GRPMagField
                                                                true,                           // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(gDataOriginTPC, "TRACKGAINHISTOS", 0, o2::framework::Lifetime::Sporadic);

  return DataProcessorSpec{
    "calib-tpc-gainmap-tracks",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCCalibPadGainTracksDevice>(ccdbRequest, publishAfterTFs, debug, useLastExtractedMapAsReference, polynomialsFile, disablePolynomialsCCDB)},
    opts}; // end DataProcessorSpec
}

} // namespace tpc
} // namespace o2

#endif
