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
  TPCCalibPadGainTracksDevice(const uint32_t publishAfterTFs, const bool debug, const bool useLastExtractedMapAsReference, const std::string polynomialsFile, const bool disablePolynomialsCCDB) : mPublishAfter(publishAfterTFs), mDebug(debug), mUseLastExtractedMapAsReference(useLastExtractedMapAsReference), mDisablePolynomialsCCDB(disablePolynomialsCCDB)
  {
    if (!polynomialsFile.empty()) {
      LOGP(info, "Loading polynomials from file {}", polynomialsFile);
      mPadGainTracks.loadPolTopologyCorrectionFromFile(polynomialsFile.data());
      mDisablePolynomialsCCDB = true;
    }
  }

  void init(o2::framework::InitContext& ic) final
  {
    // setting up the histogram ranges
    const auto nBins = ic.options().get<int>("nBins");
    const auto reldEdxMin = ic.options().get<float>("reldEdxMin");
    const auto reldEdxMax = ic.options().get<float>("reldEdxMax");
    const auto underflowBin = ic.options().get<bool>("underflowBin");
    const auto overflowBin = ic.options().get<bool>("overflowBin");
    mPadGainTracks.init(nBins, reldEdxMin, reldEdxMax, underflowBin, overflowBin);

    const auto propagateTrack = ic.options().get<bool>("propagateTrack");
    mPadGainTracks.setPropagateTrack(propagateTrack);

    const auto dedxRegionType = ic.options().get<int>("dedxRegionType");
    mPadGainTracks.setdEdxRegion(static_cast<CalibPadGainTracks::DEdxRegion>(dedxRegionType));

    const auto dedxType = ic.options().get<int>("dedxType");
    mPadGainTracks.setMode(static_cast<CalibPadGainTracks::DEdxType>(dedxType));

    const auto chargeType = ic.options().get<int>("chargeType");
    assert(chargeType == 0 || chargeType == 1);
    mPadGainTracks.setChargeType(static_cast<ChargeType>(chargeType));

    const std::string gainMapFile = ic.options().get<std::string>("gainMapFile");
    if (!gainMapFile.empty()) {
      LOGP(info, "Loading GainMap from file {}", gainMapFile);
      mPadGainTracks.setRefGainMap(gainMapFile.data(), "GainMap");
    }

    float field = ic.options().get<float>("field");
    if (field <= -10.f) {
      const auto inputGRP = o2::base::NameConf::getGRPFileName();
      const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
      if (grp != nullptr) {
        field = 5.00668f * grp->getL3Current() / 30000.f;
        LOGP(info, "Using GRP file to set the magnetic field to {} kG", field);
      }
    }

    LOGP(info, "Setting magnetic field to {} kG", field);
    mPadGainTracks.setField(field);

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
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    ++mProcessedTFs;
    auto tracks = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
    auto clRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
    const auto& clusters = getWorkflowTPCInput(pc);
    const auto nTracks = tracks.size();
    if (nTracks == 0) {
      return;
    }
    LOGP(info, "Processing TF {} with {} tracks", processing_helpers::getCurrentTF(pc), nTracks);

    if (!mDisablePolynomialsCCDB) {
      pc.inputs().get<o2::tpc::CalibdEdxTrackTopologyPolContainer*>("tpctopologygain");
    }

    if (mUseLastExtractedMapAsReference) {
      LOGP(info, "fetching residual gain map");
      pc.inputs().get<std::unordered_map<std::string, o2::tpc::CalDet<float>>*>("tpcresidualgainmap");
    }

    mPadGainTracks.setMembers(&tracks, &clRefs, clusters->clusterIndex);
    mPadGainTracks.processTracks();

    if ((mPublishAfter && (mProcessedTFs % mPublishAfter) == 0)) {
      LOGP(info, "Publishing after {} TFs", mProcessedTFs);
      mProcessedTFs = 0;
      if (mDebug) {
        mPadGainTracks.dumpToFile(fmt::format("calPadGain_TF{}.root", processing_helpers::getCurrentTF(pc)).data());
      }
      sendOutput(pc.outputs());
    }
  }

 private:
  const uint32_t mPublishAfter{0};                   ///< number of TFs after which to dump the calibration
  const bool mDebug{false};                          ///< create debug output
  const bool mUseLastExtractedMapAsReference{false}; ///< using the last extracted gain map as the reference map which will be applied
  bool mDisablePolynomialsCCDB{false};               ///< do not load the polynomials from the CCDB
  uint32_t mProcessedTFs{0};                         ///< counter to keep track of the processed TFs
  CalibPadGainTracks mPadGainTracks{false};          ///< class for creating the pad-by-pad gain map
  bool mUsingDefaultGainMapForFirstIter{true};       ///< using no reference gain map for the first iteration

  void sendOutput(DataAllocator& output)
  {
    output.snapshot(Output{"TPC", "TRACKGAINHISTOS", 0}, *mPadGainTracks.getHistos().get());
    mPadGainTracks.resetHistos();
  }
};

DataProcessorSpec getTPCCalibPadGainTracksSpec(const uint32_t publishAfterTFs, const bool debug, const bool useLastExtractedMapAsReference, const std::string polynomialsFile, const bool disablePolynomialsCCDB)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trackTPC", gDataOriginTPC, "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", gDataOriginTPC, "CLUSREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe);

  if (polynomialsFile.empty() || disablePolynomialsCCDB) {
    inputs.emplace_back("tpctopologygain", gDataOriginTPC, "TOPOLOGYGAIN", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalTopologyGain)));
  }

  if (useLastExtractedMapAsReference) {
    inputs.emplace_back("tpcresidualgainmap", gDataOriginTPC, "RESIDUALGAINMAP", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalPadGainResidual)));
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(gDataOriginTPC, "TRACKGAINHISTOS", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "calib-tpc-gainmap-tracks",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCCalibPadGainTracksDevice>(publishAfterTFs, debug, useLastExtractedMapAsReference, polynomialsFile, disablePolynomialsCCDB)},
    Options{
      {"ccdb-uri", VariantType::String, o2::base::NameConf::getCCDBServer(), {"URI for the CCDB access"}},
      {"nBins", VariantType::Int, 20, {"Number of bins per histogram"}},
      {"reldEdxMin", VariantType::Int, 0, {"Minimum x coordinate of the histogram for Q/(dE/dx)"}},
      {"reldEdxMax", VariantType::Int, 3, {"Maximum x coordinate of the histogram for Q/(dE/dx)"}},
      {"underflowBin", VariantType::Bool, false, {"Using under flow bin"}},
      {"overflowBin", VariantType::Bool, true, {"Using under flow bin"}},
      {"field", VariantType::Float, -100.f, {"Magnetic field in kG, need for track propagations, this value will be overwritten if a grp file is present"}},
      {"momMin", VariantType::Float, 0.3f, {"minimum momentum of the tracks which are used for the pad-by-pad gain map"}},
      {"momMax", VariantType::Float, 1.f, {"maximum momentum of the tracks which are used for the pad-by-pad gain map"}},
      {"etaMax", VariantType::Float, 1.f, {"maximum eta of the tracks which are used for the pad-by-pad gain map"}},
      {"minClusters", VariantType::Int, 50, {"minimum number of clusters of tracks which are used for the pad-by-pad gain map"}},
      {"gainMapFile", VariantType::String, "", {"file to reference gain map, which will be used for correcting the cluster charge"}},
      {"dedxRegionType", VariantType::Int, 2, {"using the dE/dx per chamber (0), stack (1) or per sector (2)"}},
      {"dedxType", VariantType::Int, 0, {"recalculating the dE/dx (0), using it from tracking (1)"}},
      {"chargeType", VariantType::Int, 0, {"Using qMax (0) or qTot (1) for the dE/dx and the pad-by-pad histograms"}},
      {"propagateTrack", VariantType::Bool, false, {"Propagating the track instead of performing a refit for obtaining track parameters."}},
    }}; // end DataProcessorSpec
}

} // namespace tpc
} // namespace o2

#endif
