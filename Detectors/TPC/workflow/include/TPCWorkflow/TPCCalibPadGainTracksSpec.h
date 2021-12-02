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

#ifndef O2_CALIBRATION_TPCCALIBPADGAINTRACKSSPEC_H
#define O2_CALIBRATION_TPCCALIBPADGAINTRACKSSPEC_H

/// @file   TPCCalibPadGainTracksSpec.h
/// @brief  TPC gain map extraction using particle tracks

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "TPCCalibration/CalibPadGainTracks.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsParameters/GRPObject.h"

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
  TPCCalibPadGainTracksDevice(const uint32_t publishAfterTFs, const bool debug) : mPublishAfter(publishAfterTFs), mDebug(debug) {}

  void init(o2::framework::InitContext& ic)
  {
    // setting up the histogram ranges
    const auto nBins = ic.options().get<int>("nBins");
    const auto reldEdxMin = ic.options().get<float>("reldEdxMin");
    const auto reldEdxMax = ic.options().get<float>("reldEdxMax");
    const auto underflowBin = ic.options().get<bool>("underflowBin");
    const auto overflowBin = ic.options().get<bool>("overflowBin");
    mPadGainTracks.init(nBins, reldEdxMin, reldEdxMax, underflowBin, overflowBin);

    float field = ic.options().get<float>("field");
    if (field <= -10.f) {
      const auto inputGRP = o2::base::NameConf::getGRPFileName();
      const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
      if (grp != nullptr) {
        field = 5.00668f * grp->getL3Current() / 30000.;
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

    const auto lowTrunc = ic.options().get<float>("lowTrunc");
    const auto upTrunc = ic.options().get<float>("upTrunc");
    LOGP(info, "Lower truncation range {} and  Upper truncation range {}  ", lowTrunc, upTrunc);
    mPadGainTracks.setTruncationRange(lowTrunc, upTrunc);

    const auto minEntriesHist = ic.options().get<int>("minEntriesHist");
    mPadGainTracks.setMinEntriesHis(minEntriesHist);

    // setting up the CCDB
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    if (mProcessedTFs == 0) {
      mFirstTF = getCurrentTF(pc);
    }

    auto tracks = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
    auto clRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
    const auto& clusters = getWorkflowTPCInput(pc);
    LOGP(info, "Processing TF {} with {} tracks", getCurrentTF(pc), tracks.size());

    mPadGainTracks.setMembers(&tracks, &clRefs, clusters->clusterIndex);
    mPadGainTracks.processTracks();

    if ((mPublishAfter && (++mProcessedTFs % mPublishAfter) == 0)) {
      LOGP(info, "Publishing after {} TFs", mProcessedTFs);
      mProcessedTFs = 0;
      mPadGainTracks.fillgainMap();
      if (mDebug) {
        mPadGainTracks.dumpToFile(fmt::format("calPadGain_TF{}.root", getCurrentTF(pc)).data());
      }
      if (mWriteToDB) {
        mDBapi.storeAsTFileAny<o2::tpc::CalPad>(&mPadGainTracks.getPadGainMap(), "TPC/Calib/PadGainTracks", mMetadata, mFirstTF, getCurrentTF(pc) + 1);
      }
      mPadGainTracks.resetHistos();
    }
  }

 private:
  const uint32_t mPublishAfter{0};              ///< number of TFs after which to dump the calibration
  const bool mDebug{false};                     ///< create debug output
  uint32_t mProcessedTFs{0};                    ///< counter to keep track of the processed TFs
  CalibPadGainTracks mPadGainTracks{};          ///< class for creating the pad-by-pad gain map
  bool mWriteToDB{};                            ///< flag if writing to CCDB will be done
  o2::ccdb::CcdbApi mDBapi;                     ///< API for storing the gain map in the CCDB
  std::map<std::string, std::string> mMetadata; ///< meta data of the stored object in CCDB
  uint32_t mFirstTF{};                          ///< storing of first TF used when setting the validity of the objects when writing to CCDB

  uint32_t getCurrentTF(o2::framework::ProcessingContext& pc) const { return o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true))->tfCounter; }
};

DataProcessorSpec getTPCCalibPadGainTracksSpec(const uint32_t publishAfterTFs, const bool debug)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe);

  return DataProcessorSpec{
    "calib-tpc-pad-gain-tracks",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCCalibPadGainTracksDevice>(publishAfterTFs, debug)},
    Options{
      {"ccdb-uri", VariantType::String, o2::base::NameConf::getCCDBServer(), {"URI for the CCDB access."}},
      {"nBins", VariantType::Int, 20, {"Number of bins per histogram."}},
      {"reldEdxMin", VariantType::Int, 0, {"Minimum x coordinate of the histogram for Q/dEdx."}},
      {"reldEdxMax", VariantType::Int, 3, {"Maximum x coordinate of the histogram for Q/dEdx."}},
      {"underflowBin", VariantType::Bool, false, {"Using under flow bin."}},
      {"overflowBin", VariantType::Bool, false, {"Using under flow bin."}},
      {"field", VariantType::Float, -100.f, {"magnetic field"}},
      {"momMin", VariantType::Float, 0.1f, {"minimum momentum of the tracks which are used for the pad-by-pad gain map"}},
      {"momMax", VariantType::Float, 5.f, {"minimum momentum of the tracks which are used for the pad-by-pad gain map"}},
      {"etaMax", VariantType::Float, 1.f, {"maximum eta of the tracks which are used for the pad-by-pad gain map"}},
      {"minClusters", VariantType::Int, 50, {"minimum number of clusters of tracks which are used for the pad-by-pad gain map"}},
      {"lowTrunc", VariantType::Float, 0.05f, {"lower truncation range for calculating the rel gain"}},
      {"upTrunc", VariantType::Float, 0.6f, {"upper truncation range for calculating the rel gain"}},
      {"minEntriesHist", VariantType::Int, 20, {"minimum number of entries in each histogram (if the number of entries are smaller, the gain value will be 0)"}},
    }}; // end DataProcessorSpec
}

} // namespace tpc
} // namespace o2

#endif
