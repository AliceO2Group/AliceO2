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
#include <TStopwatch.h>
#include "DataFormatsTPC/Constants.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "CommonUtils/NameConf.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "GlobalTrackingStudy/TPCDataFilter.h"
#include "TPCBase/ParameterElectronics.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "Steer/MCKinematicsReader.h"

namespace o2::global
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class TPCDataFilter : public Task
{
 public:
  enum TrackDecision : char { NA,
                              REMOVE,
                              REDUCE,
                              KEEP };

  TPCDataFilter(std::shared_ptr<DataRequest> dr, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mTracksSrc(src), mUseMC(useMC)
  {
  }
  ~TPCDataFilter() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);
  void sendOutput(ProcessingContext& pc);

 private:
  void updateTimeDependentParams(ProcessingContext& pc){};
  std::shared_ptr<DataRequest> mDataRequest;
  bool mUseMC{false}; ///< MC flag
  char mRemMode = TrackDecision::KEEP;
  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
  //
  // TPC data
  std::vector<o2::tpc::ClusterNative> mClustersLinearFiltered;
  std::vector<unsigned int> mClustersLinearStatus;
  std::vector<char> mTrackStatus;
  std::vector<o2::tpc::TrackTPC> mTracksFiltered;
  std::vector<o2::tpc::TPCClRefElem> mTrackClusIdxFiltered;
  std::vector<o2::MCCompLabel> mTPCTrkLabelsFiltered;
  o2::tpc::ClusterNativeAccess mClusFiltered;
  GTrackID::mask_t mAccTrackSources;
  float mMinAbsTgl = 0.;
  int mMinTPCClusters = 0;
};

void TPCDataFilter::init(InitContext& ic)
{
  mRemMode = ic.options().get<bool>("suppress-rejected") ? TrackDecision::REMOVE : TrackDecision::REDUCE;
  GTrackID::mask_t acceptSourcesTrc = GTrackID::getSourcesMask("ITS,TPC,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TRD,TPC-TRD-TOF,ITS-TPC-TOF,ITS-TPC-TRD-TOF");
  mAccTrackSources = acceptSourcesTrc & GTrackID::getSourcesMask(ic.options().get<std::string>("accept-track-sources"));
  mMinAbsTgl = ic.options().get<float>("min-abs-tgl");
  mMinTPCClusters = ic.options().get<int>("min-tpc-clusters");
}

void TPCDataFilter::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);

  sendOutput(pc);
}

void TPCDataFilter::sendOutput(ProcessingContext& pc)
{

  pc.outputs().snapshot(Output{"TPC", "TRACKSF", 0, Lifetime::Timeframe}, mTracksFiltered);
  pc.outputs().snapshot(Output{"TPC", "CLUSREFSF", 0, Lifetime::Timeframe}, mTrackClusIdxFiltered);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"TPC", "TRACKSMCLBLF", 0, Lifetime::Timeframe}, mTPCTrkLabelsFiltered);
  }

  o2::tpc::TPCSectorHeader clusterOutputSectorHeader{0};
  clusterOutputSectorHeader.activeSectors = (1ul << o2::tpc::constants::MAXSECTOR) - 1;
  for (int i = 0; i < o2::tpc::constants::MAXSECTOR; i++) {
    clusterOutputSectorHeader.sectorBits = (1ul << i);
    o2::header::DataHeader::SubSpecificationType subspec = i;
    char* buffer = pc.outputs().make<char>({o2::header::gDataOriginTPC, "CLUSTERNATIVEF", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}},
                                           mClusFiltered.nClustersSector[i] * sizeof(*mClusFiltered.clustersLinear) + sizeof(o2::tpc::ClusterCountIndex))
                     .data();
    o2::tpc::ClusterCountIndex* outIndex = reinterpret_cast<o2::tpc::ClusterCountIndex*>(buffer);
    memset(outIndex, 0, sizeof(*outIndex));
    for (int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
      outIndex->nClusters[i][j] = mClusFiltered.nClusters[i][j];
    }
    memcpy(buffer + sizeof(*outIndex), mClusFiltered.clusters[i][0], mClusFiltered.nClustersSector[i] * sizeof(*mClusFiltered.clustersLinear));

    if (mUseMC && mClusFiltered.clustersMCTruth) {
      o2::dataformats::MCLabelContainer cont;
      for (unsigned int j = 0; j < mClusFiltered.nClustersSector[i]; j++) {
        const auto& labels = mClusFiltered.clustersMCTruth->getLabels(mClusFiltered.clusterOffset[i][0] + j);
        for (const auto& label : labels) {
          cont.addElement(j, label);
        }
      }
      o2::dataformats::ConstMCLabelContainer contflat;
      cont.flatten_to(contflat);
      pc.outputs().snapshot({o2::header::gDataOriginTPC, "CLNATIVEMCLBLF", subspec, Lifetime::Timeframe, {clusterOutputSectorHeader}}, contflat);
    }
  }
}

void TPCDataFilter::process(o2::globaltracking::RecoContainer& recoData)
{
  auto tpcTracks = recoData.getTPCTracks();
  auto tpcTrackClusIdx = recoData.getTPCTracksClusterRefs();
  auto tpcClusterIdxStruct = &recoData.inputsTPCclusters->clusterIndex;
  gsl::span<const o2::MCCompLabel> tpcTrkLabels;
  if (mUseMC) {
    tpcTrkLabels = recoData.getTPCTracksMCLabels();
    mTPCTrkLabelsFiltered.clear();
  }
  const unsigned int DISCARD = -1U;

  mTracksFiltered.clear();
  mTrackClusIdxFiltered.clear();
  mClustersLinearFiltered.clear();
  mTrackStatus.clear();
  mTrackStatus.resize(tpcTracks.size());
  mClustersLinearStatus.clear();
  mClustersLinearStatus.resize(tpcClusterIdxStruct->nClustersTotal, DISCARD);

  size_t nSelTracks = 0;
  auto selTPCTrack = [this, tpcTracks, tpcTrackClusIdx, tpcClusterIdxStruct, &nSelTracks](int itpc, char stat) {
    this->mTrackStatus[itpc] = stat;
    if (stat != TrackDecision::KEEP) {
      return;
    }
    nSelTracks++;
    const auto& trc = tpcTracks[itpc];
    int count = trc.getNClusters();
    const o2::tpc::ClusterNative* cl = nullptr;
    for (int ic = count; ic--;) {
      const auto cl = &trc.getCluster(tpcTrackClusIdx, ic, *tpcClusterIdxStruct);
      size_t offs = std::distance(tpcClusterIdxStruct->clustersLinear, cl);
      this->mClustersLinearStatus[offs] = 0;
    }
  };

  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  int nv = vtxRefs.size();
  for (int iv = 0; iv < nv; iv++) {
    const auto& vtref = vtxRefs[iv];
    for (int is = 0; is < GTrackID::NSources; is++) {
      auto dmask = GTrackID::getSourceDetectorsMask(is);
      if (!GTrackID::getSourceDetectorsMask(is)[GTrackID::TPC]) {
        continue;
      }
      char decisionSource = mAccTrackSources[is] ? KEEP : mRemMode;
      int idMin = vtxRefs[iv].getFirstEntryOfSource(is), idMax = idMin + vtxRefs[iv].getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto decision = decisionSource;
        auto vid = trackIndex[i];
        auto tpcID = recoData.getTPCContributorGID(vid);
        if (vid.isAmbiguous() && mTrackStatus[tpcID] != TrackDecision::NA) { // already processed
          continue;
        }
        if (decisionSource == KEEP) {
          const auto& tpcTr = tpcTracks[tpcID.getIndex()];
          if (std::abs(tpcTr.getTgl()) < mMinAbsTgl || tpcTr.getNClusters() < mMinTPCClusters) {
            decision = mRemMode;
          }
        }
        selTPCTrack(tpcID, decision);
      }
    }
  }

  // copy filtered clusters and register them in the filtered access struct
  int offset = 0;
  for (unsigned int i = 0; i < o2::tpc::constants::MAXSECTOR; i++) {
    for (unsigned int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
      mClusFiltered.nClusters[i][j] = 0;
      for (unsigned int ic = 0; ic < tpcClusterIdxStruct->nClusters[i][j]; ic++) {
        if (mClustersLinearStatus[offset] != DISCARD) {
          mClustersLinearStatus[offset] = mClustersLinearFiltered.size();
          mClustersLinearFiltered.push_back(tpcClusterIdxStruct->clustersLinear[offset]);
          mClusFiltered.nClusters[i][j]++;
        }
        offset++;
      }
    }
  }
  mClusFiltered.clustersLinear = mClustersLinearFiltered.data();
  mClusFiltered.setOffsetPtrs();

  LOGP(info, "Accepted {} tracks out of {} and {} clusters out of {}", nSelTracks, tpcTracks.size(), mClusFiltered.nClustersTotal, tpcClusterIdxStruct->nClustersTotal);

  // register new tracks with updated cluster references
  for (size_t itr = 0; itr < tpcTracks.size(); itr++) {
    if (mTrackStatus[itr] == REMOVE) { // track is discarded
      continue;
    } else if (mTrackStatus[itr] == REDUCE) { // fill track by 0s, discard clusters
      auto& t = mTracksFiltered.emplace_back();
      memset(&t, 0, sizeof(o2::tpc::TrackTPC));
      if (mUseMC) {
        mTPCTrkLabelsFiltered.emplace_back();
      }
    } else { // track is accepted
      const auto& tor = tpcTracks[itr];
      const auto& cref = tor.getClusterRef();
      mTracksFiltered.push_back(tor);
      mTracksFiltered.back().shiftFirstClusterRef(int(mTrackClusIdxFiltered.size()) - tor.getClusterRef().getFirstEntry());
      size_t idx0 = mTrackClusIdxFiltered.size();
      int nclTrack = cref.getEntries();
      mTrackClusIdxFiltered.resize(mTrackClusIdxFiltered.size() + nclTrack + (nclTrack + 1) / 2);

      // see explanations in the TrackTPC::getClusterReference
      uint32_t* clIndArr = reinterpret_cast<uint32_t*>(&mTrackClusIdxFiltered[idx0]);
      uint8_t* srIndArr = reinterpret_cast<uint8_t*>(clIndArr + nclTrack);
      for (int ic = 0; ic < nclTrack; ic++) {
        uint8_t sectorIndex, rowIndex;
        const auto cl = &tor.getCluster(tpcTrackClusIdx, ic, *tpcClusterIdxStruct, sectorIndex, rowIndex);
        unsigned int oldLinear = std::distance(tpcClusterIdxStruct->clustersLinear, cl);
        unsigned int newLinear = mClustersLinearStatus[oldLinear];
        if (newLinear == DISCARD) {
          LOGP(fatal, "discarded cluster {} is selected", oldLinear);
        }
        clIndArr[ic] = newLinear - mClusFiltered.clusterOffset[sectorIndex][rowIndex];
        srIndArr[ic] = sectorIndex;
        srIndArr[ic + nclTrack] = rowIndex;
      }
      if (mUseMC) {
        mTPCTrkLabelsFiltered.emplace_back(tpcTrkLabels[itr]);
      }
    }
  }
}

void TPCDataFilter::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getTPCDataFilter(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC)
{
  std::vector<OutputSpec> outputs;
  for (int i = 0; i < o2::tpc::constants::MAXSECTOR; i++) {
    outputs.emplace_back(o2::header::gDataOriginTPC, "CLUSTERNATIVEF", i, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTPC, "CLNATIVEMCLBLF", i, Lifetime::Timeframe);
    }
  }
  outputs.emplace_back("TPC", "TRACKSF", 0, Lifetime::Timeframe);
  outputs.emplace_back("TPC", "CLUSREFSF", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TPC", "TRACKSMCLBLF", 0, Lifetime::Timeframe);
  }

  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  dataRequest->requestPrimaryVertertices(useMC);

  return DataProcessorSpec{
    "tpc-data-filter",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDataFilter>(dataRequest, srcTracks, useMC)},
    Options{
      {"suppress-rejected", VariantType::Bool, false, {"Remove suppressed tracks instead of reducing them"}},
      {"min-abs-tgl", VariantType::Float, 0.0f, {"suppress tracks with abs tgL less that this threshold (e.g. noise tracks)"}},
      {"min-tpc-clusters", VariantType::Int, 0, {"suppress tracks less clusters that this threshold (e.g. noise tracks)"}},
      {"accept-track-sources", VariantType::String, std::string{GTrackID::ALL}, {"comma-separated list of track sources to accept"}}}};
}

} // namespace o2::global
