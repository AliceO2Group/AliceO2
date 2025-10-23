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

/// \file MIPTrackFilterSpec.h
/// \brief Workflow to filter MIP tracks and streams them to other devices.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#include "TPCWorkflow/MIPTrackFilterSpec.h"

#include <algorithm>
#include <vector>
#include <memory>
#include <random>

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/Logger.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/Task.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "Headers/DataHeader.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"

using namespace o2::framework;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;

namespace o2::tpc
{

class MIPTrackFilterDevice : public Task
{
 public:
  MIPTrackFilterDevice(std::shared_ptr<o2::base::GRPGeomRequest> gr, std::shared_ptr<DataRequest> dr, GID::mask_t trackSourcesMask)
    : mGRPGeomRequest(gr), mDataRequest(dr), mTrackSourcesMask(trackSourcesMask) {}

  void init(framework::InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& eos) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void sendOutput(DataAllocator& output);

  std::shared_ptr<o2::base::GRPGeomRequest> mGRPGeomRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  GID::mask_t mTrackSourcesMask;
  TrackCuts mCuts{};                      ///< Tracks cuts object
  std::vector<TrackTPC> mMIPTracks;       ///< Filtered MIP tracks
  o2::dataformats::MeanVertexObject mVtx; ///< Mean vertex object
  unsigned int mProcessEveryNthTF{1};     ///< process every Nth TF only
  int mMaxTracksPerTF{-1};                ///< max number of MIP tracks processed per TF
  uint32_t mTFCounter{0};                 ///< counter to keep track of the TFs
  int mProcessNFirstTFs{0};               ///< number of first TFs which are not sampled
  float mDCACut{-1};                      ///< DCA cut
  float mDCAZCut{-1};                     ///< DCA z cut
  bool mSendDummy{false};                 ///< send empty data in case TF is skipped

  bool acceptDCA(o2::track::TrackPar propTrack, o2::math_utils::Point3D<float> refPoint, bool useDCAz = false);
};

void MIPTrackFilterDevice::init(framework::InitContext& ic)
{
  const double minP = ic.options().get<double>("min-momentum");
  const double maxP = ic.options().get<double>("max-momentum");
  const double mindEdx = ic.options().get<double>("min-dedx");
  const double maxdEdx = ic.options().get<double>("max-dedx");
  const int minClusters = std::max(10, ic.options().get<int>("min-clusters"));
  const auto cutLoopers = !ic.options().get<bool>("dont-cut-loopers");
  mSendDummy = ic.options().get<bool>("send-dummy-data");
  mMaxTracksPerTF = ic.options().get<int>("maxTracksPerTF");
  if (mMaxTracksPerTF > 0) {
    mMIPTracks.reserve(mMaxTracksPerTF);
  }

  mProcessEveryNthTF = ic.options().get<int>("processEveryNthTF");
  if (mProcessEveryNthTF <= 0) {
    mProcessEveryNthTF = 1;
  }
  mProcessNFirstTFs = ic.options().get<int>("process-first-n-TFs");

  if (mProcessEveryNthTF > 1) {
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, mProcessEveryNthTF);
    mTFCounter = dist(rng);
    LOGP(info, "Skipping first {} TFs", mProcessEveryNthTF - mTFCounter);
  }

  mCuts.setPMin(minP);
  mCuts.setPMax(maxP);
  mCuts.setNClusMin(minClusters);
  mCuts.setdEdxMin(mindEdx);
  mCuts.setdEdxMax(maxdEdx);
  mCuts.setCutLooper(cutLoopers);

  mDCACut = ic.options().get<float>("dca-cut");
  mDCAZCut = ic.options().get<float>("dca-z-cut");

  o2::base::GRPGeomHelper::instance().setRequest(mGRPGeomRequest);
}

void MIPTrackFilterDevice::run(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");

  const auto currentTF = processing_helpers::getCurrentTF(pc);
  if ((mTFCounter++ % mProcessEveryNthTF) && (currentTF >= mProcessNFirstTFs)) {
    LOGP(info, "Skipping TF {}", currentTF);
    mMIPTracks.clear();
    if (mSendDummy) {
      sendOutput(pc.outputs());
    }
    return;
  }

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  const auto tracksTPC = recoData.getTPCTracks();
  const auto nTracks = tracksTPC.size();

  // indices to good tracks
  std::vector<size_t> indices;
  indices.reserve(nTracks);

  const auto useGlobalTracks = mTrackSourcesMask[GID::ITSTPC];
  o2::math_utils::Point3D<float> vertex = mVtx.getXYZ();

  if (useGlobalTracks) {
    auto trackIndex = recoData.getPrimaryVertexMatchedTracks();                      // Global ID's for associated tracks
    auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs();                      // references from vertex to these track IDs
    std::vector<GID::Source> selSrc{GID::ITSTPC, GID::ITSTPCTRD, GID::ITSTPCTRDTOF}; // for Instance
    // LOGP(info, "Number of vertex tracks: {}", vtxRefs.size());
    const auto nv = (vtxRefs.size() > 0) ? vtxRefs.size() - 1 : 0; // note: the last entry groups the tracks which were not related to any vertex, to skip them, use vtxRefs.size()-1

    for (int iv = 0; iv < nv; iv++) {
      const auto& vtref = vtxRefs[iv];
      // LOGP(info, "Processing vertex {} with {} tracks", iv, vtref.getEntries());
      vertex = recoData.getPrimaryVertex(iv).getXYZ();
      // LOGP(info, "Vertex position: x={} y={} z={}", vertex.x(), vertex.y(), vertex.z());

      for (auto src : selSrc) {
        int idMin = vtxRefs[iv].getFirstEntryOfSource(src), idMax = idMin + vtxRefs[iv].getEntriesOfSource(src);
        // LOGP(info, "Source {}: idMin={} idMax={}", GID::getSourceName(src), idMin, idMax);

        for (int i = idMin; i < idMax; i++) {
          auto vid = trackIndex[i];
          const auto& track = recoData.getTrackParam(vid); // this is a COPY of the track param which we will modify during DCA calculation
          auto gidTPC = recoData.getTPCContributorGID(vid);
          if (gidTPC.isSourceSet()) {
            const auto idxTPC = gidTPC.getIndex();
            if (mCuts.goodTrack(tracksTPC[idxTPC]) && acceptDCA(tracksTPC[idxTPC], vertex, true)) {
              indices.emplace_back(idxTPC);
            }
          }
        }
      }
    }

  } else {
    for (size_t i = 0; i < nTracks; ++i) {
      if (mCuts.goodTrack(tracksTPC[i]) && acceptDCA(tracksTPC[i], vertex)) {
        indices.emplace_back(i);
      }
    }
  }

  size_t nTracksSel = indices.size();

  if ((mMaxTracksPerTF != -1) && (nTracksSel > mMaxTracksPerTF)) {
    // in case no good tracks have been found
    if (indices.empty()) {
      mMIPTracks.clear();
      if (mSendDummy) {
        sendOutput(pc.outputs());
      }
      return;
    }

    // shuffle indices to good tracks
    std::minstd_rand rng(std::time(nullptr));
    std::shuffle(indices.begin(), indices.end(), rng);

    // copy good tracks
    nTracksSel = (mMaxTracksPerTF > indices.size()) ? indices.size() : mMaxTracksPerTF;
  }

  for (int i = 0; i < nTracksSel; ++i) {
    mMIPTracks.emplace_back(tracksTPC[indices[i]]);
  }

  LOGP(info, "Filtered {} / {} MIP tracks out of {} total tpc tracks, using {}", mMIPTracks.size(), indices.size(), tracksTPC.size(), useGlobalTracks ? "global tracks" : "TPC only tracks");
  sendOutput(pc.outputs());
  mMIPTracks.clear();
}

void MIPTrackFilterDevice::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("GLO", "MEANVERTEX", 0)) {
    LOG(info) << "Setting new MeanVertex: " << ((const o2::dataformats::MeanVertexObject*)obj)->asString();
    mVtx = *(const o2::dataformats::MeanVertexObject*)obj;
    return;
  }
}

void MIPTrackFilterDevice::sendOutput(DataAllocator& output) { output.snapshot(Output{header::gDataOriginTPC, "MIPS", 0}, mMIPTracks); }

void MIPTrackFilterDevice::endOfStream(EndOfStreamContext& eos)
{
  LOG(info) << "Finalizig MIP Tracks filter";
}

bool MIPTrackFilterDevice::acceptDCA(o2::track::TrackPar propTrack, o2::math_utils::Point3D<float> refPoint, bool useDCAz)
{
  if (mDCACut < 0) {
    return true;
  }

  auto propagator = o2::base::Propagator::Instance();
  std::array<float, 2> dca;
  const auto ok = propagator->propagateToDCABxByBz(refPoint, propTrack, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT, &dca);
  const auto dcar = std::abs(dca[0]);

  return ok && (dcar < mDCACut) && (!useDCAz || (std::abs(dca[1]) < mDCAZCut));
}

DataProcessorSpec getMIPTrackFilterSpec(GID::mask_t srcTracks)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(header::gDataOriginTPC, "MIPS", 0, Lifetime::Sporadic);

  const auto useMC = false;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestPrimaryVertices(useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  dataRequest->inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, o2::framework::ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));

  return DataProcessorSpec{
    "tpc-miptrack-filter",
    dataRequest->inputs,
    outputs,
    adaptFromTask<MIPTrackFilterDevice>(ggRequest, dataRequest, srcTracks),
    Options{
      {"min-momentum", VariantType::Double, 0.35, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Double, 0.55, {"maximum momentum cut"}},
      {"min-dedx", VariantType::Double, 10., {"minimum dEdx cut"}},
      {"max-dedx", VariantType::Double, 200., {"maximum dEdx cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}},
      {"processEveryNthTF", VariantType::Int, 1, {"Using only a fraction of the data: 1: Use every TF, 10: Process only every tenth TF."}},
      {"maxTracksPerTF", VariantType::Int, -1, {"Maximum number of processed tracks per TF (-1 for processing all tracks)"}},
      {"process-first-n-TFs", VariantType::Int, 1, {"Number of first TFs which are not sampled"}},
      {"send-dummy-data", VariantType::Bool, false, {"Send empty data in case TF is skipped"}},
      {"dont-cut-loopers", VariantType::Bool, false, {"Do not cut loopers by comparing zout-zin"}},
      {"dca-cut", VariantType::Float, 3.f, {"DCA cut in xy (cm), < 0 to disable cut in xy and z"}},
      {"dca-z-cut", VariantType::Float, 5.f, {"DCA cut in z (cm)"}},
    }};
}

} // namespace o2::tpc
