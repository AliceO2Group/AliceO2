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

#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"

#include "ITSReconstruction/FastMultEstConfig.h"
#include "ITSReconstruction/FastMultEst.h"

#include "ITStracking/TrackingInterface.h"

#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/PhysTrigger.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "CommonDataFormat/IRFrame.h"

namespace o2
{
using namespace framework;
namespace its
{

std::string asString(TrackingMode mode)
{
  switch (mode) {
    case TrackingMode::Sync:
      return "sync";
    case TrackingMode::Async:
      return "async";
    case TrackingMode::Cosmics:
      return "cosmics";
  }
  return "unknown";
}

std::ostream& operator<<(std::ostream& os, TrackingMode v)
{
  os << asString(v);
  return os;
}

void ITSTrackingInterface::initialise()
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  std::vector<TrackingParameters> trackParams;

  if (mMode == TrackingMode::Async) {

    trackParams.resize(3);
    for (auto& param : trackParams) {
      param.ZBins = 64;
      param.PhiBins = 32;
      param.CellsPerClusterLimit = 1.e3f;
      param.TrackletsPerClusterLimit = 1.e3f;
    }
    trackParams[1].TrackletMinPt = 0.2f;
    trackParams[1].CellDeltaTanLambdaSigma *= 2.;
    trackParams[2].TrackletMinPt = 0.1f;
    trackParams[2].CellDeltaTanLambdaSigma *= 4.;
    trackParams[2].MinTrackLength = 4;
    LOG(info) << "Initializing tracker in async. phase reconstruction with " << trackParams.size() << " passes";

  } else if (mMode == TrackingMode::Sync) {
    trackParams.resize(1);
    trackParams[0].ZBins = 64;
    trackParams[0].PhiBins = 32;
    trackParams[0].MinTrackLength = 4;
    LOG(info) << "Initializing tracker in sync. phase reconstruction with " << trackParams.size() << " passes";
  } else if (mMode == TrackingMode::Cosmics) {
    mCosmicsProcessing = true;
    mRunVertexer = false;
    trackParams.resize(1);
    trackParams[0].MinTrackLength = 4;
    trackParams[0].CellDeltaTanLambdaSigma *= 10;
    trackParams[0].PhiBins = 4;
    trackParams[0].ZBins = 16;
    trackParams[0].PVres = 1.e5f;
    trackParams[0].MaxChi2ClusterAttachment = 60.;
    trackParams[0].MaxChi2NDF = 40.;
    trackParams[0].TrackletsPerClusterLimit = 100.;
    trackParams[0].CellsPerClusterLimit = 100.;
    LOG(info) << "Initializing tracker in reconstruction for cosmics with " << trackParams.size() << " passes";

  } else {
    throw std::runtime_error(fmt::format("Unsupported ITS tracking mode {:s} ", asString(mMode)));
  }

  for (auto& params : trackParams) {
    params.CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT;
  }
  mTracker->setParameters(trackParams);
}

void ITSTrackingInterface::run(framework::ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  gsl::span<const o2::itsmft::PhysTrigger> physTriggers;
  std::vector<o2::itsmft::PhysTrigger> fromTRD;
  if (mUseTriggers == 2) { // use TRD triggers
    o2::InteractionRecord ir{0, pc.services().get<o2::framework::TimingInfo>().firstTForbit};
    auto trdTriggers = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("phystrig");
    for (const auto& trig : trdTriggers) {
      if (trig.getBCData() >= ir && trig.getNumberOfTracklets()) {
        ir = trig.getBCData();
        fromTRD.emplace_back(o2::itsmft::PhysTrigger{ir, 0});
      }
    }
    physTriggers = gsl::span<const o2::itsmft::PhysTrigger>(fromTRD.data(), fromTRD.size());
  } else if (mUseTriggers == 1) { // use Phys triggers from ITS stream
    physTriggers = pc.inputs().get<gsl::span<o2::itsmft::PhysTrigger>>("phystrig");
  }

  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0}, rofsinput.begin(), rofsinput.end());
  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0});
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance(); // RS: this should come from CCDB

  irFrames.reserve(rofs.size());
  int nBCPerTF = alpParams.roFrameLengthInBC;

  LOGP(info, "ITSTracker pulled {} clusters, {} RO frames", compClusters.size(), rofs.size());

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mIsMC) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("itsmclabels").release();
    // get the array as read-only span, a snapshot is sent forward
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0}, pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("ITSMC2ROframes"));
    LOG(info) << labels->getIndexedSize() << " MC label objects , in " << mc2rofs.size() << " MC events";
  }

  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0});
  auto& allTracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0});
  auto& vertROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "VERTICESROF", 0});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0});

  // MC
  static pmr::vector<o2::MCCompLabel> dummyMCLabTracks, dummyMCLabVerts;
  auto& allTrackLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "TRACKSMCTR", 0}) : dummyMCLabTracks;
  auto& allVerticesLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "VERTICESMCTR", 0}) : dummyMCLabVerts;

  std::uint32_t roFrame = 0;

  bool continuous = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(info) << "ITSTracker RO: continuous=" << continuous;

  if (mOverrideBeamEstimation) {
    mTimeFrame->setBeamPosition(mMeanVertex->getX(),
                                mMeanVertex->getY(),
                                mMeanVertex->getSigmaY2(),
                                mTracker->getParameters()[0].LayerResolution[0],
                                mTracker->getParameters()[0].SystErrorY2[0]);
  }

  mTracker->setBz(o2::base::Propagator::Instance()->getNominalBz());

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();

  gsl::span<itsmft::ROFRecord> rofspan(rofs);
  mTimeFrame->loadROFrameData(rofspan, compClusters, pattIt, mDict, labels);
  pattIt = patterns.begin();
  std::vector<int> savedROF;
  auto logger = [&](std::string s) { LOG(info) << s; };
  auto fatalLogger = [&](std::string s) { LOG(fatal) << s; };
  auto errorLogger = [&](std::string s) { LOG(error) << s; };

  FastMultEst multEst; // mult estimator
  std::vector<bool> processingMask;
  int cutVertexMult{0}, cutRandomMult = int(rofs.size()) - multEst.selectROFs(rofs, compClusters, physTriggers, processingMask);
  mTimeFrame->setMultiplicityCutMask(processingMask);
  float vertexerElapsedTime{0.f};
  if (mRunVertexer) {
    vertROFvec.reserve(rofs.size());
    // Run seeding vertexer
    vertexerElapsedTime = mVertexer->clustersToVertices(logger);
  } else { // cosmics
    mTimeFrame->resetRofPV();
  }
  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  for (auto iRof{0}; iRof < rofspan.size(); ++iRof) {
    std::vector<Vertex> vtxVecLoc;
    auto& vtxROF = vertROFvec.emplace_back(rofspan[iRof]);
    vtxROF.setFirstEntry(vertices.size());
    if (mRunVertexer) {
      auto vtxSpan = mTimeFrame->getPrimaryVertices(iRof);
      vtxROF.setNEntries(vtxSpan.size());
      bool selROF = vtxSpan.size() == 0;
      for (auto iV{0}; iV < vtxSpan.size(); ++iV) {
        auto& v = vtxSpan[iV];
        if (multEstConf.isVtxMultCutRequested() && !multEstConf.isPassingVtxMultCut(v.getNContributors())) {
          continue; // skip vertex of unwanted multiplicity
        }
        selROF = true;
        vertices.push_back(v);
        if (mIsMC) {
          auto vLabels = mTimeFrame->getPrimaryVerticesLabels(iRof)[iV];
          allVerticesLabels.reserve(allVerticesLabels.size() + vLabels.size());
          std::copy(vLabels.begin(), vLabels.end(), std::back_inserter(allVerticesLabels));
        }
      }
      if (processingMask[iRof] && !selROF) { // passed selection in clusters and not in vertex multiplicity
        LOG(debug) << fmt::format("ROF {} rejected by the vertex multiplicity selection [{},{}]",
                                  iRof,
                                  multEstConf.cutMultVtxLow,
                                  multEstConf.cutMultVtxHigh);
        processingMask[iRof] = selROF;
        cutVertexMult++;
      }
    } else { // cosmics
      vtxVecLoc.emplace_back(Vertex());
      vtxVecLoc.back().setNContributors(1);
      vtxROF.setNEntries(vtxVecLoc.size());
      for (auto& v : vtxVecLoc) {
        vertices.push_back(v);
      }
      mTimeFrame->addPrimaryVertices(vtxVecLoc);
    }
  }
  LOG(info) << fmt::format(" - rejected {}/{} ROFs: random/mult.sel:{} (seed {}), vtx.sel:{}", cutRandomMult + cutVertexMult, rofspan.size(), cutRandomMult, multEst.lastRandomSeed, cutVertexMult);
  LOG(info) << fmt::format(" - Vertex seeding total elapsed time: {} ms for {} vertices found in {} ROFs", vertexerElapsedTime, mTimeFrame->getPrimaryVerticesNum(), rofspan.size());

  if (mOverrideBeamEstimation) {
    LOG(info) << fmt::format(" - Beam position set to: {}, {} from meanvertex object", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  } else {
    LOG(info) << fmt::format(" - Beam position computed for the TF: {}, {}", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  }
  if (mCosmicsProcessing && compClusters.size() > 1500 * rofspan.size()) {
    LOG(error) << "Cosmics processing was requested with an average detector occupancy exceeding 1.e-7, skipping TF processing.";
  } else {

    mTimeFrame->setMultiplicityCutMask(processingMask);
    // Run CA tracker
    if (mMode == o2::its::TrackingMode::Async) {
      mTracker->clustersToTracks(logger, fatalLogger);
    } else {
      mTracker->clustersToTracks(logger, errorLogger);
    }
    size_t totTracks{mTimeFrame->getNumberOfTracks()}, totClusIDs{mTimeFrame->getNumberOfUsedClusters()};
    allTracks.reserve(totTracks);
    allClusIdx.reserve(totClusIDs);

    if (mTimeFrame->hasBogusClusters()) {
      LOG(warning) << fmt::format(" - The processed timeframe had {} clusters with wild z coordinates, check the dictionaries", mTimeFrame->hasBogusClusters());
    }

    for (unsigned int iROF{0}; iROF < rofs.size(); ++iROF) {
      auto& rof{rofs[iROF]};
      auto& tracks = mTimeFrame->getTracks(iROF);
      auto number{tracks.size()};
      auto first{allTracks.size()};
      int offset = -rof.getFirstEntry(); // cluster entry!!!
      rof.setFirstEntry(first);
      rof.setNEntries(number);

      if (processingMask[iROF]) {
        irFrames.emplace_back(rof.getBCData(), rof.getBCData() + nBCPerTF - 1).info = tracks.size();
      }
      allTrackLabels.reserve(mTimeFrame->getTracksLabel(iROF).size()); // should be 0 if not MC
      std::copy(mTimeFrame->getTracksLabel(iROF).begin(), mTimeFrame->getTracksLabel(iROF).end(), std::back_inserter(allTrackLabels));
      // Some conversions that needs to be moved in the tracker internals
      for (unsigned int iTrk{0}; iTrk < tracks.size(); ++iTrk) {
        auto& trc{tracks[iTrk]};
        trc.setFirstClusterEntry(allClusIdx.size()); // before adding tracks, create final cluster indices
        int ncl = trc.getNumberOfClusters(), nclf = 0;
        for (int ic = TrackITSExt::MaxClusters; ic--;) { // track internally keeps in->out cluster indices, but we want to store the references as out->in!!!
          auto clid = trc.getClusterIndex(ic);
          if (clid >= 0) {
            trc.setClusterSize(ic, mTimeFrame->getClusterSize(clid));
            allClusIdx.push_back(clid);
            nclf++;
          }
        }
        assert(ncl == nclf);
        allTracks.emplace_back(trc);
      }
    }
    LOGP(info, "ITSTracker pushed {} tracks and {} vertices", allTracks.size(), vertices.size());
    if (mIsMC) {
      LOGP(info, "ITSTracker pushed {} track labels", allTrackLabels.size());
      LOGP(info, "ITSTracker pushed {} vertex labels", allVerticesLabels.size());
    }
  }
}

void ITSTrackingInterface::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alppar");
    if (pc.inputs().getPos("itsTGeo") >= 0) {
      pc.inputs().get<o2::its::GeometryTGeo*>("itsTGeo");
    }
    GeometryTGeo* geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
    mVertexer->getGlobalConfiguration();
    mTracker->getGlobalConfiguration();
    if (mOverrideBeamEstimation) {
      pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");
    }
  }
}

void ITSTrackingInterface::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher("GLO", "MEANVERTEX", 0)) {
    LOGP(info, "mean vertex acquired");
    setMeanVertex((const o2::dataformats::MeanVertexObject*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "GEOMTGEO", 0)) {
    LOG(info) << "ITS GeomtetryTGeo loaded from ccdb";
    o2::its::GeometryTGeo::adopt((o2::its::GeometryTGeo*)obj);
    return;
  }
}

void ITSTrackingInterface::setTraitsFromProvider(VertexerTraits* vertexerTraits,
                                                 TrackerTraits* trackerTraits,
                                                 TimeFrame* frame)
{
  mVertexer = std::make_unique<Vertexer>(vertexerTraits);
  mTracker = std::make_unique<Tracker>(trackerTraits);
  mTimeFrame = frame;
  mVertexer->adoptTimeFrame(*mTimeFrame);
  mTracker->adoptTimeFrame(*mTimeFrame);
}
} // namespace its
} // namespace o2