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

#include <memory>

#include <oneapi/tbb/task_arena.h>

#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"

#include "ITSReconstruction/FastMultEstConfig.h"
#include "ITSReconstruction/FastMultEst.h"

#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/TrackingInterface.h"

#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/PhysTrigger.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "CommonDataFormat/IRFrame.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITStracking/BoundedAllocator.h"
#include "Framework/DeviceSpec.h"

using namespace o2::framework;
using namespace o2::its;

void ITSTrackingInterface::initialise()
{
  // get parameters
  const auto& trackConf = o2::its::TrackerParamConfig::Instance();
  const auto& vertConf = o2::its::VertexerParamConfig::Instance();
  if (auto parmode = (TrackingMode::Type)trackConf.trackingMode; mMode == TrackingMode::Unset || (parmode != TrackingMode::Unset && mMode != parmode)) {
    LOGP(info, "Tracking mode overwritten by configurable params from {} to {}", TrackingMode::toString(mMode), TrackingMode::toString(parmode));
    mMode = parmode;
  }
  auto trackParams = TrackingMode::getTrackingParameters(mMode);
  auto vertParams = TrackingMode::getVertexingParameters(mMode);
  LOGP(info, "Initializing tracker in {} phase reconstruction with {} passes for tracking and {}/{} for vertexing", TrackingMode::toString(mMode), trackParams.size(), o2::its::VertexerParamConfig::Instance().nIterations, vertParams.size());
  mTracker->setParameters(trackParams);
  mVertexer->setParameters(vertParams);

  if (mMode == TrackingMode::Cosmics) {
    mRunVertexer = false;
    mCosmicsProcessing = true;
    LOGP(info, "Cosmic mode enabled, will skip vertexing");
  }

  // threading
  if (trackConf.nThreads == vertConf.nThreads) {
    bool clamped{false};
    int nThreads = trackConf.nThreads;
    if (nThreads > 0) {
      const int hw = std::thread::hardware_concurrency();
      const int maxThreads = (hw == 0 ? 1 : hw);
      nThreads = std::clamp(nThreads, 1, maxThreads);
      clamped = trackConf.nThreads > maxThreads;
    }
    LOGP(info, "Tracker and Vertexer will share the task arena with {} thread(s){}", nThreads, (clamped) ? " (clamped)" : "");
    mTaskArena = std::make_shared<tbb::task_arena>(std::abs(nThreads));
  }
  mVertexer->setNThreads(vertConf.nThreads, mTaskArena);
  mTracker->setNThreads(trackConf.nThreads, mTaskArena);
}

void ITSTrackingInterface::run(framework::ProcessingContext& pc)
{
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
  auto& trackROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0}, rofsinput.begin(), rofsinput.end());
  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0});
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance(); // RS: this should come from CCDB

  irFrames.reserve(trackROFvec.size());
  int nBCPerTF = alpParams.roFrameLengthInBC;

  LOGP(info, "ITSTracker pulled {} clusters, {} RO frames", compClusters.size(), trackROFvec.size());
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
  static pmr::vector<float> dummyMCPurVerts;
  auto& allTrackLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "TRACKSMCTR", 0}) : dummyMCLabTracks;
  auto& allVerticesLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "VERTICESMCTR", 0}) : dummyMCLabVerts;
  bool writeContLabels = mIsMC && o2::its::VertexerParamConfig::Instance().outputContLabels;
  auto& allVerticesContLabels = writeContLabels ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "VERTICESMCTRCONT", 0}) : dummyMCLabVerts;
  auto& allVerticesPurities = mIsMC ? pc.outputs().make<std::vector<float>>(Output{"ITS", "VERTICESMCPUR", 0}) : dummyMCPurVerts;

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

  gsl::span<itsmft::ROFRecord> trackROFspan(trackROFvec);
  loadROF(trackROFspan, compClusters, pattIt, labels);
  pattIt = patterns.begin();
  std::vector<int> savedROF;
  auto logger = [&](const std::string& s) { LOG(info) << s; };
  auto fatalLogger = [&](const std::string& s) { LOG(fatal) << s; };
  auto errorLogger = [&](const std::string& s) { LOG(error) << s; };

  FastMultEst multEst; // mult estimator
  std::vector<uint8_t> processingMask, processUPCMask;
  int cutVertexMult{0}, cutUPCVertex{0}, cutRandomMult = int(trackROFvec.size()) - multEst.selectROFs(trackROFvec, compClusters, physTriggers, processingMask);
  processUPCMask.resize(processingMask.size(), false);
  mTimeFrame->setMultiplicityCutMask(processingMask);
  float vertexerElapsedTime{0.f};
  if (mRunVertexer) {
    vertROFvec.reserve(trackROFvec.size());
    // Run seeding vertexer
    vertexerElapsedTime = mVertexer->clustersToVertices(logger);
  } else { // cosmics
    mTimeFrame->resetRofPV();
  }
  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  gsl::span<const std::pair<MCCompLabel, float>> vMCRecInfo;
  gsl::span<const MCCompLabel> vMCContLabels;
  for (auto iRof{0}; iRof < trackROFspan.size(); ++iRof) {
    bounded_vector<Vertex> vtxVecLoc;
    auto& vtxROF = vertROFvec.emplace_back(trackROFspan[iRof]);
    vtxROF.setFirstEntry(vertices.size());
    if (mRunVertexer) {
      auto vtxSpan = mTimeFrame->getPrimaryVertices(iRof);
      if (mIsMC) {
        vMCRecInfo = mTimeFrame->getPrimaryVerticesMCRecInfo(iRof);
        if (o2::its::VertexerParamConfig::Instance().outputContLabels) {
          vMCContLabels = mTimeFrame->getPrimaryVerticesContributors(iRof);
        }
      }
      if (o2::its::TrackerParamConfig::Instance().doUPCIteration) {
        if (!vtxSpan.empty()) {
          if (vtxSpan[0].isFlagSet(Vertex::UPCMode) == 1) { // at least one vertex in this ROF and it is from second vertex iteration
            LOGP(debug, "ROF {} rejected as vertices are from the UPC iteration", iRof);
            processUPCMask[iRof] = true;
            cutUPCVertex++;
            vtxROF.setFlag(o2::itsmft::ROFRecord::VtxUPCMode);
          } else { // in all cases except if as standard mode vertex was found, the ROF was processed with UPC settings
            vtxROF.setFlag(o2::itsmft::ROFRecord::VtxStdMode);
          }
        } else {
          vtxROF.setFlag(o2::itsmft::ROFRecord::VtxUPCMode);
        }
      } else {
        vtxROF.setFlag(o2::itsmft::ROFRecord::VtxStdMode);
      }
      vtxROF.setNEntries(vtxSpan.size());
      bool selROF = vtxSpan.empty();
      for (int iV{0}, iVC{0}; iV < vtxSpan.size(); ++iV) {
        const auto& v = vtxSpan[iV];
        if (multEstConf.isVtxMultCutRequested() && !multEstConf.isPassingVtxMultCut(v.getNContributors())) {
          iVC += v.getNContributors();
          continue; // skip vertex of unwanted multiplicity
        }
        selROF = true;
        vertices.push_back(v);
        if (mIsMC && !VertexerParamConfig::Instance().useTruthSeeding) {
          allVerticesLabels.push_back(vMCRecInfo[iV].first);
          allVerticesPurities.push_back(vMCRecInfo[iV].second);
          if (o2::its::VertexerParamConfig::Instance().outputContLabels) {
            allVerticesContLabels.insert(allVerticesContLabels.end(), vMCContLabels.begin() + iVC, vMCContLabels.begin() + iVC + v.getNContributors());
          }
        }
        iVC += v.getNContributors();
      }
      if (processingMask[iRof] && !selROF) { // passed selection in clusters and not in vertex multiplicity
        LOGP(info, "ROF {} rejected by the vertex multiplicity selection [{},{}]", iRof, multEstConf.cutMultVtxLow, multEstConf.cutMultVtxHigh);
        processingMask[iRof] = selROF;
        cutVertexMult++;
      }
    } else { // cosmics
      vtxVecLoc.emplace_back();
      vtxVecLoc.back().setNContributors(1);
      vtxROF.setNEntries(vtxVecLoc.size());
      for (auto& v : vtxVecLoc) {
        vertices.push_back(v);
      }
      mTimeFrame->addPrimaryVertices(vtxVecLoc, 0);
    }
  }
  if (mRunVertexer) {
    LOG(info) << fmt::format(" - Vertex seeding total elapsed time: {} ms for {} ({} + {}) vertices found in {}/{} ROFs",
                             vertexerElapsedTime,
                             mTimeFrame->getPrimaryVerticesNum(),
                             mTimeFrame->getTotVertIteration()[0],
                             o2::its::VertexerParamConfig::Instance().nIterations > 1 ? mTimeFrame->getTotVertIteration()[1] : 0,
                             trackROFspan.size() - mTimeFrame->getNoVertexROF(),
                             trackROFspan.size());
    LOG(info) << fmt::format(" - FastMultEst: rejected {}/{} ROFs: random/mult.sel:{} (seed {}), vtx.sel:{}", cutRandomMult + cutVertexMult, trackROFspan.size(), cutRandomMult, multEst.lastRandomSeed, cutVertexMult);
  }
  if (mOverrideBeamEstimation) {
    LOG(info) << fmt::format(" - Beam position set to: {}, {} from meanvertex object", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  } else {
    LOG(info) << fmt::format(" - Beam position computed for the TF: {}, {}", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  }
  if (mCosmicsProcessing && compClusters.size() > 1500 * trackROFspan.size()) {
    LOG(error) << "Cosmics processing was requested with an average detector occupancy exceeding 1.e-7, skipping TF processing.";
  } else {

    mTimeFrame->setMultiplicityCutMask(processingMask);
    mTimeFrame->setROFMask(processUPCMask);
    // Run CA tracker
    if (mMode == o2::its::TrackingMode::Async && o2::its::TrackerParamConfig::Instance().fataliseUponFailure) {
      mTracker->clustersToTracks(logger, fatalLogger);
    } else {
      mTracker->clustersToTracks(logger, errorLogger);
    }
    size_t totTracks{mTimeFrame->getNumberOfTracks()}, totClusIDs{mTimeFrame->getNumberOfUsedClusters()};
    if (totTracks) {
      allTracks.reserve(totTracks);
      allClusIdx.reserve(totClusIDs);

      if (mTimeFrame->hasBogusClusters()) {
        LOG(warning) << fmt::format(" - The processed timeframe had {} clusters with wild z coordinates, check the dictionaries", mTimeFrame->hasBogusClusters());
      }

      for (unsigned int iROF{0}; iROF < trackROFvec.size(); ++iROF) {
        auto& tracksROF{trackROFvec[iROF]};
        auto& vtxROF = vertROFvec[iROF];
        auto& tracks = mTimeFrame->getTracks(iROF);
        auto number{tracks.size()};
        auto first{allTracks.size()};
        int offset = -tracksROF.getFirstEntry(); // cluster entry!!!
        tracksROF.setFirstEntry(first);
        tracksROF.setNEntries(number);
        tracksROF.setFlags(vtxROF.getFlags()); // copies 0xffffffff if cosmics
        if (processingMask[iROF]) {
          irFrames.emplace_back(tracksROF.getBCData(), tracksROF.getBCData() + nBCPerTF - 1).info = tracks.size();
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
    } else {
      for (auto& r : trackROFvec) { // reset data copied from the clusters
        r.setFirstEntry(0);
        r.setNEntries(0);
      }
    }
    LOGP(info, "ITSTracker pushed {} tracks and {} vertices", allTracks.size(), vertices.size());
    if (mIsMC) {
      LOGP(info, "ITSTracker pushed {} track labels", allTrackLabels.size());
      LOGP(info, "ITSTracker pushed {} vertex labels", allVerticesLabels.size());
      if (!allVerticesContLabels.empty()) {
        LOGP(info, "ITSTracker pushed {} vertex contributor labels", allVerticesContLabels.size());
      }
      LOGP(info, "ITSTracker pushed {} vertex purities", allVerticesPurities.size());
    }
  }
  mTimeFrame->wipe();
}

void ITSTrackingInterface::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (mOverrideBeamEstimation) {
    pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");
  }
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("itscldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("itsalppar");
    if (pc.inputs().getPos("itsTGeo") >= 0) {
      pc.inputs().get<o2::its::GeometryTGeo*>("itsTGeo");
    }
    GeometryTGeo* geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
    initialise();

    if (pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0) { // print settings only for the 1st pipeling
      o2::its::VertexerParamConfig::Instance().printKeyValues();
      o2::its::TrackerParamConfig::Instance().printKeyValues();
      const auto& vtxParams = mVertexer->getParameters();
      for (size_t it = 0; it < vtxParams.size(); it++) {
        const auto& par = vtxParams[it];
        LOGP(info, "vtxIter#{} : {}", it, par.asString());
      }
      const auto& trParams = mTracker->getParameters();
      for (size_t it = 0; it < trParams.size(); it++) {
        const auto& par = trParams[it];
        LOGP(info, "recoIter#{} : {}", it, par.asString());
      }
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
    LOGP(info, "Mean vertex acquired");
    setMeanVertex((const o2::dataformats::MeanVertexObject*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "GEOMTGEO", 0)) {
    LOG(info) << "ITS GeometryTGeo loaded from ccdb";
    o2::its::GeometryTGeo::adopt((o2::its::GeometryTGeo*)obj);
    return;
  }
}

void ITSTrackingInterface::printSummary() const
{
  mTracker->printSummary();
}

void ITSTrackingInterface::setTraitsFromProvider(VertexerTraitsN* vertexerTraits,
                                                 TrackerTraitsN* trackerTraits,
                                                 TimeFrameN* frame)
{
  mVertexer = std::make_unique<VertexerN>(vertexerTraits);
  mTracker = std::make_unique<TrackerN>(trackerTraits);
  mTimeFrame = frame;
  mVertexer->adoptTimeFrame(*mTimeFrame);
  mTracker->adoptTimeFrame(*mTimeFrame);

  // set common memory resource
  if (!mMemoryPool) {
    mMemoryPool = std::make_shared<BoundedMemoryResource>();
  }
  vertexerTraits->setMemoryPool(mMemoryPool);
  trackerTraits->setMemoryPool(mMemoryPool);
  mTimeFrame->setMemoryPool(mMemoryPool);
  mTracker->setMemoryPool(mMemoryPool);
  mVertexer->setMemoryPool(mMemoryPool);
}

void ITSTrackingInterface::loadROF(gsl::span<itsmft::ROFRecord>& trackROFspan,
                                   gsl::span<const itsmft::CompClusterExt> clusters,
                                   gsl::span<const unsigned char>::iterator& pattIt,
                                   const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  mTimeFrame->loadROFrameData(trackROFspan, clusters, pattIt, mDict, mcLabels);
}
