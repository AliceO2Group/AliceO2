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

#include <memory>

#include <oneapi/tbb/task_arena.h>

#include "DataFormatsITSMFT/DPLAlpideParam.h"
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
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
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

  // prepare data filter
  for (int iLayer = 0; iLayer < ((mDoStaggering) ? NLayers : 1); ++iLayer) {
    mFilter.emplace_back("compClusters", "ITS", "COMPCLUSTERS", iLayer, Lifetime::Timeframe);
    mFilter.emplace_back("patterns", "ITS", "PATTERNS", iLayer, Lifetime::Timeframe);
    mFilter.emplace_back("ROframe", "ITS", "CLUSTERSROF", iLayer, Lifetime::Timeframe);
    if (mIsMC) {
      mFilter.emplace_back("itsmclabels", "ITS", "CLUSTERSMCTR", iLayer, Lifetime::Timeframe);
    }
  }
}

void ITSTrackingInterface::run(framework::ProcessingContext& pc)
{
  if (static bool doneOnce{false}; !doneOnce) {
    doneOnce = true;

    // prepare rof lookup table(s)
    // has to be done here to ensure we get the right number of HB per TF
    const int nOrbitsPerTF = o2::base::GRPGeomHelper::getNHBFPerTF();
    TimeFrameN::ROFOverlapTableN rofTable;
    TimeFrameN::ROFVertexLookupTableN vtxTable;
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    const auto& trackParams = mTracker->getParameters();
    for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
      const unsigned int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / par.getROFLengthInBC(iLayer);
      const LayerTiming timing{.mNROFsTF = (nROFsPerOrbit * nOrbitsPerTF), .mROFLength = (uint32_t)par.getROFLengthInBC(iLayer), .mROFDelay = (uint32_t)par.getROFDelayInBC(iLayer), .mROFBias = (uint32_t)par.getROFBiasInBC(iLayer), .mROFAddTimeErr = trackParams[0].AddTimeError[iLayer]};
      rofTable.defineLayer(iLayer, timing);
      vtxTable.defineLayer(iLayer, timing);
    }
    rofTable.init();
    mTimeFrame->setROFOverlapTable(rofTable);
    vtxTable.init();
    mTimeFrame->setROFVertexLookupTable(vtxTable);
  }

  // filter input and compose
  std::array<gsl::span<const itsmft::CompClusterExt>, NLayers> compClusters;
  std::array<gsl::span<const unsigned char>, NLayers> patterns;
  std::array<gsl::span<const itsmft::ROFRecord>, NLayers> rofsinput;
  std::array<const dataformats::MCTruthContainer<MCCompLabel>*, NLayers> labels{};
  for (const DataRef& ref : framework::InputRecordWalker{pc.inputs(), mFilter}) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (framework::DataRefUtils::match(ref, {"compClusters", framework::ConcreteDataTypeMatcher{"ITS", "COMPCLUSTERS"}})) {
      compClusters[dh->subSpecification] = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>(ref);
    }
    if (framework::DataRefUtils::match(ref, {"patterns", framework::ConcreteDataTypeMatcher{"ITS", "PATTERNS"}})) {
      patterns[dh->subSpecification] = pc.inputs().get<gsl::span<unsigned char>>(ref);
    }
    if (framework::DataRefUtils::match(ref, {"ROframes", framework::ConcreteDataTypeMatcher{"ITS", "CLUSTERSROF"}})) {
      rofsinput[dh->subSpecification] = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>(ref);
    }
    if (framework::DataRefUtils::match(ref, {"itsmclabels", framework::ConcreteDataTypeMatcher{"ITS", "CLUSTERSMCTR"}})) {
      labels[dh->subSpecification] = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>(ref).release();
    }
  }
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  for (int iLayer = 0; iLayer < ((mDoStaggering) ? NLayers : 1); ++iLayer) {
    LOGP(info, "ITSTracker:{} pulled {} clusters, {} RO frames", iLayer, compClusters[iLayer].size(), rofsinput[iLayer].size());
    if (compClusters[iLayer].empty()) {
      LOGP(warn, " -> received no processable data on layer {}", iLayer);
    }
    if (mIsMC) {
      LOG(info) << " -> " << labels[iLayer]->getIndexedSize() << " MC label objects";
    }
  }

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

  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0});

  irFrames.reserve(rofsinput.size());
  int nBCPerTF = alpParams.roFrameLengthInBC;

  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0});
  auto& allTracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0});
  auto& allTrackROFs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0});

  // MC
  static pmr::vector<o2::MCCompLabel> dummyMCLabTracks, dummyMCLabVerts;
  static pmr::vector<float> dummyMCPurVerts;
  auto& allTrackLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "TRACKSMCTR", 0}) : dummyMCLabTracks;
  auto& allVerticesLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "VERTICESMCTR", 0}) : dummyMCLabVerts;
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

  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    gsl::span<const unsigned char>::iterator pattIt = patterns[iLayer].begin();
    loadROF(rofsinput[iLayer], compClusters[iLayer], pattIt, iLayer, labels[iLayer]);
  }

  auto logger = [&](const std::string& s) { LOG(info) << s; };
  auto fatalLogger = [&](const std::string& s) { LOG(fatal) << s; };
  auto errorLogger = [&](const std::string& s) { LOG(error) << s; };

  FastMultEst multEst; // mult estimator
  std::vector<uint8_t> processingMask, processUPCMask;
  // int cutVertexMult{0}, cutUPCVertex{0}, cutRandomMult = int(trackROFvec.size()) - multEst.selectROFs(trackROFvec, compClusters, physTriggers, processingMask);
  // processUPCMask.resize(processingMask.size(), false);
  // mTimeFrame->setMultiplicityCutMask(processingMask);
  float vertexerElapsedTime{0.f};
  if (mRunVertexer) {
    // Run seeding vertexer
    if (!compClusters.empty()) {
      vertexerElapsedTime = mVertexer->clustersToVertices(logger);
      // FIXME: this is a temporary stop-gap measure until we figure the rest out
      const auto& vtx = mTimeFrame->getPrimaryVertices();
      vertices.insert(vertices.begin(), vtx.begin(), vtx.end());
    }
  }
  // const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  // gsl::span<const VertexLabel> vMCRecInfo;
  // gsl::span<const MCCompLabel> vMCContLabels;
  // for (auto iRof{0}; iRof < trackROFspan.size(); ++iRof) {
  //   bounded_vector<Vertex> vtxVecLoc;
  //   auto& vtxROF = vertROFvec.emplace_back(trackROFspan[iRof]);
  //   vtxROF.setFirstEntry(vertices.size());
  //   if (mRunVertexer) {
  //     auto vtxSpan = mTimeFrame->getPrimaryVertices(iRof);
  //     if (mIsMC) {
  //       vMCRecInfo = mTimeFrame->getPrimaryVerticesMCRecInfo(iRof);
  //     }
  //     if (o2::its::TrackerParamConfig::Instance().doUPCIteration) {
  //       if (!vtxSpan.empty()) {
  //         if (vtxSpan[0].isFlagSet(Vertex::UPCMode) == 1) { // at least one vertex in this ROF and it is from second vertex iteration
  //           LOGP(debug, "ROF {} rejected as vertices are from the UPC iteration", iRof);
  //           processUPCMask[iRof] = true;
  //           cutUPCVertex++;
  //           vtxROF.setFlag(o2::itsmft::ROFRecord::VtxUPCMode);
  //         } else { // in all cases except if as standard mode vertex was found, the ROF was processed with UPC settings
  //           vtxROF.setFlag(o2::itsmft::ROFRecord::VtxStdMode);
  //         }
  //       } else {
  //         vtxROF.setFlag(o2::itsmft::ROFRecord::VtxUPCMode);
  //       }
  //     } else {
  //       vtxROF.setFlag(o2::itsmft::ROFRecord::VtxStdMode);
  //     }
  //     vtxROF.setNEntries(vtxSpan.size());
  //     bool selROF = vtxSpan.empty();
  //     for (int iV{0}, iVC{0}; iV < vtxSpan.size(); ++iV) {
  //       const auto& v = vtxSpan[iV];
  //       if (multEstConf.isVtxMultCutRequested() && !multEstConf.isPassingVtxMultCut(v.getNContributors())) {
  //         iVC += v.getNContributors();
  //         continue; // skip vertex of unwanted multiplicity
  //       }
  //       selROF = true;
  //       vertices.push_back(v);
  //       if (mIsMC && !VertexerParamConfig::Instance().useTruthSeeding) {
  //         allVerticesLabels.push_back(vMCRecInfo[iV].first);
  //         allVerticesPurities.push_back(vMCRecInfo[iV].second);
  //       }
  //       iVC += v.getNContributors();
  //     }
  //     if (processingMask[iRof] && !selROF) { // passed selection in clusters and not in vertex multiplicity
  //       LOGP(info, "ROF {} rejected by the vertex multiplicity selection [{},{}]", iRof, multEstConf.cutMultVtxLow, multEstConf.cutMultVtxHigh);
  //       processingMask[iRof] = selROF;
  //       cutVertexMult++;
  //     }
  //   }
  // }
  if (mRunVertexer && !compClusters.empty()) {
    LOG(info) << fmt::format(" - Vertex seeding total elapsed time: {} ms for {} vertices found",
                             vertexerElapsedTime,
                             mTimeFrame->getPrimaryVerticesNum());
    // FIXME
    // LOG(info) << fmt::format(" - FastMultEst: rejected {}/{} ROFs: random/mult.sel:{} (seed {}), vtx.sel:{}", cutRandomMult + cutVertexMult, trackROFspan.size(), cutRandomMult, multEst.lastRandomSeed, cutVertexMult);
  }
  if (mOverrideBeamEstimation) {
    LOG(info) << fmt::format(" - Beam position set to: {}, {} from meanvertex object", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  } else {
    LOG(info) << fmt::format(" - Beam position computed for the TF: {}, {}", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  }
  if (!compClusters.empty()) {
    mTimeFrame->setMultiplicityCutMask(processingMask);
    mTimeFrame->setROFMask(processUPCMask);
    // Run CA tracker
    if (mMode == o2::its::TrackingMode::Async && o2::its::TrackerParamConfig::Instance().fataliseUponFailure) {
      mTracker->clustersToTracks(logger, fatalLogger);
    } else {
      mTracker->clustersToTracks(logger, errorLogger);
    }
  }
  size_t totTracks{mTimeFrame->getNumberOfTracks()}, totClusIDs{mTimeFrame->getNumberOfUsedClusters()};
  if (totTracks) {
    allTracks.reserve(totTracks);
    allClusIdx.reserve(totClusIDs);

    if (mTimeFrame->hasBogusClusters()) {
      LOG(warning) << fmt::format(" - The processed timeframe had {} clusters with wild z coordinates, check the dictionaries", mTimeFrame->hasBogusClusters());
    }

    // FIXME
    // if (processingMask[iROF]) {
    //   irFrames.emplace_back(tracksROF.getBCData(), tracksROF.getBCData() + nBCPerTF - 1).info = tracks.size();
    // }
    auto& tracks = mTimeFrame->getTracks();
    allTrackLabels.reserve(mTimeFrame->getTracksLabel().size()); // should be 0 if not MC
    std::copy(mTimeFrame->getTracksLabel().begin(), mTimeFrame->getTracksLabel().end(), std::back_inserter(allTrackLabels));
    // Some conversions that needs to be moved in the tracker internals
    // also we create the track to clock ROF association here
    // the clock ROF is just the fastest ROF (the number of ROFs does not necessarily reflect the actual ROFs due to
    // possible delay of other layers)
    // tracks are guaranteed to be sorted here by their lower edge
    const auto& clockROF = mTimeFrame->getROFOverlapTableView().getClockLayer();
    // TODO:

    for (unsigned int iTrk{0}; iTrk < tracks.size(); ++iTrk) {
      auto& trc{tracks[iTrk]};
      trc.setFirstClusterEntry(allClusIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters(), nclf = 0;
      for (int ic = TrackITSExt::MaxClusters; ic--;) { // track internally keeps in->out cluster indices, but we want to store the references as out->in!!!
        auto clid = trc.getClusterIndex(ic);
        if (clid >= 0) {
          trc.setClusterSize(ic, mTimeFrame->getClusterSize(ic, clid));
          allClusIdx.push_back(clid);
          nclf++;
        }
      }
      assert(ncl == nclf);
      allTracks.emplace_back(trc);
    }
  }
  LOGP(info, "ITSTracker pushed {} tracks in {} rofs and {} vertices", allTracks.size(), allTrackROFs.size(), vertices.size());
  if (mIsMC) {
    LOGP(info, "ITSTracker pushed {} track labels", allTrackLabels.size());
    LOGP(info, "ITSTracker pushed {} vertex labels", allVerticesLabels.size());
    LOGP(info, "ITSTracker pushed {} vertex purities", allVerticesPurities.size());
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

void ITSTrackingInterface::loadROF(gsl::span<const itsmft::ROFRecord>& trackROFspan,
                                   gsl::span<const itsmft::CompClusterExt> clusters,
                                   gsl::span<const unsigned char>::iterator& pattIt,
                                   int layer,
                                   const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  mTimeFrame->loadROFrameData(trackROFspan, clusters, pattIt, mDict, layer, mcLabels);
}
