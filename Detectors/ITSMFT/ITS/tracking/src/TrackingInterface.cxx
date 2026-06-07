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

#include <algorithm>
#include <array>
#include <format>
#include <memory>

#include <oneapi/tbb/task_arena.h>

#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "ITSBase/GeometryTGeo.h"

#include "ITStracking/FastMultEstConfig.h"
#include "ITStracking/FastMultEst.h"

#include "ITStracking/ROFLookupTables.h"
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
  overrideParameters(trackParams, vertParams);
  LOGP(info, "Initializing tracker in {} phase reconstruction with {} passes for tracking and {}/{} for vertexing", TrackingMode::toString(mMode), trackParams.size(), o2::its::VertexerParamConfig::Instance().nIterations, vertParams.size());
  mTracker->setParameters(trackParams);
  mVertexer->setParameters(vertParams);
  TrackingParameters vertexTrackingParams;
  mTimeFrame->initVertexingTopology(vertexTrackingParams);
  if (!trackParams.empty()) {
    mTimeFrame->initDefaultTrackingTopology(trackParams[0], NLayers);
    mTimeFrame->initTrackerTopologies(gsl::span<const TrackingParameters>(trackParams.data(), trackParams.size()));
  }

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
  mTimeFrame->setIsStaggered(mDoStaggering);

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
  const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();

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

  bool hasClusters = false;
  for (int iLayer = 0; iLayer < ((mDoStaggering) ? NLayers : 1); ++iLayer) {
    LOGP(info, "ITSTracker{} pulled {} clusters, {} RO frames", ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""), compClusters[iLayer].size(), rofsinput[iLayer].size());
    if (compClusters[iLayer].empty()) {
      LOGP(warn, " -> received no processable data{}", (mDoStaggering) ? std::format(" on layer {}", iLayer) : "");
    } else {
      hasClusters = true;
    }
    if (mIsMC) {
      LOG(info) << " -> " << labels[iLayer]->getIndexedSize() << " MC label objects";
    }
  }

  const auto& tfInfo = pc.services().get<o2::framework::TimingInfo>();
  gsl::span<const o2::itsmft::PhysTrigger> physTriggers;
  std::vector<o2::itsmft::PhysTrigger> fromTRD;
  if (mUseTriggers == 2) { // use TRD triggers
    o2::InteractionRecord irFirstTF{0, tfInfo.firstTForbit};
    auto trdTriggers = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("phystrig");
    for (const auto& trig : trdTriggers) {
      if (trig.getBCData() >= irFirstTF && trig.getNumberOfTracklets()) {
        irFirstTF = trig.getBCData();
        fromTRD.emplace_back(o2::itsmft::PhysTrigger{.ir = irFirstTF, .data = 0});
      }
    }
    physTriggers = gsl::span<const o2::itsmft::PhysTrigger>(fromTRD.data(), fromTRD.size());
  } else if (mUseTriggers == 1) { // use Phys triggers from ITS stream
    physTriggers = pc.inputs().get<gsl::span<o2::itsmft::PhysTrigger>>("phystrig");
  }

  const int clockLayerId{mDoStaggering ? mTimeFrame->getROFOverlapTableView().getClock() : 0};
  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0});
  irFrames.reserve(rofsinput[clockLayerId].size());

  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0});
  auto& allTracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0});
  auto& allTrackROFs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0});
  auto& vertROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "VERTICESROF", 0}); // TODO fill this!

  // MC
  static pmr::vector<o2::MCCompLabel> dummyMCLabTracks, dummyMCLabVerts;
  static pmr::vector<float> dummyMCPurVerts;
  auto& allTrackLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "TRACKSMCTR", 0}) : dummyMCLabTracks;
  auto& allVerticesLabels = mIsMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "VERTICESMCTR", 0}) : dummyMCLabVerts;
  auto& allVerticesPurities = mIsMC ? pc.outputs().make<std::vector<float>>(Output{"ITS", "VERTICESMCPUR", 0}) : dummyMCPurVerts;

  const auto clock = mTimeFrame->getROFOverlapTableView().getClock();
  const auto& clockLayer = mTimeFrame->getROFOverlapTableView().getClockLayer();
  auto setBCData = [&](auto& rofs) {
    for (size_t iROF{0}; iROF < rofs.size(); ++iROF) { // set BC data
      auto& rof = rofs[iROF];
      int orb = (iROF * par.getROFLengthInBC(clock) / o2::constants::lhc::LHCMaxBunches) + tfInfo.firstTForbit;
      int bc = (iROF * par.getROFLengthInBC(clock) % o2::constants::lhc::LHCMaxBunches) + par.getROFDelayInBC(clock);
      o2::InteractionRecord ir(bc, orb);
      rof.setBCData(ir);
      rof.setROFrame(iROF);
      rof.setNEntries(0);
      rof.setFirstEntry(-1);
    }
  };

  if (!hasClusters) {
    // skip processing if no data is received entirely but still create empty output so consumers do not wait
    allTrackROFs.resize(clockLayer.mNROFsTF);
    vertROFvec.resize(clockLayer.mNROFsTF);
    setBCData(allTrackROFs);
    setBCData(vertROFvec);
    return;
  }

  if (mOverrideBeamEstimation) {
    mTimeFrame->setBeamPosition(mMeanVertex->getX(),
                                mMeanVertex->getY(),
                                mMeanVertex->getSigmaY2(),
                                mTracker->getParameters()[0].LayerResolution[0],
                                mTracker->getParameters()[0].SystErrorY2[0]);
  }

  mTracker->setBz(o2::base::Propagator::Instance()->getNominalBz());
  mTracker->setTimeSlice(tfInfo.timeslice);

  for (int iLayer = 0; iLayer < ((mDoStaggering) ? NLayers : 1); ++iLayer) {
    gsl::span<const unsigned char>::iterator pattIt = patterns[iLayer].begin();
    loadROF(rofsinput[iLayer], compClusters[iLayer], pattIt, ((mDoStaggering) ? iLayer : -1), labels[iLayer]);
  }

  auto logger = [&](const std::string& s) { LOG(info) << s; };
  auto fatalLogger = [&](const std::string& s) { LOG(fatal) << s; };
  auto errorLogger = [&](const std::string& s) { LOG(error) << s; };

  FastMultEst multEst; // mult estimator
  o2::its::ROFMaskTable<NLayers> processMultiplictyMask{mTimeFrame->getROFOverlapTable()}, processUPCMask{mTimeFrame->getROFOverlapTable()};
  multEst.selectROFs(rofsinput, compClusters, physTriggers, tfInfo.firstTForbit, mDoStaggering, mTimeFrame->getROFOverlapTableView(), processMultiplictyMask);
  mTimeFrame->setMultiplicityCutMask(processMultiplictyMask);
  for (int iLayer = 0; iLayer < ((mDoStaggering) ? NLayers : 1); ++iLayer) {
    mTimeFrame->getROFMaskView().print(iLayer);
  }

  float vertexerElapsedTime{0.f}, trackerElapsedTime{0.f};
  if (mRunVertexer) {
    // Run seeding vertexer
    vertexerElapsedTime = mVertexer->clustersToVertices(logger);
    const auto& vtx = mTimeFrame->getPrimaryVertices();
    vertices.insert(vertices.begin(), vtx.begin(), vtx.end());
    if (mIsMC) {
      allVerticesLabels.reserve(vertices.size());
      allVerticesPurities.reserve(vertices.size());
      for (const auto& lbl : mTimeFrame->getPrimaryVerticesLabels()) {
        allVerticesLabels.push_back(lbl.first);
        allVerticesPurities.push_back(lbl.second);
      }
    }
  }
  multEst.selectROFsWithVertices(vertices, mTimeFrame->getROFOverlapTableView(), processMultiplictyMask);

  auto clockROFspan = rofsinput[clockLayerId];
  auto clockTiming = mTimeFrame->getROFOverlapTableView().getClockLayer();
  for (auto iRof{0}; iRof < clockROFspan.size(); ++iRof) {
    auto& vtxROF = vertROFvec.emplace_back(clockROFspan[iRof]);
    if (mRunVertexer) {
      auto vtxSpan = mTimeFrame->getPrimaryVertices(clockLayerId, iRof);
      if (o2::its::TrackerParamConfig::Instance().doUPCIteration) {
        if (!vtxSpan.empty()) {
          bool hasUPC = std::any_of(vtxSpan.begin(), vtxSpan.end(), [](const auto& v) { return v.isFlagSet(Vertex::UPCMode); });
          if (hasUPC) { // at least one vertex in this ROF and it is from second vertex iteration
            LOGP(debug, "ROF {} accepted as vertices are from the UPC iteration", iRof);
            const auto startBC = clockTiming.getROFStartInBC(iRof);
            const auto endBC = clockTiming.getROFEndInBC(iRof);
            processUPCMask.selectROF({startBC, endBC - startBC});
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
    }
  }

  if (mRunVertexer && hasClusters) {
    LOGP(info, " + Vertex seeding total elapsed time: {} ms for {} vertices found", vertexerElapsedTime, mTimeFrame->getPrimaryVerticesNum());
  }

  if (mOverrideBeamEstimation) {
    LOG(info) << fmt::format(" + Beam position set to: {}, {} from meanvertex object", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  } else {
    LOG(info) << fmt::format(" + Beam position computed for the TF: {}, {}", mTimeFrame->getBeamX(), mTimeFrame->getBeamY());
  }

  if (hasClusters) {
    mTimeFrame->setMultiplicityCutMask(processMultiplictyMask);
    mTimeFrame->setUPCCutMask(processUPCMask);
    if (mMode == o2::its::TrackingMode::Async && o2::its::TrackerParamConfig::Instance().fataliseUponFailure) {
      trackerElapsedTime = mTracker->clustersToTracks(logger, fatalLogger);
    } else {
      trackerElapsedTime = mTracker->clustersToTracks(logger, errorLogger);
    }
    LOGP(info, " + Tracking total elapse time: {} ms for {} tracks found", trackerElapsedTime, mTimeFrame->getNumberOfTracks());
  }
  if constexpr (constants::DoTimeBenchmarks) {
    const auto& trackConf = o2::its::TrackerParamConfig::Instance();
    const auto& vertConf = o2::its::VertexerParamConfig::Instance();
    logger(std::format("=== TimeSlice {} processing completed in: {:.2f} ms using {}/{} thread(s) ===", tfInfo.timeslice, trackerElapsedTime + vertexerElapsedTime, vertConf.nThreads, trackConf.nThreads));
  }

  size_t totTracks{mTimeFrame->getNumberOfTracks()}, totClusIDs{mTimeFrame->getNumberOfUsedClusters()};
  if (totTracks) {
    allTracks.reserve(totTracks);
    allClusIdx.reserve(totClusIDs);

    if (mTimeFrame->hasBogusClusters()) {
      LOG(warning) << fmt::format(" + The processed timeframe had {} clusters with wild z coordinates, check the dictionaries", mTimeFrame->hasBogusClusters());
    }
  }

  auto& tracks = mTimeFrame->getTracks();
  allTrackLabels.reserve(mTimeFrame->getTracksLabel().size()); // should be 0 if not MC
  std::copy(mTimeFrame->getTracksLabel().begin(), mTimeFrame->getTracksLabel().end(), std::back_inserter(allTrackLabels));
  // create the track to clock ROF association here
  // the clock ROF is just the fastest ROF
  // the number of ROFs does not necessarily reflect the actual ROFs
  // due to possible delay of other layers, however it is guaranteed to be >=0
  // tracks are guaranteed to be sorted here by their lower edge
  // we pick whatever is the largest possible number of rofs since there might be tracks/vertices which are beyond
  // the clock layer
  int highestROF{0};
  for (const auto& trc : tracks) {
    highestROF = std::max(highestROF, (int)clockLayer.getROF(trc.getTimeStamp()));
  }
  for (const auto& vtx : vertices) {
    highestROF = std::max(highestROF, (int)clockLayer.getROF(vtx.getTimeStamp().lower()));
  }
  highestROF = std::max(highestROF, (int)clockLayer.mNROFsTF);
  allTrackROFs.resize(highestROF);
  vertROFvec.resize(highestROF);
  setBCData(allTrackROFs);
  setBCData(vertROFvec);

  mTimeFrame->useMultiplictyMask(); // use multiplicty selection for IR frames

  std::vector<int> rofEntries(highestROF + 1, 0);
  for (unsigned int iTrk{0}; iTrk < tracks.size(); ++iTrk) {
    auto& trc{tracks[iTrk]};
    trc.setFirstClusterEntry((int)allClusIdx.size()); // before adding tracks, create final cluster indices
    int ncl = trc.getNumberOfClusters(), nclf = 0;
    for (int ic = TrackITSExt::MaxClusters; ic--;) { // track internally keeps in->out cluster indices, but we want to store the references as out->in!!!
      auto clid = trc.getClusterIndex(ic);
      if (clid >= 0) {
        trc.setClusterSize(ic, mTimeFrame->getClusterSize((mDoStaggering) ? ic : 0, clid));
        allClusIdx.push_back(clid);
        nclf++;
      }
    }
    assert(ncl == nclf);
    allTracks.emplace_back(trc);
    auto rof = clockLayer.getROF(trc.getTimeStamp());
    ++rofEntries[rof];
  }
  std::exclusive_scan(rofEntries.begin(), rofEntries.end(), rofEntries.begin(), 0);
  for (size_t iROF{0}; iROF < allTrackROFs.size(); ++iROF) {
    allTrackROFs[iROF].setFirstEntry(rofEntries[iROF]);
    allTrackROFs[iROF].setNEntries(rofEntries[iROF + 1] - rofEntries[iROF]);
    allTrackROFs[iROF].setFlags(vertROFvec[iROF].getFlags());
    if (mTimeFrame->getROFMaskView().isROFEnabled(clockLayerId, (int)iROF)) {
      auto& irFrame = irFrames.emplace_back(allTrackROFs[iROF].getBCData(), allTrackROFs[iROF].getBCData() + clockLayer.mROFLength - 1);
      irFrame.info = allTrackROFs[iROF].getNEntries();
    }
  }
  // same thing for vertices rofs
  std::fill(rofEntries.begin(), rofEntries.end(), 0);
  for (const auto& vtx : vertices) {
    auto rof = clockLayer.getROF(vtx.getTimeStamp().lower());
    ++rofEntries[rof];
  }
  std::exclusive_scan(rofEntries.begin(), rofEntries.end(), rofEntries.begin(), 0);
  for (size_t iROF{0}; iROF < vertROFvec.size(); ++iROF) {
    vertROFvec[iROF].setFirstEntry(rofEntries[iROF]);
    vertROFvec[iROF].setNEntries(rofEntries[iROF + 1] - rofEntries[iROF]);
  }

  LOGP(info, "ITSTracker pushed {} tracks in {} rofs and {} vertices {}", allTracks.size(), allTrackROFs.size(), vertices.size(), ((mDoStaggering) ? "in staggered-readout mode" : ""));
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
    requestTopologyDictionary(pc);
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("itsalppar");
    if (pc.inputs().getPos("itsTGeo") >= 0) {
      pc.inputs().get<o2::its::GeometryTGeo*>("itsTGeo");
    }
    GeometryTGeo* geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
    initialise();

    if (pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0) { // print settings only for the 1st pipeling
      // print all used settings
      if (o2::its::FastMultEstConfig::Instance().isRequested()) {
        o2::its::FastMultEstConfig::Instance().printKeyValues(true, true);
      }
      const auto& vtxParams = mVertexer->getParameters();
      if (!vtxParams.empty()) {
        o2::its::VertexerParamConfig::Instance().printKeyValues(true, true);
      }
      const auto& trParams = mTracker->getParameters();
      if (!trParams.empty()) {
        o2::its::TrackerParamConfig::Instance().printKeyValues(true, true);
      }
      // quick summary
      for (size_t it = 0; it < vtxParams.size(); it++) {
        const auto& par = vtxParams[it];
        LOGP(info, "vtxIter#{} : {}", it, par.asString());
      }
      for (size_t it = 0; it < trParams.size(); it++) {
        const auto& par = trParams[it];
        LOGP(info, "recoIter#{} : {}", it, par.asString());
      }
    }

    // prepare rof lookup table(s)
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    const int nOrbitsPerTF = o2::base::GRPGeomHelper::getNHBFPerTF();
    TimeFrameN::ROFOverlapTableN rofTable;
    TimeFrameN::ROFVertexLookupTableN vtxTable;
    const auto& trackParams = mTracker->getParameters();
    for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
      const unsigned int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / par.getROFLengthInBC(iLayer);
      const LayerTiming timing{
        .mNROFsTF = (nROFsPerOrbit * nOrbitsPerTF),
        .mROFLength = (uint32_t)par.getROFLengthInBC(iLayer),
        .mROFDelay = (uint32_t)par.getROFDelayInBC(iLayer),
        .mROFBias = (uint32_t)par.getROFBiasInBC(iLayer),
        .mROFAddTimeErr = (trackParams.empty() ? o2::its::TrackerParamConfig::Instance().addTimeError[iLayer] : trackParams[0].AddTimeError[iLayer])};
      rofTable.defineLayer(iLayer, timing);
      vtxTable.defineLayer(iLayer, timing);
    }
    rofTable.init();
    mTimeFrame->setROFOverlapTable(rofTable);
    vtxTable.init();
    mTimeFrame->setROFVertexLookupTable(vtxTable);
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
  mVertexer->printSummary();
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

void ITSTrackingInterface::requestTopologyDictionary(framework::ProcessingContext& pc)
{
  pc.inputs().get<o2::itsmft::TopologyDictionary*>("itscldict"); // just to trigger the finaliseCCDB
}

void ITSTrackingInterface::loadROF(gsl::span<const itsmft::ROFRecord>& trackROFspan,
                                   gsl::span<const itsmft::CompClusterExt> clusters,
                                   gsl::span<const unsigned char>::iterator& pattIt,
                                   int layer,
                                   const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  mTimeFrame->loadROFrameData(trackROFspan, clusters, pattIt, mDict, layer, mcLabels);
}
