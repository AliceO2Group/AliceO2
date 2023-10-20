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

/// @file   GPUWorkflowITS.cxx
/// @author David Rohr

#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/DataRefUtils.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/SerializationMethods.h"
#include "Framework/Logger.h"
#include "Framework/CallbackService.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/RawDataTypes.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "TPCReconstruction/TPCTrackingDigitsPreCheck.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCFastTransform.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonUtils/NameConf.h"
#include "TPCBase/RDHUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceQA.h"
#include "GPUO2Interface.h"
#include "CalibdEdxContainer.h"
#include "GPUNewCalibValues.h"
#include "TPCPadGainCalib.h"
#include "TPCZSLinkMapping.h"
#include "display/GPUDisplayInterface.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/VDriftHelper.h"
#include "CorrectionMapsHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Algorithm/Parser.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/GeometryFlat.h"
#include "ITSBase/GeometryTGeo.h"
#include "CommonUtils/VerbosityConfig.h"
#include "CommonUtils/DebugStreamer.h"
#include <filesystem>
#include <memory> // for make_shared
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>
#include "GPUReconstructionConvert.h"
#include "DetectorsRaw/RDHUtils.h"
#include <TStopwatch.h>
#include <TObjArray.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH1D.h>
#include <TGraphAsymmErrors.h>

#include "ITStracking/TimeFrame.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/PhysTrigger.h"
#include "CommonDataFormat/IRFrame.h"
#include "ITSReconstruction/FastMultEst.h"
#include "ITSReconstruction/FastMultEstConfig.h"

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;

namespace o2::gpu
{

int GPURecoWorkflowSpec::runITSTracking(o2::framework::ProcessingContext& pc)
{
  using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  gsl::span<const o2::itsmft::PhysTrigger> physTriggers;
  std::vector<o2::itsmft::PhysTrigger> fromTRD;
  if (mSpecConfig.itsTriggerType == 2) { // use TRD triggers
    o2::InteractionRecord ir{0, pc.services().get<o2::framework::TimingInfo>().firstTForbit};
    auto trdTriggers = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("phystrig");
    for (const auto& trig : trdTriggers) {
      if (trig.getBCData() >= ir && trig.getNumberOfTracklets()) {
        ir = trig.getBCData();
        fromTRD.emplace_back(o2::itsmft::PhysTrigger{ir, 0});
      }
    }
    physTriggers = gsl::span<const o2::itsmft::PhysTrigger>(fromTRD.data(), fromTRD.size());
  } else if (mSpecConfig.itsTriggerType == 1) { // use Phys triggers from ITS stream
    physTriggers = pc.inputs().get<gsl::span<o2::itsmft::PhysTrigger>>("phystrig");
  }

  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());
  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0, Lifetime::Timeframe});
  irFrames.reserve(rofs.size());

  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance(); // RS: this should come from CCDB
  int nBCPerTF = alpParams.roFrameLengthInBC;

  LOG(info) << "ITSTracker pulled " << compClusters.size() << " clusters, " << rofs.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mSpecConfig.processMC) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("itsmclabels").release();
    // get the array as read-only span, a snapshot is sent forward
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe}, pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("ITSMC2ROframes"));
    LOG(info) << labels->getIndexedSize() << " MC label objects , in " << mc2rofs.size() << " MC events";
  }

  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0, Lifetime::Timeframe});
  auto& allTracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0, Lifetime::Timeframe});
  auto& vertROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe});

  // MC
  static pmr::vector<o2::MCCompLabel> dummyMCLabTracks, dummyMCLabVerts;
  auto& allTrackLabels = mSpecConfig.processMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}) : dummyMCLabTracks;
  auto& allVerticesLabels = mSpecConfig.processMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"ITS", "VERTICESMCTR", 0, Lifetime::Timeframe}) : dummyMCLabVerts;

  std::uint32_t roFrame = 0;

  bool continuous = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(info) << "ITSTracker RO: continuous=" << continuous;

  if (mSpecConfig.itsOverrBeamEst) {
    mITSTimeFrame->setBeamPosition(mMeanVertex->getX(),
                                   mMeanVertex->getY(),
                                   mMeanVertex->getSigmaY2(),
                                   mITSTracker->getParameters()[0].LayerResolution[0],
                                   mITSTracker->getParameters()[0].SystErrorY2[0]);
  }

  mITSTracker->setBz(o2::base::Propagator::Instance()->getNominalBz());

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();

  gsl::span<itsmft::ROFRecord> rofspan(rofs);
  mITSTimeFrame->loadROFrameData(rofspan, compClusters, pattIt, mITSDict, labels);
  pattIt = patterns.begin();
  std::vector<int> savedROF;
  auto logger = [&](std::string s) { LOG(info) << s; };
  auto errorLogger = [&](std::string s) { LOG(error) << s; };

  o2::its::FastMultEst multEst; // mult estimator
  std::vector<bool> processingMask;
  int cutVertexMult{0}, cutRandomMult = int(rofs.size()) - multEst.selectROFs(rofs, compClusters, physTriggers, processingMask);
  mITSTimeFrame->setMultiplicityCutMask(processingMask);
  float vertexerElapsedTime{0.f};
  if (mITSRunVertexer) {
    // Run seeding vertexer
    vertROFvec.reserve(rofs.size());
    vertexerElapsedTime = mITSVertexer->clustersToVertices(logger);
  } else { // cosmics
    mITSTimeFrame->resetRofPV();
  }
  const auto& multEstConf = o2::its::FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  for (auto iRof{0}; iRof < rofspan.size(); ++iRof) {
    std::vector<Vertex> vtxVecLoc;
    auto& vtxROF = vertROFvec.emplace_back(rofspan[iRof]);
    vtxROF.setFirstEntry(vertices.size());
    if (mITSRunVertexer) {
      auto vtxSpan = mITSTimeFrame->getPrimaryVertices(iRof);
      vtxROF.setNEntries(vtxSpan.size());
      bool selROF = vtxSpan.size() == 0;
      for (auto iV{0}; iV < vtxSpan.size(); ++iV) {
        auto& v = vtxSpan[iV];
        if (multEstConf.isVtxMultCutRequested() && !multEstConf.isPassingVtxMultCut(v.getNContributors())) {
          continue; // skip vertex of unwanted multiplicity
        }
        selROF = true;
        vertices.push_back(v);
        if (mSpecConfig.processMC) {
          auto vLabels = mITSTimeFrame->getPrimaryVerticesLabels(iRof)[iV];
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
      mITSTimeFrame->addPrimaryVertices(vtxVecLoc);
    }
  }
  LOG(info) << fmt::format(" - rejected {}/{} ROFs: random/mult.sel:{} (seed {}), vtx.sel:{}", cutRandomMult + cutVertexMult, rofspan.size(), cutRandomMult, multEst.lastRandomSeed, cutVertexMult);
  LOG(info) << fmt::format(" - Vertex seeding total elapsed time: {} ms for {} vertices found in {} ROFs", vertexerElapsedTime, mITSTimeFrame->getPrimaryVerticesNum(), rofspan.size());

  if (mSpecConfig.itsOverrBeamEst) {
    LOG(info) << fmt::format(" - Beam position set to: {}, {} from meanvertex object", mITSTimeFrame->getBeamX(), mITSTimeFrame->getBeamY());
  } else {
    LOG(info) << fmt::format(" - Beam position computed for the TF: {}, {}", mITSTimeFrame->getBeamX(), mITSTimeFrame->getBeamY());
  }
  if (mITSCosmicsProcessing && compClusters.size() > 1500 * rofspan.size()) {
    LOG(error) << "Cosmics processing was requested with an average detector occupancy exceeding 1.e-7, skipping TF processing.";
  } else {

    mITSTimeFrame->setMultiplicityCutMask(processingMask);
    // Run CA tracker
    mITSTracker->clustersToTracks(logger, errorLogger);
    size_t totTracks{mITSTimeFrame->getNumberOfTracks()}, totClusIDs{mITSTimeFrame->getNumberOfUsedClusters()};
    allTracks.reserve(totTracks);
    allClusIdx.reserve(totClusIDs);

    if (mITSTimeFrame->hasBogusClusters()) {
      LOG(warning) << fmt::format(" - The processed timeframe had {} clusters with wild z coordinates, check the dictionaries", mITSTimeFrame->hasBogusClusters());
    }

    for (unsigned int iROF{0}; iROF < rofs.size(); ++iROF) {
      auto& rof{rofs[iROF]};
      auto& tracks = mITSTimeFrame->getTracks(iROF);
      auto number{tracks.size()};
      auto first{allTracks.size()};
      int offset = -rof.getFirstEntry(); // cluster entry!!!
      rof.setFirstEntry(first);
      rof.setNEntries(number);

      if (processingMask[iROF]) {
        irFrames.emplace_back(rof.getBCData(), rof.getBCData() + nBCPerTF - 1).info = tracks.size();
      }

      allTrackLabels.reserve(mITSTimeFrame->getTracksLabel(iROF).size()); // should be 0 if not MC
      std::copy(mITSTimeFrame->getTracksLabel(iROF).begin(), mITSTimeFrame->getTracksLabel(iROF).end(), std::back_inserter(allTrackLabels));
      // Some conversions that needs to be moved in the tracker internals
      for (unsigned int iTrk{0}; iTrk < tracks.size(); ++iTrk) {
        auto& trc{tracks[iTrk]};
        trc.setFirstClusterEntry(allClusIdx.size()); // before adding tracks, create final cluster indices
        int ncl = trc.getNumberOfClusters(), nclf = 0;
        for (int ic = o2::its::TrackITSExt::MaxClusters; ic--;) { // track internally keeps in->out cluster indices, but we want to store the references as out->in!!!
          auto clid = trc.getClusterIndex(ic);
          if (clid >= 0) {
            allClusIdx.push_back(clid);
            nclf++;
          }
        }
        assert(ncl == nclf);
        allTracks.emplace_back(trc);
      }
    }
    LOGP(info, "ITSTracker pushed {} tracks and {} vertices", allTracks.size(), vertices.size());
    if (mSpecConfig.processMC) {
      LOGP(info, "ITSTracker pushed {} track labels", allTrackLabels.size());
      LOGP(info, "ITSTracker pushed {} vertex labels", allVerticesLabels.size());
    }
  }
  return 0;
}

void GPURecoWorkflowSpec::initFunctionITS(InitContext& ic)
{
  std::transform(mITSMode.begin(), mITSMode.end(), mITSMode.begin(), [](unsigned char c) { return std::tolower(c); });
  o2::its::VertexerTraits* vtxTraits = nullptr;
  o2::its::TrackerTraits* trkTraits = nullptr;
  mGPUReco->GetITSTraits(trkTraits, vtxTraits, mITSTimeFrame);
  mITSVertexer = std::make_unique<o2::its::Vertexer>(vtxTraits);
  mITSTracker = std::make_unique<o2::its::Tracker>(trkTraits);
  mITSVertexer->adoptTimeFrame(*mITSTimeFrame);
  mITSTracker->adoptTimeFrame(*mITSTimeFrame);
  mITSRunVertexer = true;
  mITSCosmicsProcessing = false;
  std::vector<o2::its::TrackingParameters> trackParams;

  if (mITSMode == "async") {
    trackParams.resize(3);
    for (auto& param : trackParams) {
      param.ZBins = 64;
      param.PhiBins = 32;
    }
    trackParams[1].TrackletMinPt = 0.2f;
    trackParams[1].CellDeltaTanLambdaSigma *= 2.;
    trackParams[2].TrackletMinPt = 0.1f;
    trackParams[2].CellDeltaTanLambdaSigma *= 4.;
    trackParams[2].MinTrackLength = 4;
    LOG(info) << "Initializing tracker in async. phase reconstruction with " << trackParams.size() << " passes";
  } else if (mITSMode == "sync") {
    trackParams.resize(1);
    trackParams[0].ZBins = 64;
    trackParams[0].PhiBins = 32;
    trackParams[0].MinTrackLength = 4;
    LOG(info) << "Initializing tracker in sync. phase reconstruction with " << trackParams.size() << " passes";
  } else if (mITSMode == "cosmics") {
    mITSCosmicsProcessing = true;
    mITSRunVertexer = false;
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
    throw std::runtime_error(fmt::format("Unsupported ITS tracking mode {:s} ", mITSMode));
  }

  for (auto& params : trackParams) {
    params.CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT;
  }
  mITSTracker->setParameters(trackParams);
}

void GPURecoWorkflowSpec::finaliseCCDBITS(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
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
    if (obj) {
      mMeanVertex = (const o2::dataformats::MeanVertexObject*)obj;
    }
    return;
  }
}

bool GPURecoWorkflowSpec::fetchCalibsCCDBITS(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("itscldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("itsalppar");
    mITSVertexer->getGlobalConfiguration();
    mITSTracker->getGlobalConfiguration();
    if (mSpecConfig.itsOverrBeamEst) {
      pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");
    }
  }
  return false;
}

} // namespace o2::gpu
