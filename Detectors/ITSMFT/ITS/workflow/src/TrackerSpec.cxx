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

/// @file   TrackerSpec.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSMFTReconstruction/ClustererParam.h"

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"
#include "CommonDataFormat/IRFrame.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#include "ITSReconstruction/FastMultEstConfig.h"
#include "ITSReconstruction/FastMultEst.h"
#include <fmt/format.h>

namespace o2
{
using namespace framework;
namespace its
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

TrackerDPL::TrackerDPL(bool isMC, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType) : mIsMC{isMC}, mMode{trModeS}, mRecChain{o2::gpu::GPUReconstruction::CreateInstance(dType, true)}
{
  std::transform(mMode.begin(), mMode.end(), mMode.begin(), [](unsigned char c) { return std::tolower(c); });
}

void TrackerDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = parameters::GRPObject::loadFrom(filename);
  if (grp) {
    mGRP.reset(grp);
    base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    base::GeometryManager::loadGeometry();
    GeometryTGeo* geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                   o2::math_utils::TransformType::T2G));

    std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
    std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
    if (o2::utils::Str::pathExists(matLUTFile)) {
      auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
      o2::base::Propagator::Instance()->setMatLUT(lut);
      LOG(info) << "Loaded material LUT from " << matLUTFile;
    } else {
      LOG(info) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
    }

    auto* chainITS = mRecChain->AddChain<o2::gpu::GPUChainITS>();
    mVertexer = std::make_unique<Vertexer>(chainITS->GetITSVertexerTraits());
    mTracker = std::make_unique<Tracker>(new TrackerTraitsCPU);

    std::vector<TrackingParameters> trackParams;
    std::vector<MemoryParameters> memParams;

    mRunVertexer = true;
    if (mMode == "async") {

      trackParams.resize(3);
      trackParams[0].TrackletMaxDeltaPhi = 0.05f;
      trackParams[0].DeltaROF = 0;
      trackParams[1].CopyCuts(trackParams[0], 2.);
      trackParams[1].DeltaROF = 0;
      trackParams[2].CopyCuts(trackParams[1], 2.);
      trackParams[2].DeltaROF = 1;
      trackParams[2].MinTrackLength = 4;
      memParams.resize(3);
      LOG(info) << "Initializing tracker in async. phase reconstruction with " << trackParams.size() << " passes";

    } else if (mMode == "sync_misaligned") {

      trackParams.resize(1);
      trackParams[0].PhiBins = 32;
      trackParams[0].ZBins = 64;
      trackParams[0].TrackletMaxDeltaPhi *= 2;
      trackParams[0].CellDeltaTanLambdaSigma *= 10;
      trackParams[0].LayerMisalignment[0] = 3.e-2;
      trackParams[0].LayerMisalignment[1] = 3.e-2;
      trackParams[0].LayerMisalignment[2] = 3.e-2;
      trackParams[0].LayerMisalignment[3] = 1.e-1;
      trackParams[0].LayerMisalignment[4] = 1.e-1;
      trackParams[0].LayerMisalignment[5] = 1.e-1;
      trackParams[0].LayerMisalignment[6] = 1.e-1;
      trackParams[0].FitIterationMaxChi2[0] = 1.e28;
      trackParams[0].FitIterationMaxChi2[1] = 1.e28;
      trackParams[0].MinTrackLength = 4;
      memParams.resize(1);
      LOG(info) << "Initializing tracker in misaligned sync. phase reconstruction with " << trackParams.size() << " passes";

    } else if (mMode == "sync") {
      memParams.resize(1);
      trackParams.resize(1);
      LOG(info) << "Initializing tracker in sync. phase reconstruction with " << trackParams.size() << " passes";
    } else if (mMode == "cosmics") {
      mRunVertexer = false;
      trackParams.resize(1);
      memParams.resize(1);
      trackParams[0].MinTrackLength = 4;
      trackParams[0].TrackletMaxDeltaPhi = o2::its::constants::math::Pi * 0.5f;
      trackParams[0].CellDeltaTanLambdaSigma *= 400;
      trackParams[0].PhiBins = 4;
      trackParams[0].ZBins = 16;
      trackParams[0].PVres = 1.e5f;
      trackParams[0].LayerMisalignment[0] = 3.e-2;
      trackParams[0].LayerMisalignment[1] = 3.e-2;
      trackParams[0].LayerMisalignment[2] = 3.e-2;
      trackParams[0].LayerMisalignment[3] = 1.e-1;
      trackParams[0].LayerMisalignment[4] = 1.e-1;
      trackParams[0].LayerMisalignment[5] = 1.e-1;
      trackParams[0].LayerMisalignment[6] = 1.e-1;
      trackParams[0].FitIterationMaxChi2[0] = 1.e28;
      trackParams[0].FitIterationMaxChi2[1] = 1.e28;
      LOG(info) << "Initializing tracker in reconstruction for cosmics with " << trackParams.size() << " passes";

    } else {
      throw std::runtime_error(fmt::format("Unsupported ITS tracking mode {:s} ", mMode));
    }
    mTracker->setParameters(memParams, trackParams);

    mVertexer->getGlobalConfiguration();
    mTracker->getGlobalConfiguration();
    LOG(info) << Form("Using %s for material budget approximation", (mTracker->isMatLUT() ? "lookup table" : "TGeometry"));

    double origD[3] = {0., 0., 0.};
    mTracker->setBz(field->getBz(origD));
  } else {
    throw std::runtime_error(o2::utils::Str::concat_string("Cannot retrieve GRP from the ", filename));
  }

  std::string dictPath = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance().dictFilePath;
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, dictPath, "bin");
  if (o2::utils::Str::pathExists(dictFile)) {
    mDict.readBinaryFile(dictFile);
    LOG(info) << "Tracker running with a provided dictionary: " << dictFile;
  } else {
    LOG(info) << "Dictionary " << dictFile << " is absent, Tracker expects cluster patterns";
  }
}

void TrackerDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0, Lifetime::Timeframe});

  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance(); // RS: this should come from CCDB
  int nBCPerTF = alpParams.roFrameLengthInBC;

  LOG(info) << "ITSTracker pulled " << compClusters.size() << " clusters, " << rofs.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mIsMC) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release();
    // get the array as read-only span, a snapshot is send forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(info) << labels->getIndexedSize() << " MC label objects , in " << mc2rofs.size() << " MC events";
  }

  std::vector<o2::its::TrackITSExt> tracks;
  auto& allClusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0, Lifetime::Timeframe});
  std::vector<o2::MCCompLabel> trackLabels;
  auto& allTracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0, Lifetime::Timeframe});
  std::vector<o2::MCCompLabel> allTrackLabels;

  auto& vertROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe});

  std::uint32_t roFrame = 0;
  ROframe event(0, 7);

  bool continuous = mGRP->isDetContinuousReadOut("ITS");
  LOG(info) << "ITSTracker RO: continuous=" << continuous;

  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  FastMultEst multEst;                                     // mult estimator

  TimeFrame mTimeFrame;
  mTracker->adoptTimeFrame(mTimeFrame);

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();

  gsl::span<itsmft::ROFRecord> rofspan(rofs);
  mTimeFrame.loadROFrameData(rofspan, compClusters, pattIt, mDict, labels);
  pattIt = patterns.begin();
  std::vector<int> savedROF;
  auto logger = [&](std::string s) { LOG(info) << s; };
  float vertexerElapsedTime{0.f};
  int nclUsed = 0;

  std::vector<bool> processingMask;
  int cutClusterMult{0}, cutVertexMult{0}, cutTotalMult{0};
  for (auto& rof : rofspan) {
    nclUsed += ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels);
    // prepare in advance output ROFRecords, even if this ROF to be rejected
    int first = allTracks.size();

    // for vertices output
    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
    vtxROF.setNEntries(0);

    bool multCut = (multEstConf.cutMultClusLow <= 0 && multEstConf.cutMultClusHigh <= 0); // cut was requested
    if (!multCut) {
      float mult = multEst.process(rof.getROFData(compClusters));
      multCut = mult >= multEstConf.cutMultClusLow && mult <= multEstConf.cutMultClusHigh;
      LOG(debug) << fmt::format("ROF {} rejected by the cluster multiplicity selection [{},{}]", processingMask.size(), multEstConf.cutMultClusLow, multEstConf.cutMultClusHigh);
      cutClusterMult += !multCut;
    }

    std::vector<Vertex> vtxVecLoc;
    if (multCut) {
      if (mRunVertexer) {
        vertexerElapsedTime += mVertexer->clustersToVertices(event, false, logger);
        auto allVerts = mVertexer->exportVertices();
        multCut = allVerts.size() == 0;
        for (const auto& vtx : allVerts) {
          if (vtx.getNContributors() < multEstConf.cutMultVtxLow || (multEstConf.cutMultVtxHigh > 0 && vtx.getNContributors() > multEstConf.cutMultVtxHigh)) {
            continue; // skip vertex of unwanted multiplicity
          }
          multCut = true; // At least one passes the selection
          vtxVecLoc.push_back(vtx);
        }
      } else {
        vtxVecLoc.emplace_back(Vertex());
        vtxVecLoc.back().setNContributors(1);
      }

      if (!multCut) {
        LOG(debug) << fmt::format("ROF {} rejected by the vertex multiplicity selection [{},{}]", processingMask.size(), multEstConf.cutMultVtxLow, multEstConf.cutMultVtxHigh);
        cutVertexMult++;
      }
    }
    cutTotalMult += !multCut;
    processingMask.push_back(multCut);
    mTimeFrame.addPrimaryVertices(vtxVecLoc);

    vtxROF.setNEntries(vtxVecLoc.size());
    for (const auto& vtx : vtxVecLoc) {
      vertices.push_back(vtx);
    }
    savedROF.push_back(roFrame);
    roFrame++;
  }

  LOG(info) << fmt::format(" - In total, multiplicity selection rejected {}/{} ROFs", cutTotalMult, rofspan.size());
  LOG(info) << fmt::format("\t - Cluster multiplicity selection rejected {}/{} ROFs", cutClusterMult, rofspan.size());
  LOG(info) << fmt::format("\t - Vertex multiplicity selection rejected {}/{} ROFs", cutVertexMult, rofspan.size());
  LOG(info) << fmt::format(" - Vertex seeding total elapsed time: {} ms for {} clusters in {} ROFs", vertexerElapsedTime, nclUsed, rofspan.size());
  LOG(info) << fmt::format(" - Beam position computed for the TF: {}, {}", mTimeFrame.getBeamX(), mTimeFrame.getBeamY());

  mTimeFrame.setMultiplicityCutMask(processingMask);
  mTracker->clustersToTracks(logger);
  if (mTimeFrame.hasBogusClusters()) {
    LOG(warning) << fmt::format(" - The processed timeframe had {} clusters with wild z coordinates, check the dictionaries", mTimeFrame.hasBogusClusters());
  }

  for (unsigned int iROF{0}; iROF < rofs.size(); ++iROF) {

    auto& rof{rofs[iROF]};
    tracks = mTimeFrame.getTracks(iROF);
    trackLabels = mTimeFrame.getTracksLabel(iROF);
    auto number{tracks.size()};
    auto first{allTracks.size()};
    int offset = -rof.getFirstEntry(); // cluster entry!!!
    rof.setFirstEntry(first);
    rof.setNEntries(number);

    if (tracks.size()) {
      irFrames.emplace_back(rof.getBCData(), rof.getBCData() + nBCPerTF - 1);
    }
    std::copy(trackLabels.begin(), trackLabels.end(), std::back_inserter(allTrackLabels));
    // Some conversions that needs to be moved in the tracker internals
    for (unsigned int iTrk{0}; iTrk < tracks.size(); ++iTrk) {
      auto& trc{tracks[iTrk]};
      trc.setFirstClusterEntry(allClusIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters(), nclf = 0;
      for (int ic = TrackITSExt::MaxClusters; ic--;) { // track internally keeps in->out cluster indices, but we want to store the references as out->in!!!
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

  LOG(info) << "ITSTracker pushed " << allTracks.size() << " tracks";
  if (mIsMC) {
    LOG(info) << "ITSTracker pushed " << allTrackLabels.size() << " track labels";

    pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }
  mTimer.Stop();
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "ITS CA-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "IRFRAMES", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(useMC, trModeS, dType)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace its
} // namespace o2
