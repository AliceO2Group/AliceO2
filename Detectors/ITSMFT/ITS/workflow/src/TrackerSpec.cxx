// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"
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
  const auto grp = parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    base::GeometryManager::loadGeometry();
    GeometryTGeo* geom = GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                   o2::math_utils::TransformType::T2G));

    auto* chainITS = mRecChain->AddChain<o2::gpu::GPUChainITS>();
    mRecChain->Init();
    mVertexer = std::make_unique<Vertexer>(chainITS->GetITSVertexerTraits());
    mTracker = std::make_unique<Tracker>(chainITS->GetITSTrackerTraits());

    std::vector<TrackingParameters> trackParams;
    std::vector<MemoryParameters> memParams;

    if (mMode == "sync") {
      trackParams.resize(3);
      memParams.resize(3);
      trackParams[0].TrackletMaxDeltaPhi = 0.05f;
      trackParams[1].TrackletMaxDeltaPhi = 0.1f;
      trackParams[2].MinTrackLength = 4;
      trackParams[2].TrackletMaxDeltaPhi = 0.3;
      LOG(INFO) << "Initializing tracker in async. phase reconstruction with " << trackParams.size() << " passes";
    } else if (mMode == "async") {
      trackParams.resize(1);
      memParams.resize(1);
      LOG(INFO) << "Initializing tracker in sync. phase reconstruction with " << trackParams.size() << " passes";
    } else if (mMode == "cosmics") {
      trackParams.resize(1);
      memParams.resize(1);
      trackParams[0].MinTrackLength = 3;
      trackParams[0].TrackletMaxDeltaPhi = o2::its::constants::math::Pi * 0.5f;
      for (int iLayer = 0; iLayer < o2::its::constants::its2::TrackletsPerRoad; iLayer++) {
        trackParams[0].TrackletMaxDeltaZ[iLayer] = o2::its::constants::its2::LayersZCoordinate()[iLayer + 1];
        memParams[0].TrackletsMemoryCoefficients[iLayer] = 0.5f;
        // trackParams[0].TrackletMaxDeltaZ[iLayer] = 10.f;
      }
      for (int iLayer = 0; iLayer < o2::its::constants::its2::CellsPerRoad; iLayer++) {
        trackParams[0].CellMaxDCA[iLayer] = 10000.f;    //cm
        trackParams[0].CellMaxDeltaZ[iLayer] = 10000.f; //cm
        memParams[0].CellsMemoryCoefficients[iLayer] = 0.001f;
      }
      LOG(INFO) << "Initializing tracker in reconstruction for cosmics with " << trackParams.size() << " passes";
    } else {
      throw std::runtime_error(fmt::format("Unsupported ITS tracking mode {:s} ", mMode));
    }
    mTracker->setParameters(memParams, trackParams);

    mVertexer->getGlobalConfiguration();
    mTracker->getGlobalConfiguration();
    LOG(INFO) << Form("%ssing lookup table for material budget approximation", (mTracker->isMatLUT() ? "U" : "Not u"));

    double origD[3] = {0., 0., 0.};
    mTracker->setBz(field->getBz(origD));
  } else {
    throw std::runtime_error(o2::utils::concat_string("Cannot retrieve GRP from the ", filename));
  }

  std::string dictPath = ic.options().get<std::string>("its-dictionary-path");
  std::string dictFile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, dictPath, ".bin");
  if (o2::base::NameConf::pathExists(dictFile)) {
    mDict.readBinaryFile(dictFile);
    LOG(INFO) << "Tracker running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Tracker expects cluster patterns";
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

  LOG(INFO) << "ITSTracker pulled " << compClusters.size() << " clusters, " << rofs.size() << " RO frames";

  const dataformats::MCTruthContainer<MCCompLabel>* labels = nullptr;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mIsMC) {
    labels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("labels").release();
    // get the array as read-only span, a snapshot is send forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
    LOG(INFO) << labels->getIndexedSize() << " MC label objects , in " << mc2rofs.size() << " MC events";
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
  LOG(INFO) << "ITSTracker RO: continuous=" << continuous;

  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  FastMultEst multEst;                                     // mult estimator

  // snippet to convert found tracks to final output tracks with separate cluster indices
  auto copyTracks = [](auto& tracks, auto& allTracks, auto& allClusIdx, int offset = 0) {
    for (auto& trc : tracks) {
      trc.setFirstClusterEntry(allClusIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters(), nclf = 0;
      uint8_t patt = 0;
      for (int ic = TrackITSExt::MaxClusters; ic--;) { // track internally keeps in->out cluster indices, but we want to store the references as out->in!!!
        auto clid = trc.getClusterIndex(ic);
        if (clid >= 0) {
          allClusIdx.push_back(clid + offset);
          nclf++;
          patt |= 0x1 << ic;
        }
      }
      assert(ncl == nclf);
      trc.setPattern(patt);
      allTracks.emplace_back(trc);
    }
  };

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();
  if (continuous) {
    for (auto& rof : rofs) {
      int nclUsed = ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels);
      // prepare in advance output ROFRecords, even if this ROF to be rejected
      int first = allTracks.size();

      if (nclUsed) {
        LOG(INFO) << "ROframe: " << roFrame << ", clusters loaded : " << nclUsed;

        // for vertices output
        auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
        vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
        vtxROF.setNEntries(0);

        if (multEstConf.cutMultClusLow > 0 || multEstConf.cutMultClusHigh > 0) { // cut was requested
          auto mult = multEst.process(rof.getROFData(compClusters));
          if (mult < multEstConf.cutMultClusLow || mult > multEstConf.cutMultClusHigh) {
            LOG(INFO) << "Estimated cluster mult. " << mult << " is outside of requested range "
                      << multEstConf.cutMultClusLow << " : " << multEstConf.cutMultClusHigh << " | ROF " << rof.getBCData();
            rof.setFirstEntry(first);
            rof.setNEntries(0);
            continue;
          }
        }

        mVertexer->clustersToVertices(event);
        auto vtxVecLoc = mVertexer->exportVertices();

        if (multEstConf.cutMultVtxLow > 0 || multEstConf.cutMultVtxHigh > 0) { // cut was requested
          std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> vtxVecSel;
          vtxVecSel.swap(vtxVecLoc);
          for (const auto& vtx : vtxVecSel) {
            if (vtx.getNContributors() < multEstConf.cutMultVtxLow || (multEstConf.cutMultVtxHigh > 0 && vtx.getNContributors() > multEstConf.cutMultVtxHigh)) {
              LOG(INFO) << "Found vertex mult. " << vtx.getNContributors() << " is outside of requested range "
                        << multEstConf.cutMultVtxLow << " : " << multEstConf.cutMultVtxHigh << " | ROF " << rof.getBCData();
              continue; // skip vertex of unwanted multiplicity
            }
            vtxVecLoc.push_back(vtx);
          }
          if (vtxVecLoc.empty()) { // reject ROF
            rof.setFirstEntry(first);
            rof.setNEntries(0);
            continue;
          }
        }

        event.addPrimaryVertices(vtxVecLoc);
        mTracker->setROFrame(roFrame);
        mTracker->clustersToTracks(event);
        tracks.swap(mTracker->getTracks());
        LOG(INFO) << "Found tracks: " << tracks.size();
        int number = tracks.size();
        trackLabels.swap(mTracker->getTrackLabels()); /// FIXME: assignment ctor is not optimal.
        int shiftIdx = -rof.getFirstEntry();          // cluster entry!!!
        rof.setFirstEntry(first);
        rof.setNEntries(number);
        copyTracks(tracks, allTracks, allClusIdx, shiftIdx);
        std::copy(trackLabels.begin(), trackLabels.end(), std::back_inserter(allTrackLabels));
        trackLabels.clear();
        vtxROF.setNEntries(vtxVecLoc.size());
        for (const auto& vtx : vtxVecLoc) {
          vertices.push_back(vtx);
        }
      }
      roFrame++;
    }
  } else {
    ioutils::loadEventData(event, compClusters, pattIt, mDict, labels);
    // RS: FIXME: this part seems to be not functional !!!
    event.addPrimaryVertex(0.f, 0.f, 0.f); //FIXME :  run an actual vertex finder !
    mTracker->clustersToTracks(event);
    tracks.swap(mTracker->getTracks());
    copyTracks(tracks, allTracks, allClusIdx);
    allTrackLabels.swap(mTracker->getTrackLabels()); /// FIXME: assignment ctor is not optimal.
  }

  LOG(INFO) << "ITSTracker pushed " << allTracks.size() << " tracks";
  if (mIsMC) {
    pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}, allTrackLabels);
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }
  mTimer.Stop();
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "ITS CA-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
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
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}}}};
}

} // namespace its
} // namespace o2
