// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CookedTrackerSpec.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void CookedTrackerDPL::init(InitContext& ic)
{
  auto nthreads = ic.options().get<int>("nthreads");
  mTracker.setNumberOfThreads(nthreads);
  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    o2::base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    o2::base::GeometryManager::loadGeometry();
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                              o2::TransformType::T2G));
    mTracker.setGeometry(geom);

    double origD[3] = {0., 0., 0.};
    mTracker.setBz(field->getBz(origD));

    bool continuous = mGRP->isDetContinuousReadOut("ITS");
    LOG(INFO) << "ITSCookedTracker RO: continuous=" << continuous;
    mTracker.setContinuousMode(continuous);
  } else {
    LOG(ERROR) << "Cannot retrieve GRP from the " << filename.c_str() << " file !";
    mState = 0;
  }

  std::string dictPath = ic.options().get<std::string>("its-dictionary-path");
  std::string dictFile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, dictPath, ".bin");
  if (o2::base::NameConf::pathExists(dictFile)) {
    mDict.readBinaryFile(dictFile);
    LOG(INFO) << "Tracker running with a provided dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Tracker expects cluster patterns";
  }
  mState = 1;
}

void CookedTrackerDPL::run(ProcessingContext& pc)
{
  if (mState != 1)
    return;

  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto clusters = pc.inputs().get<gsl::span<o2::itsmft::Cluster>>("clusters");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    // get the array as read-onlt span, a snapshot is send forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
  }

  LOG(INFO) << "ITSCookedTracker pulled " << clusters.size() << " clusters, in "
            << rofs.size() << " RO frames";

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackLabels;
  if (mUseMC) {
    mTracker.setMCTruthContainers(labels.get(), &trackLabels);
  }

  o2::its::VertexerTraits vertexerTraits;
  o2::its::Vertexer vertexer(&vertexerTraits);
  o2::its::ROframe event(0);

  auto& vertROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe});
  auto& tracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0, Lifetime::Timeframe});
  auto& clusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0, Lifetime::Timeframe});

  gsl::span<const unsigned char>::iterator pattIt = patterns.begin();
  for (auto& rof : rofs) {
    auto it = pattIt;
    o2::its::ioutils::loadROFrameData(rof, event, compClusters, pattIt, mDict, labels.get());
    vertexer.clustersToVertices(event);
    auto vtxVecLoc = vertexer.exportVertices();

    // for vertices output
    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
    vtxROF.setNEntries(vtxVecLoc.size());
    for (const auto& vtx : vtxVecLoc) {
      vertices.push_back(vtx);
    }

    if (vtxVecLoc.empty()) {
      vtxVecLoc.emplace_back();
    }
    mTracker.setVertices(vtxVecLoc);
    mTracker.process(compClusters, it, mDict, tracks, clusIdx, rof);
  }

  LOG(INFO) << "ITSCookedTracker pushed " << tracks.size() << " tracks";

  if (mUseMC) {
    pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}, trackLabels);
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }

  mState = 2;
}

DataProcessorSpec getCookedTrackerSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusters", "ITS", "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSClusterROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-cooked-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CookedTrackerDPL>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"nthreads", VariantType::Int, 1, {"Number of threads"}}}};
}

} // namespace its
} // namespace o2
