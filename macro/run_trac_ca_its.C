#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>
#include <chrono>
#include <iostream>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>

#include "SimulationDataFormat/MCEventHeader.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/Vertexer.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUO2Interface.h"
#define GPUCA_O2_LIB // Temporary workaround, must not be set globally, but needed right now for GPUReconstruction/GPUChain
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

#include <TGraph.h>

#include "ITStracking/Configuration.h"

using namespace o2::gpu;
using o2::its::MemoryParameters;
using o2::its::TrackingParameters;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void run_trac_ca_its(bool useITSVertex = false,
                     std::string path = "./",
                     std::string outputfile = "o2trac_its.root",
                     std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                     std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root",
                     std::string paramfilename = "o2sim_par.root")
{

  gSystem->Load("libO2ITStracking.so");

  std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance());
  // std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance("CUDA", true)); // for GPU with CUDA
  auto* chainITS = rec->AddChain<GPUChainITS>();
  rec->Init();

  o2::its::Tracker tracker(chainITS->GetITSTrackerTraits());
  //o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  o2::its::ROframe event(0);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object";
  }
  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma";
  }
  double origD[3] = {0., 0., 0.};
  tracker.setBz(field->getBz(origD));

  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Get event header
  TChain mcHeaderTree("o2sim");
  mcHeaderTree.AddFile(simfilename.data());
  o2::dataformats::MCEventHeader* mcHeader = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader.")) {
    LOG(FATAL) << "Did not find MC event header in the input header file.";
  }
  mcHeaderTree.SetBranchAddress("MCEventHeader.", &mcHeader);

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  //<<<---------- attach input data ---------------<<<
  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree";
  }
  std::vector<o2::itsmft::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  }
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  std::vector<o2::its::TrackITSExt> tracks;
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  MCLabCont trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);

  TChain itsClustersROF("ITSClustersROF");
  itsClustersROF.AddFile((path + inputClustersITS).data());

  if (!itsClustersROF.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClustersROF.SetBranchAddress("ITSClustersROF", &rofs);
  itsClustersROF.GetEntry(0);

  o2::its::VertexerTraits* traits = o2::its::createVertexerTraits();
  o2::its::Vertexer vertexer(traits);

  int roFrameCounter{ 0 };


  std::vector<double> ncls;
  std::vector<double> time;


  std::vector<TrackingParameters> trackParams(3);
  trackParams[0].TrackletMaxDeltaPhi = 0.05f;
  trackParams[1].TrackletMaxDeltaPhi = 0.1f;
  trackParams[2].MinTrackLength = 4;
  trackParams[2].TrackletMaxDeltaPhi = 0.3;

  std::vector<MemoryParameters> memParams(3);

  tracker.setParameters(memParams, trackParams);
  
  int currentEvent = -1;
  for (auto& rof : *rofs) {

    if (currentEvent != rof.getROFEntry().getEvent()) {
      currentEvent = rof.getROFEntry().getEvent();
      itsClusters.GetEntry(rof.getROFEntry().getEvent());
      mcHeaderTree.GetEntry(rof.getROFEntry().getEvent());
    }

    auto start = std::chrono::high_resolution_clock::now();

    o2::its::ioutils::loadROFrameData(rof, event, clusters, labels);

    if (useITSVertex) {

      vertexer.initialiseVertexer(&event);
      vertexer.findTracklets();
      // vertexer.filterMCTracklets(); // to use MC check
      vertexer.validateTracklets();
      vertexer.findVertices();
      std::vector<Vertex> vertITS = vertexer.exportVertices();
      if (!vertITS.empty()) {
        // Using only the first vertex in the list
        cout << " - Reconstructed vertexer: x = " << vertITS[0].getX() << " y = " << vertITS[0].getY() << " x = " << vertITS[0].getZ() << std::endl;
        event.addPrimaryVertex(vertITS[0].getX(), vertITS[0].getY(), vertITS[0].getZ());
      } else {
        cout << " - Vertex not reconstructed, tracking skipped" << std::endl;
      }
    } else {
      std::cout << Form("MC Vertex for roFrame %i: %f %f %f", roFrameCounter, mcHeader->GetX(), mcHeader->GetY(), mcHeader->GetZ()) << std::endl;
      event.addPrimaryVertex(mcHeader->GetX(), mcHeader->GetY(), mcHeader->GetZ());
    }
    trackClIdx.clear();
    tracksITS.clear();
    tracker.clustersToTracks(event);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_t{ end - start };

    ncls.push_back(event.getTotalClusters());
    time.push_back(diff_t.count());

    tracks.swap(tracker.getTracks());
    for (auto& trc : tracks) {
      trc.setFirstClusterEntry(trackClIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters();
      for (int ic = 0; ic < ncl; ic++) {
        trackClIdx.push_back(trc.getClusterIndex(ic));
      }
      tracksITS.emplace_back(trc);
    }

    trackLabels = tracker.getTrackLabels(); /// FIXME: assignment ctor is not optimal.
    outTree.Fill();
    roFrameCounter++;
  }

  outFile.cd();
  outTree.Write();
  outFile.Close();

  TGraph* graph = new TGraph(ncls.size(),ncls.data(),time.data());
  graph->SetMarkerStyle(20);
  graph->Draw("AP");

}

#endif
