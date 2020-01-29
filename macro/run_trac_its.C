#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <TStopwatch.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>
#include <FairMCEventHeader.h>

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CookedTracker.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/Vertex.h"
#endif

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"

using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void run_trac_its(std::string path = "./", std::string outputfile = "o2trac_its.root",
                  std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                  std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root")
{

  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  // Setup timer
  TStopwatch timer;

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object";
  }
  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2GRot)); // request cached transforms

  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma";
  }

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree";
  }
  std::vector<o2::itsmft::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  MCLabCont* labels = nullptr;
  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(WARNING) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  } else {
    itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);
  }

  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }

  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersMC2ROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);

  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);

  itsClusters.GetEntry(0);
  //<<<---------- attach input data ---------------<<<

  //>>>--------- create/attach output ------------->>>
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "Cooked ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  MCLabCont trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  outTree.Branch("ITSTracksROF", &rofs);
  outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);
  //<<<--------- create/attach output -------------<<<

  //=================== INIT ==================
  Int_t n = 1;            // Number of threads
  Bool_t mcTruth = kTRUE; // kFALSE if no comparison with MC is needed
  o2::its::CookedTracker tracker(n);
  tracker.setContinuousMode(isContITS);
  tracker.setBz(field->solenoidField()); // in kG
  tracker.setGeometry(gman);
  if (mcTruth) {
    tracker.setMCTruthContainers(labels, trackLabelsPtr);
  }
  //===========================================

  o2::its::VertexerTraits vertexerTraits;
  o2::its::Vertexer vertexer(&vertexerTraits);
  o2::its::ROframe event(0);

  auto clSpan = gsl::span(clusters->data(), clusters->size());
  for (auto& rof : *rofs) {
    o2::its::ioutils::loadROFrameData(rof, event, clSpan, labels);
    vertexer.clustersToVertices(event);
    auto verticesL = vertexer.exportVertices();

    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
    vtxROF.setNEntries(verticesL.size());
    for (const auto& vtx : verticesL) {
      vertices.push_back(vtx);
    }
    if (verticesL.empty()) {
      verticesL.emplace_back();
    }
    tracker.setVertices(verticesL);
    tracker.process(clSpan, tracksITS, trackClIdx, rof);
  }
  outTree.Fill();

  outFile.cd();
  outTree.Write();
  outFile.Close();

  timer.Stop();
  timer.Print();
}
