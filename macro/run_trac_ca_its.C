#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>
#include "DetectorsCommonDataFormats/NameConf.h"

#include "SimulationDataFormat/MCEventHeader.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
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
#include "DetectorsBase/Propagator.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

#include <TGraph.h>

#include "ITStracking/Configuration.h"

using namespace o2::gpu;
using o2::its::MemoryParameters;
using o2::its::TrackingParameters;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void run_trac_ca_its(bool cosmics = false,
                     std::string path = "./",
                     std::string outputfile = "o2trac_its.root",
                     std::string inputClustersITS = "o2clus_its.root",
                     std::string dictfile = "",
                     std::string inputGRP = "o2sim_grp.root")
{

  gSystem->Load("libO2ITStracking");

  //std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance());
  // std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance("CUDA", true)); // for GPU with CUDA
  // auto* chainITS = rec->AddChain<GPUChainITS>();
  // rec->Init();

  // o2::its::Tracker tracker(chainITS->GetITSTrackerTraits());
  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  o2::its::ROframe event(0, 7);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object";
  }
  o2::base::GeometryManager::loadGeometry(path);
  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma";
  }
  double origD[3] = {0., 0., 0.};
  tracker.setBz(field->getBz(origD));
  // tracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo);

  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  if (!itsClusters.GetBranch("ITSClusterComp")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterComp in the input tree";
  }
  std::vector<o2::itsmft::CompClusterExt>* cclusters = nullptr;
  itsClusters.SetBranchAddress("ITSClusterComp", &cclusters);

  if (!itsClusters.GetBranch("ITSClusterPatt")) {
    LOG(FATAL) << "Did not find ITS cluster patterns branch ITSClusterPatt in the input tree";
  }
  std::vector<unsigned char>* patterns = nullptr;
  itsClusters.SetBranchAddress("ITSClusterPatt", &patterns);

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

  //-------------------------------------------------

  o2::itsmft::TopologyDictionary dict;
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
    dict.readBinaryFile(dictfile);
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

  //-------------------------------------------------

  std::vector<o2::its::TrackITSExt> tracks;
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  std::vector<o2::MCCompLabel> trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);
  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }

  o2::its::VertexerTraits* traits = o2::its::createVertexerTraits();
  o2::its::Vertexer vertexer(traits);

  o2::its::VertexingParameters parameters;
  parameters.phiCut = 0.005f;
  parameters.tanLambdaCut = 0.002f;
  vertexer.setParameters(parameters);

  int roFrameCounter{0};

  std::vector<double> ncls;
  std::vector<double> time;

  std::vector<TrackingParameters> trackParams(1);
  std::vector<MemoryParameters> memParams(1);
  if (cosmics) {
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
  } else {
    // Comment for pp
    // ----
    trackParams.resize(3);
    memParams.resize(3);
    for (int iParam{0}; iParam < 3; ++iParam) {
      for (int iLayer = 0; iLayer < o2::its::constants::its2::TrackletsPerRoad; iLayer++) {
        memParams[iParam].TrackletsMemoryCoefficients[iLayer] = 5.f;
        memParams[iParam].CellsMemoryCoefficients[iLayer] = 0.1f;
      }
    }
    for (auto i{0}; i < 3; ++i) {
      trackParams[i].TrackletMaxDeltaPhi = 3.f;
      trackParams[i].CellMaxDeltaPhi = 3.f;
      trackParams[i].CellMaxDeltaTanLambda = 1.f;
      for (auto j{0}; j < 6; ++j) {
        trackParams[i].TrackletMaxDeltaZ[j] = 10.f;
      }
      for (auto j{0}; j < 5; ++j) {
        trackParams[i].CellMaxDeltaZ[j] = 10.f;
      }
    }
    trackParams[2].MinTrackLength = 4;
    // ---
    // Uncomment for pp
    // trackParams.resize(2);
    // std::array<const float, 5> kmaxDCAxy1 = {1.f * 2.0, 0.4f * 2.0, 0.4f * 2.0, 2.0f * 2.0, 3.f * 2.0};
    // std::array<const float, 5> kmaxDCAz1 = {1.f * 2.0, 0.4f * 2.0, 0.4f * 2.0, 2.0f * 2.0, 3.f * 2.0};
    // std::array<const float, 4> kmaxDN1 = {0.005f * 2.0, 0.0035f * 2.0, 0.009f * 2.0, 0.03f * 2.0};
    // std::array<const float, 4> kmaxDP1 = {0.02f * 2.0, 0.005f * 2.0, 0.006f * 2.0, 0.007f * 2.0};
    // std::array<const float, 6> kmaxDZ1 = {1.f * 2.0, 1.f * 2.0, 2.0f * 2.0, 2.0f * 2.0, 2.0f * 2.0, 2.0f * 2.0};
    // const float kDoublTanL1 = 0.05f * 5.;
    // const float kDoublPhi1 = 0.2f * 5.;
    // trackParams[1].MinTrackLength = 7;
    // trackParams[1].TrackletMaxDeltaPhi = 0.3;
    // trackParams[1].CellMaxDeltaPhi = 0.2 * 2;
    // trackParams[1].CellMaxDeltaTanLambda = 0.05 * 2;
    // std::copy(kmaxDZ1.begin(), kmaxDZ1.end(), trackParams[1].TrackletMaxDeltaZ.begin());
    // std::copy(kmaxDCAxy1.begin(), kmaxDCAxy1.end(), trackParams[1].CellMaxDCA.begin());
    // std::copy(kmaxDCAz1.begin(), kmaxDCAz1.end(), trackParams[1].CellMaxDeltaZ.begin());
    // std::copy(kmaxDP1.begin(), kmaxDP1.end(), trackParams[1].NeighbourMaxDeltaCurvature.begin());
    // std::copy(kmaxDN1.begin(), kmaxDN1.end(), trackParams[1].NeighbourMaxDeltaN.begin());
    // memParams.resize(2);
    // for (auto& coef : memParams[1].CellsMemoryCoefficients)
    //   coef *= 40;
    // for (auto& coef : memParams[1].TrackletsMemoryCoefficients)
    //   coef *= 40;
    // ---
  }

  tracker.setParameters(memParams, trackParams);

  int currentEvent = -1;
  gsl::span<const unsigned char> patt(patterns->data(), patterns->size());
  auto pattIt = patt.begin();
  auto clSpan = gsl::span(cclusters->data(), cclusters->size());

  for (auto& rof : *rofs) {

    auto start = std::chrono::high_resolution_clock::now();
    auto it = pattIt;
    o2::its::ioutils::loadROFrameData(rof, event, clSpan, pattIt, dict, labels);

    vertexer.initialiseVertexer(&event);
    vertexer.findTracklets();
    // vertexer.filterMCTracklets(); // to use MC check
    vertexer.validateTracklets();
    vertexer.findVertices();
    std::vector<Vertex> vertITS = vertexer.exportVertices();
    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
    vtxROF.setNEntries(vertITS.size());
    for (const auto& vtx : vertITS) {
      vertices.push_back(vtx);
    }

    if (!vertITS.empty()) {
      // Using only the first vertex in the list
      std::cout << " - Reconstructed vertex: x = " << vertITS[0].getX() << " y = " << vertITS[0].getY() << " x = " << vertITS[0].getZ() << std::endl;
      event.addPrimaryVertex(vertITS[0].getX(), vertITS[0].getY(), vertITS[0].getZ());
    } else {
      std::cout << " - Vertex not reconstructed, tracking skipped" << std::endl;
    }
    trackClIdx.clear();
    tracksITS.clear();
    tracker.clustersToTracks(event);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_t{end - start};

    ncls.push_back(event.getTotalClusters());
    time.push_back(diff_t.count());

    tracks.swap(tracker.getTracks());
    if (tracks.size()) {
      std::cout << "\t\tFound " << tracks.size() << " tracks" << std::endl;
    }
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

  TGraph* graph = new TGraph(ncls.size(), ncls.data(), time.data());
  graph->SetMarkerStyle(20);
  graph->Draw("AP");
}

#endif