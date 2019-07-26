#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>
#include <TNtuple.h>

#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
// DEBUG
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"

// #define __VERTEXER_ITS_DEBUG
#if defined(__VERTEXER_ITS_GPU)
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUChainITS.h"
#endif

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using namespace o2::gpu;

int run_primary_vertexer_ITS(const bool useGPU = false,
                             const bool useMCcheck = false,
                             const int inspEvt = -1,
                             const int numEvents = 1,
                             const std::string inputClustersITS = "o2clus_its.root",
                             const std::string inputGRP = "o2sim_grp.root",
                             const std::string simfilename = "o2sim.root",
                             const std::string paramfilename = "O2geometry.root",
                             const std::string path = "./")
{
  std::string outfile;

  if (useGPU) {
    if (useMCcheck) {
      outfile = "vertexer_gpu_data_MCCheck.root";
    } else {
      outfile = "vertexer_gpu_data.root";
    }
  } else {
    if (useMCcheck) {
      outfile = "vertexer_serial_data_MCCheck.root";
    } else {
      outfile = "vertexer_serial_data.root";
    }
  }

  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  const bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  const bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  std::cout << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode" << std::endl;
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  // Setup Runtime DB
  TFile paramFile((path + paramfilename).data());
  paramFile.Get("FAIRGeom");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Get event header
  TChain mcHeaderTree("o2sim");
  mcHeaderTree.AddFile((path + simfilename).data());
  o2::dataformats::MCEventHeader* mcHeader = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader.")) {
    LOG(FATAL) << "Did not find MC event header in the input header file." << FairLogger::endl;
  }
  mcHeaderTree.SetBranchAddress("MCEventHeader.", &mcHeader);

  // get clusters
  std::vector<o2::itsmft::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  TChain itsClustersROF("ITSClustersROF");
  itsClustersROF.AddFile((path + inputClustersITS).data());

  if (!itsClustersROF.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree" << FairLogger::endl;
  }
  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClustersROF.SetBranchAddress("ITSClustersROF", &rofs);
  itsClustersROF.GetEntry(0);

  //get labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  TFile* outputfile = new TFile(outfile.data(), "recreate");

  TTree outTree("o2sim", "Vertexer Vertices");
  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>* verticesITS =
    new std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>;
  outTree.Branch("ITSVertices", &verticesITS);

  // DEBUG
#if defined(__VERTEXER_ITS_DEBUG)
  TNtuple tracklets("Tracklets", "Tracklets", "oX:oY:oZ:c1:c2:c3:DCAvtx:DCAz");
  TNtuple comb01("comb01", "comb01", "tanLambda:phi");
  TNtuple comb12("comb12", "comb12", "tanLambda:phi");
  TNtuple clusPhi01("clus_phi01", "clus_phi01", "phi0:phi1");
  TNtuple clusPhi12("clus_phi12", "clus_phi12", "phi1:phi2");
  TNtuple trackdeltaTanLambdas("dtl", "dtl", "deltatanlambda:c0z:c0r:c1z:c1r:c2z:c2r:evtId:valid");
  TNtuple centroids("centroids", "centroids", "id:x:y:z:dca");
  TNtuple linesData("ld", "linesdata", "x:xy:xz:y:yz:z");
  const o2::its::Line zAxis{ std::array<float, 3>{ 0.f, 0.f, -1.f }, std::array<float, 3>{ 0.f, 0.f, 1.f } };
#endif

  // Benchmarks
  TNtuple foundVerticesBenchmark("foundVerticesBenchmark", "Found vertices benchmark", "frameId:foundVertices:nTracklets");
  TNtuple timeBenchmark("timeBenchmark", "Time benchmarks", "init:trackletFinder:vertexFinder");
  // \Benchmarks

  std::uint32_t roFrame = 0;

  //Settings
  o2::its::VertexingParameters parameters;
  // e.g. parameters.clusterContributorsCut = 5;
  //\Settings

  const int stopAt = (inspEvt == -1) ? rofs->size() : inspEvt + numEvents;
  const int startAt = (inspEvt == -1) ? 0 : inspEvt;

  o2::its::ROframe frame(-123);

  o2::its::VertexerTraits* traits = nullptr;

#if defined(__VERTEXER_ITS_GPU)
  if (useGPU) {
    traits = o2::its::createVertexerTraitsGPU();
  }
#else
  if (!useGPU) {
    traits = o2::its::createVertexerTraits();
  }
#endif

  o2::its::Vertexer vertexer(traits);
  vertexer.setParameters(parameters);

  for (size_t iROfCount{ static_cast<size_t>(startAt) }; iROfCount < static_cast<size_t>(stopAt); ++iROfCount) {
    auto& rof = (*rofs)[iROfCount];
    std::cout << "ROframe: " << iROfCount << std::endl;
    itsClusters.GetEntry(rof.getROFEntry().getEvent());
    mcHeaderTree.GetEntry(rof.getROFEntry().getEvent());
    int nclUsed = o2::its::ioutils::loadROFrameData(rof, frame, clusters, labels);

    std::array<float, 3> total{ 0.f, 0.f, 0.f };
    o2::its::ROframe* eventptr = &frame;

    total[0] = vertexer.evaluateTask(&o2::its::Vertexer::initialiseVertexer, "Vertexer initialisation", std::cout, eventptr);
    total[1] = vertexer.evaluateTask(&o2::its::Vertexer::findTracklets, "Tracklet finding", std::cout, useMCcheck);
    // total[1] = vertexer.evaluateTask(&o2::its::Vertexer::findTrivialMCTracklets, "Trivial Tracklet finding", std::cout);
    total[2] = vertexer.evaluateTask(&o2::its::Vertexer::findVertices, "Vertex finding", std::cout);
    // vertexer.findVerticesDBS();

#if defined(__VERTEXER_ITS_DEBUG)
    // vertexer.processLines();
    vertexer.dumpTraits();
    std::vector<std::array<float, 6>> linesdata = vertexer.getLinesData();
    std::vector<std::array<float, 4>> centroidsData = vertexer.getCentroids();
    std::vector<o2::its::Line> lines = vertexer.getLines();
    std::vector<o2::its::Tracklet> c01 = vertexer.getTracklets01();
    std::vector<o2::its::Tracklet> c12 = vertexer.getTracklets12();
    std::array<std::vector<o2::its::Cluster>, 3> clusters = vertexer.getClusters();
    std::vector<std::array<float, 9>> dtlambdas = vertexer.getDeltaTanLambdas();
    for (auto& line : lines)
      tracklets.Fill(line.originPoint[0], line.originPoint[1], line.originPoint[2], line.cosinesDirector[0], line.cosinesDirector[1], line.cosinesDirector[2],
                     o2::its::Line::getDistanceFromPoint(line, std::array<float, 3>{ 0.f, 0.f, 0.f }), o2::its::Line::getDCA(line, zAxis));
    for (int i{ 0 }; i < static_cast<int>(c01.size()); ++i) {
      comb01.Fill(c01[i].tanLambda, c01[i].phiCoordinate);
      clusPhi01.Fill(clusters[0][c01[i].firstClusterIndex].phiCoordinate, clusters[1][c01[i].secondClusterIndex].phiCoordinate);
    }
    for (int i{ 0 }; i < static_cast<int>(c12.size()); ++i) {
      comb12.Fill(c12[i].tanLambda, c12[i].phiCoordinate);
      clusPhi12.Fill(clusters[1][c12[i].firstClusterIndex].phiCoordinate, clusters[2][c12[i].secondClusterIndex].phiCoordinate);
    }
    for (auto& delta : dtlambdas) {
      trackdeltaTanLambdas.Fill(delta.data());
    }
    for (auto& centroid : centroidsData) {
      auto cdata = centroid.data();
      centroids.Fill(roFrame, cdata[0], cdata[1], cdata[2], cdata[3]);
    }
    for (auto& linedata : linesdata) {
      linesData.Fill(linedata.data());
    }
#endif

    std::vector<Vertex> vertITS = vertexer.exportVertices();
    const size_t numVert = vertITS.size();
    foundVerticesBenchmark.Fill(static_cast<float>(iROfCount), static_cast<float>(numVert) /* , static_cast<float>(linesdata.size())*/);
    verticesITS->swap(vertITS);
    // TODO: get vertexer postion form MC truth

    if (numVert > 0) {
      timeBenchmark.Fill(total[0], total[1], total[2]);
    }
    outTree.Fill();
  }

  outTree.Write();
  foundVerticesBenchmark.Write();
  timeBenchmark.Write();

#if defined(__VERTEXER_ITS_DEBUG)
  tracklets.Write();
  comb01.Write();
  comb12.Write();
  clusPhi01.Write();
  clusPhi12.Write();
  trackdeltaTanLambdas.Write();
  centroids.Write();
  linesData.Write();
#endif

  outputfile->Close();
  return 0;
}
#endif