#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "iostream"
#include "TChain.h"
#include "TFile.h"
#include "TSystem.h"
#include <TNtuple.h>

#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
// #include "ITStrackingCUDA/VertexerTraitsGPU.h"

// DEBUG
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

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
    outfile = "vertexer_gpu_data.root";
  } else {
    outfile = "vertexer_serial_data.root";
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
  TNtuple tracklets("Tracklets", "Tracklets", "oX:oY:oZ:c1:c2:c3:DCAvtx:DCAz");
  TNtuple comb01("comb01", "comb01", "tanLambda:phi");
  TNtuple comb12("comb12", "comb12", "tanLambda:phi");
  TNtuple clusPhi01("clus_phi01", "clus_phi01", "phi0:phi1");
  TNtuple clusPhi12("clus_phi12", "clus_phi12", "phi1:phi2");
  TNtuple trackdeltaTanLambdas("dtl", "dtl", "deltatanlambda:c0z:c0r:c1z:c1r:c2z:c2r");
  TNtuple centroids("centroids", "centroids", "id:x:y:z:dca");
  TNtuple linesData("ld", "linesdata", "x:xy:xz:y:yz:z");

  std::uint32_t roFrame = 0;

  const int stopAt = (inspEvt == -1) ? itsClusters.GetEntries() : inspEvt + numEvents;
  o2::its::ROframe frame(-123);

  o2::its::VertexerTraits* traits = nullptr;
  // if (useGPU) {
  //   traits = o2::its::createVertexerTraitsGPU();
  // } else {
  traits = o2::its::createVertexerTraits();
  // }
  const o2::its::Line zAxis{ std::array<float, 3>{ 0.f, 0.f, -1.f }, std::array<float, 3>{ 0.f, 0.f, 1.f } };
  o2::its::Vertexer vertexer(traits);

  for (auto& rof : *rofs) {
    itsClusters.GetEntry(rof.getROFEntry().getEvent());
    mcHeaderTree.GetEntry(rof.getROFEntry().getEvent());
    int nclUsed = o2::its::ioutils::loadROFrameData(rof, frame, clusters, labels);
    // float total = vertexer.clustersToVertices(frame, true);
    vertexer.initialiseVertexer(&frame);
    vertexer.findTracklets(useMCcheck);
    // vertexer.findTrivialMCTracklets();
    //     vertexer.processLines();
    //     std::vector<std::array<float, 6>> linesdata = vertexer.getLinesData();
    //     std::vector<std::array<float, 4>> centroidsData = vertexer.getCentroids();
    //     std::vector<o2::its::Line> lines = vertexer.getLines();
    //     std::vector<o2::its::Tracklet> c01 = vertexer.getTracklets01();
    //     std::vector<o2::its::Tracklet> c12 = vertexer.getTracklets12();
    //     std::array<std::vector<o2::its::Cluster>, 3> clusters = vertexer.getClusters();
    //     std::vector<std::array<float, 7>> dtlambdas = vertexer.getDeltaTanLambdas();
    //
    //     for (auto& line : lines)
    //       tracklets.Fill(line.originPoint[0], line.originPoint[1], line.originPoint[2], line.cosinesDirector[0], line.cosinesDirector[1], line.cosinesDirector[2],
    //                      o2::its::Line::getDistanceFromPoint(line, std::array<float, 3>{ 0.f, 0.f, 0.f }), o2::its::Line::getDCA(line, zAxis));
    //     for (int i{ 0 }; i < static_cast<int>(c01.size()); ++i) {
    //       comb01.Fill(c01[i].tanLambda, c01[i].phiCoordinate);
    //       clusPhi01.Fill(clusters[0][c01[i].firstClusterIndex].phiCoordinate, clusters[1][c01[i].secondClusterIndex].phiCoordinate);
    //     }
    //     for (int i{ 0 }; i < static_cast<int>(c12.size()); ++i) {
    //       comb12.Fill(c12[i].tanLambda, c12[i].phiCoordinate);
    //       clusPhi12.Fill(clusters[1][c12[i].firstClusterIndex].phiCoordinate, clusters[2][c12[i].secondClusterIndex].phiCoordinate);
    //     }
    //     for (auto& delta : dtlambdas) {
    //       trackdeltaTanLambdas.Fill(delta.data());
    //     }
    //     for (auto& centroid : centroidsData) {
    //       auto cdata = centroid.data();
    //       centroids.Fill(roFrame, cdata[0], cdata[1], cdata[2], cdata[3]);
    //     }
    //     for (auto& linedata: linesdata) {
    //       linesData.Fill(linedata.data());
    //     }
    //
    vertexer.findVertices();
    vertexer.dumpTraits();
    // std::cout << " - TOTAL elapsed time: " << total << "ms." << std::endl;
    std::vector<Vertex> vertITS = vertexer.exportVertices();
    verticesITS->swap(vertITS);
    // std::array<float,3> trueVertex{mcHeader->GetX(),mcHeader->GetY(),mcHeader->GetZ()}; // UNCOMMENT TO GET THE MC VERTEX POS FOR CURRENT ROFRAME
    outTree.Fill();
  }

  outTree.Write();
  //   tracklets.Write();
  //   comb01.Write();
  //   comb12.Write();
  //   clusPhi01.Write();
  //   clusPhi12.Write();
  //   trackdeltaTanLambdas.Write();
  //   centroids.Write();
  // linesData.Write();
  outputfile->Close();
  return 0;
}
#endif
