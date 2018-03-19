#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TH1I.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include <sstream>
#include <FairLogger.h>
#include "FairRunAna.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairGeoParSet.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/IOUtils.h"
#include "ITSReconstruction/CA/vertexer/Vertexer.h"
#include "ITSReconstruction/CA/vertexer/ClusterLines.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TLine.h"
#include "FairMCEventHeader.h"

using o2::ITS::CA::Line;
using o2::ITS::CA::Cluster;

void run_vert_ca_its( int startfrom = 0, int nevents = 1000, std::string path = "./", std::string inputClustersITS = "o2clus_its.root",
                     std::string paramfilename = "o2sim_par.root", std::string mctruthfile = "o2sim.root", std::string outfile = "vertexer_data.root")
{
  o2::ITS::CA::Event event;

  if (path.back() != '/') {
    path += '/';
  }

  // Setup Runtime DB
  TFile paramFile(paramfilename.data());
  paramFile.Get("FairGeoParSet");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  //>>>---------- attach input data --------------->>>
  TChain mcHeaderTree("o2sim");
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());
  mcHeaderTree.AddFile((path + mctruthfile).data());

  //<<<---------- attach input data ---------------<<<
  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree" << FairLogger::endl;
  }
  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree" << FairLogger::endl;
  }
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  FairMCEventHeader* header = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader.")) {
    LOG(FATAL) << "Did not find MC event header in the input header file." << FairLogger::endl;
  }

  mcHeaderTree.SetBranchAddress("MCEventHeader.", &header);

  TFile* outputfile = new TFile(outfile.data(), "recreate");
  TNtuple* verTupleResiduals = new TNtuple("residuals", "residuals", "residualX:residualY:residualZ");
  TNtuple* numVerticesTuple = new TNtuple("numvertices", "numvertices", "vertices");
  TNtuple* dcaTuple = new TNtuple("dcapoints","dcapoints","DCAp1:DCAp2");
  TNtuple* dcaTuplevtx = new TNtuple("dcapoints","dcapoints","DCAvtx");
  TH1I* hDCA = new TH1I("DCA", "DCA Z axis", 1000, 0., 5.);
  
  int startevent = static_cast<int>(std::max(0, startfrom));
  std::cout<<itsClusters.GetEntries()<<std::endl;
  int endevent = std::min(static_cast<int>(itsClusters.GetEntries()), nevents+startevent);
  std::cout<<"endevent: "<<endevent<<std::endl;
  const Line zAxis{std::array<float, 3>{ 0., 0., 0.}, std::array<float, 3>{ 0., 0., 1.}};
  const Line zParAxis{std::array<float, 3>{1., 0., 0}, std::array<float, 3>{ 1., 0., -1. }};
  const Line zOrtAxis{std::array<float, 3>{1., 1., 0}, std::array<float, 3>{ 0., 0., 0. }};
  const Line zSkewAxis{std::array<float, 3>{1., 0., 0}, std::array<float, 3>{ 0., 1., 0. }};
  TCanvas* c1 = new TCanvas;
  TCanvas* c2 = new TCanvas;
  c1->cd();
  c1->DrawFrame(-17, -4, 17, 4);
  c2->cd();
  c2->DrawFrame(-4.5,-4.5,4.5,4.5); 
  std::cout<<"parallela "<<Line::getDCA(zParAxis, zAxis)<<"\tortogonale "<< Line::getDCA(zAxis, zOrtAxis) << "\tsghemba "<<Line::getDCA(zAxis,zSkewAxis)<<std::endl;
  for (int iEvent {startevent}; iEvent < endevent; ++iEvent) { // itsClusters.GetEntries()
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);
    o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);
    o2::ITS::CA::Vertexer vertexer(event);
    vertexer.initialise(0.02, 0.005, 0.04, 0.8, 3);
    vertexer.generateTracklets();
    // vertexer.findTracklets();
    auto tracklets = vertexer.getTracklets();
    

    // Analysis + Debug
    std::string title = "DCA"+ std::to_string( iEvent );
    std::string titlev = title + "_vtx"; 

    
    // TH1F* hDCA = new TH1F(title.data(), title.data(), 1000, 0., 5.);
    // TH1F* hDCAvtx = new TH1F(titlev.data(), titlev.data(), 1000, 0., 5.);

    for ( auto& tracklet : tracklets ) {
      // dcaTuple->Fill(Line::getDistanceFromPoint2(tracklet, tracklet.originPoint), Line::getDistanceFromPoint2(tracklet, tracklet.destinationPoint));
      // dcaTuplevtx->Fill(Line::getDistanceFromPoint(tracklet, std::array<float, 3>{0.,0.,0.}));
      dcaTuplevtx->Fill(Line::getDCA(tracklet, zAxis));
      TLine* line = new TLine(tracklet.originPoint[2],  tracklet.originPoint[1], tracklet.destinationPoint[2], tracklet.destinationPoint[1]);
      TLine* line2 = new TLine(tracklet.originPoint[0], tracklet.originPoint[1], tracklet.destinationPoint[0], tracklet.destinationPoint[1]);

      // if (Line::getDistanceFromPoint2(tracklet, std::array<float, 3>{0.,0.,0.}) > 1.f) {
      /*  for ( int i {0}; i<3 ; ++i ) {
          std::cout<<"origin"<<tracklet.originPoint[i]<<std::endl;
          std::cout<<"dest"<<tracklet.destinationPoint[i]<<std::endl;
          std::cout<<"cd"<<tracklet.cosinesDirector[i]<<std::endl;
          std::cout<<"dca"<<Line::getDistanceFromPoint2(tracklet, std::array<float, 3>{0.,0.,0.})<<std::endl;
          line->Draw();
        }*/
        // line->Draw();
      // }
      // hDCA->Fill(Line::getDCA(Tracklets[i], zAxis));
      // hDCAvtx->Fill(Line::getDistanceFromPoint(tracklet, std::array<float, 3>{0.,0.,0.}));
      // if (Line::getDCA(tracklet, zAxis) > 9) {
      //   std::cout<< "origin: "<<tracklet.originPoint[0]<<"\t"<<tracklet.originPoint[1]<<"\t"<<tracklet.originPoint[2]<<std::endl;
      //   std::cout<< "dest: "<<tracklet.destinationPoint[0]<<"\t"<<tracklet.destinationPoint[1]<<"\t"<<tracklet.destinationPoint[2]<<std::endl;
      //   std::cout<< "cosines: "<<tracklet.cosinesDirector[0]<<"\t"<<tracklet.cosinesDirector[1]<<"\t"<<tracklet.cosinesDirector[2]<<std::endl;
        c1->cd();
        line->Draw();
        c2->cd();
        line2->Draw();
      }
    }
    // std::array<std::vector<Cluster>, 3> clusters = vertexer.getClusters();
    /* for ( int i{0}; i<3; ++i) {
      std::string titleclus = "Rlayer" + std::to_string( i ) + "_" + std::to_string( iEvent );
      TH1F* hradii = new TH1F(titleclus.data(), titleclus.data(), 1000, 0., 5.);
      for ( auto& cluster : clusters[i] ) {
        hradii->Fill(cluster.rCoordinate);
      }
      hradii->Write();
    }*/ 
    
    // vertexer.findVertices();
    // string title = "indices evt " + std::to_string( iEvent );
    // std::vector<std::array<int, 3>> triplets = vertexer.getTriplets();
    // for ( auto& triplet : triplets ) {
    //   indexHisto->Fill(triplet[0], triplet[1]);
    // }
    // indexHisto->Write();
    // std::vector<std::array<float, 3>> vertices = vertexer.getVertices();
    // numVerticesTuple->Fill(static_cast<int>(vertices.size()));
    // for (auto& vertex : vertices ) { // ATM works only with {0,0,0} MC vertices
    //    verTupleResiduals->Fill(vertex[0], vertex[1], vertex[2]);
    // }
    // hDCA->Write();
    // hDCAvtx->Write();
  // }
  // dcaTuple->Write();
  // verTupleResiduals->Write();
  // numVerticesTuple->Write();
  dcaTuplevtx->Write();
  outputfile->Close();
}
#endif