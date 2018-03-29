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
#include "TPolyLine3D.h"
#include "TView3D.h"
#include "FairMCEventHeader.h"

using o2::ITS::CA::Line;
using o2::ITS::CA::Cluster;
using o2::ITS::CA::MathUtils::calculatePhiCoordinate;
using o2::ITS::CA::MathUtils::calculateRCoordinate;

void run_vert_ca_its( int startfrom = 0, int nevents = 1000, std::string path = "./", std::string inputClustersITS = "o2clus_its.root",
                     std::string paramfilename = "o2sim_par.root", std::string mctruthfile = "o2sim.root", std::string outfile = "vertexer_data.root")
{
  o2::ITS::CA::Event event;
  if (path.back() != '/')
    path += '/';

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
  if (!itsClusters.GetBranch("ITSCluster"))
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree" << FairLogger::endl;
  
  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth"))
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree" << FairLogger::endl;
  
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  FairMCEventHeader* header = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader."))
    LOG(FATAL) << "Did not find MC event header in the input header file." << FairLogger::endl;

  mcHeaderTree.SetBranchAddress("MCEventHeader.", &header);
  TFile* outputfile = new TFile(outfile.data(), "recreate");

  // TNtuple* verTupleResiduals = new TNtuple("residuals", "residuals", "residualX:residualY:residualZ");
  // TNtuple* numVerticesTuple = new TNtuple("numvertices", "numvertices", "vertices");
  TNtuple* dcaTupleZAxis = new TNtuple("dcazaxis","dcazaxis","DCAZaxis");
  TNtuple* dcaTupleZAxis_montecarlo = new TNtuple("dcazaxis_montecarlo","dcazaxis_montecarlo","DCAZaxis_montecarlo");
  TNtuple* deltaZProjMc = new TNtuple("deltaZProjMc","deltaZProjMc","#deltaZ_{Proj}");
  TNtuple* deltaZProj = new TNtuple("deltaZProj","deltaZProj","#deltaZ_{Proj}");
  // TNtuple* delta#PhiProj = new TNtuple("delta#PhiProjections","delta#PhiProjections","#delta#Phi_{Proj}");
  
  int startevent = static_cast<int>(std::max(0, startfrom));
  int endevent = std::min(static_cast<int>(itsClusters.GetEntries()), nevents+startevent);
  std::cout<<"Will start from event "<<startevent<<" and end on event " <<endevent<<std::endl;

  TH1I* hXD = new TH1I("XDestinationPoint", "XDestinationPoint", 1000, -4.5, 4.5);
  TH1I* hXDmc = new TH1I("XDestinationPointMC", "XDestinationPointMC", 1000, -4.5, 4.5);
  TH1I* hYD = new TH1I("YDestinationPoint", "YDestinationPoint", 1000, -4.5, 4.5);
  TH1I* hYDmc = new TH1I("YDestinationPointMC", "YDestinationPointMC", 1000, -4.5, 4.5);
  TH1I* hZD = new TH1I("ZDestinationPoint", "ZDestinationPoint", 1000, -17.5, 17.5);
  TH1I* hZDmc = new TH1I("ZDestinationPointMC", "ZDestinationPointMC", 1000, -17.5, 17.5);
  TH1I* hPhiD = new TH1I("PhiDestinationPoint", "PhiDestinationPoint", 50, 0., 6.3);
  TH1I* hPhiDmc = new TH1I("PhiDestinationPointMC", "PhiDestinationPointMC", 50, 0., 6.3);
  TH1I* hRD = new TH1I("RDestinationPoint", "RDestinationPoint", 500, 0., 4.5);
  TH1I* hRDmc = new TH1I("RDestinationPointMC", "RDestinationPointMC", 500, 0., 4.5);
  
  TH1I* hXO = new TH1I("XOriginPoint", "XOriginPoint", 1000, -4.5, 4.5);
  TH1I* hXOmc = new TH1I("XOriginPointMC", "XOriginPointMC", 1000, -4.5, 4.5);
  TH1I* hYO = new TH1I("YOriginPoint", "YOriginPoint", 1000, -4.5, 4.5);
  TH1I* hYOmc = new TH1I("YOriginPointMC", "YOriginPointMC", 1000, -4.5, 4.5);
  TH1I* hZO = new TH1I("ZOriginPoint", "ZOriginPoint", 1000, -17.5, 17.5);
  TH1I* hZOmc = new TH1I("ZOriginPointMC", "ZOriginPointMC", 1000, -17.5, 17.5);
  TH1I* hPhiO = new TH1I("PhiOriginPoint", "PhiOriginPoint", 50, 0., 6.3);
  TH1I* hPhiOmc = new TH1I("PhiOriginPointMC", "PhiOriginPointMC", 50, 0., 6.3);
  TH1I* hRO = new TH1I("ROriginPoint", "ROriginPoint", 500, 0., 4.5);
  TH1I* hROmc = new TH1I("ROriginPointMC", "ROriginPointMC", 500, 0., 4.5);

  TH1I* hXOdiff = new TH1I("XOriginPointDiff", "XOriginPointDiff", 1000, -4.5, 4.5);
  TH1I* hYOdiff = new TH1I("YOriginPointDiff", "YOriginPointDiff", 1000, -4.5, 4.5);
  TH1I* hZOdiff = new TH1I("ZOriginPointDiff", "ZOriginPointDiff", 1000, -17.5, 17.5);
  TH1I* hPhiOdiff = new TH1I("PhiOriginPointDiff", "PhiOriginPointDiff", 50, 0., 6.3);
  TH1I* hROdiff = new TH1I("ROriginPointDiff", "ROriginPointDiff", 500, 0., 4.5);
  TH1I* hXDdiff = new TH1I("XDestinationPointDiff", "XDestinationPointDiff", 1000, -4.5, 4.5);
  TH1I* hYDdiff = new TH1I("YDestinationPointDiff", "YDestinationPointDiff", 1000, -4.5, 4.5);
  TH1I* hZDdiff = new TH1I("ZDestinationPointDiff", "ZDestinationPointDiff", 1000, -17.5, 17.5);
  TH1I* hPhiDdiff = new TH1I("PhiDestinationPointDiff", "PhiDestinationPointDiff", 50, 0., 6.3);
  TH1I* hRDdiff = new TH1I("RDestinationPointDiff", "RDestinationPointDiff", 500, 0., 4.5);

  const Line zAxis{std::array<float, 3>{ 0., 0., 0.}, std::array<float, 3>{ 0., 0., 1.}};
  
  // TCanvas* c1 = new TCanvas("Trackleting", "Trackleting", 1920, 1200);
  // c1->Divide(2,2); 
  // c1->cd(4);
  // TView *view = TView3D::CreateView(1);
  // view->SetRange(-4.5,-4.5, -17,4.5,4.5,17);
  // view->ShowAxis();
  // c1->cd(2)->DrawFrame(-4.5,-4.5,4.5,4.5);
// 
  // // Montecarlo
  // c1->cd(3);
  // TView* view_montecarlo = TView3D::CreateView(1);
  // view_montecarlo->SetRange(-4.5,-4.5, -17,4.5,4.5,17);
  // view_montecarlo->ShowAxis();
  // c1->cd(1)->DrawFrame(-4.5,-4.5,4.5,4.5);

  for (int iEvent { startevent }; iEvent < endevent; ++iEvent) {

    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);

    o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);
    o2::ITS::CA::Vertexer vertexer(event);
    o2::ITS::CA::Vertexer vertexer_montecarlo(event);
    vertexer.initialise(0.02, 0.005, 0.04, 0.8, 3);
    vertexer_montecarlo.initialise(0.02, 0.005, 0.04, 0.8, 3);
    vertexer_montecarlo.generateTracklets();
    vertexer.findTracklets();
    for ( auto& proj : vertexer_montecarlo.getZDelta() )
      deltaZProjMc->Fill(proj);
    for ( auto& proj : vertexer.getZDelta() )
      deltaZProj->Fill(proj);
    // for ( auto& proj : vertexer_montecarlo.getPhiDelta() )
    //   deltaPhiProj->Fill(proj);

    auto tracklets = vertexer.getTracklets();
    auto tracklets_montecarlo = vertexer_montecarlo.getTracklets(); 

    // TLine* line = new TLine();
    // TLine* line_montecarlo = new TLine();
    
    for ( auto& tracklet : tracklets ) {
      // Graphics
      // std::array<float, 6> points { tracklet.originPoint[0], tracklet.originPoint[1], tracklet.originPoint[2], tracklet.destinationPoint[0], tracklet.destinationPoint[1], tracklet.destinationPoint[2] };
      // TPolyLine3D* line3d = new TPolyLine3D(2, points.data());
      // c1->cd(3);
      // line3d->Draw();
      // dcaTupleZAxis->Fill(Line::getDCA(tracklet, zAxis));
      // c1->cd(1);
      // line->DrawLine(tracklet.originPoint[0], tracklet.originPoint[1], tracklet.destinationPoint[0], tracklet.destinationPoint[1]);
      // Data
      hXO->Fill(tracklet.originPoint[0]); 
      hYO->Fill(tracklet.originPoint[1]); 
      hZO->Fill(tracklet.originPoint[2]);
      hPhiO->Fill(calculatePhiCoordinate(tracklet.originPoint[0], tracklet.originPoint[1])); 
      hRO->Fill(calculateRCoordinate(tracklet.originPoint[0], tracklet.originPoint[1]));
      hXD->Fill(tracklet.destinationPoint[0]); 
      hYD->Fill(tracklet.destinationPoint[1]); 
      hZD->Fill(tracklet.destinationPoint[2]); 
      hPhiD->Fill(calculatePhiCoordinate(tracklet.destinationPoint[0], tracklet.destinationPoint[1])); 
      hRD->Fill(calculateRCoordinate(tracklet.destinationPoint[0], tracklet.destinationPoint[1]));
    }
    
    // Montecarlo
    for ( auto& tracklet : tracklets_montecarlo ) {
      // Graphics
      // std::array<float, 6> points { tracklet.originPoint[0], tracklet.originPoint[1], tracklet.originPoint[2], tracklet.destinationPoint[0], tracklet.destinationPoint[1], tracklet.destinationPoint[2] };
      // TPolyLine3D* line3d_montecarlo = new TPolyLine3D(2, points.data());
      // c1->cd(4);
      // line3d_montecarlo->Draw();
      // dcaTupleZAxis_montecarlo->Fill(Line::getDCA(tracklet, zAxis));
      // c1->cd(2);
      // line_montecarlo->DrawLine(tracklet.originPoint[0], tracklet.originPoint[1], tracklet.destinationPoint[0], tracklet.destinationPoint[1]);
      // Data
      hXOmc->Fill(tracklet.originPoint[0]); 
      hYOmc->Fill(tracklet.originPoint[1]); 
      hZOmc->Fill(tracklet.originPoint[2]); 
      hPhiOmc->Fill(calculatePhiCoordinate(tracklet.originPoint[0], tracklet.originPoint[1])); 
      hROmc->Fill(calculateRCoordinate(tracklet.originPoint[0], tracklet.originPoint[1]));
      hXDmc->Fill(tracklet.destinationPoint[0]); 
      hYDmc->Fill(tracklet.destinationPoint[1]); 
      hZDmc->Fill(tracklet.destinationPoint[2]); 
      hPhiDmc->Fill(calculatePhiCoordinate(tracklet.destinationPoint[0], tracklet.destinationPoint[1])); 
      hRDmc->Fill(calculateRCoordinate(tracklet.destinationPoint[0], tracklet.destinationPoint[1]));
    }

    // Check efficiency
    int good { 0 }, bad { 0 };
    
    for ( auto& tracklet : tracklets) {
      bool found { false };
      for( auto& trackletMC : tracklets_montecarlo) {
        if ( trackletMC.originID != trackletMC.destinID ) std::cout<<"????"<<std::endl;
        if ( tracklet.originID == trackletMC.originID && tracklet.destinID == trackletMC.destinID && !found) {
          ++good;
          found = true;
        } else {
          if ( tracklet.originID == trackletMC.originID && tracklet.destinID == trackletMC.destinID && !found )
            std::cout<<"Found already!\n";
        };
      }
      if ( !found )
        ++bad;
    }
    std::cout<<"evt: " <<iEvent<<" ntrack MC: "<<tracklets_montecarlo.size()<<"\n\tgood: "<<good<<" bad: " <<bad<< std::endl;
    std::cout<<"\tcut  eff -> ratio: "<<  std::setprecision(4) << 100 * (float)good / (float)(tracklets.size()) << "%"<<std::endl;
    std::cout<<"\treco eff -> ratio: "<<  std::setprecision(4) << 100 * (float)good / (float)(tracklets_montecarlo.size()) << "%"<<std::endl;
  } // Loop on events;
  hXOdiff->Add(hXOmc, hXO, -1);
  hYOdiff->Add(hYOmc, hYO, -1);
  hZOdiff->Add(hZOmc, hZO, -1);
  hPhiOdiff->Add(hPhiOmc, hPhiO, -1);
  hROdiff->Add(hROmc, hRO, -1);
  hXDdiff->Add(hXDmc, hXD, -1);
  hYDdiff->Add(hYDmc, hYD, -1);
  hZDdiff->Add(hZDmc, hZD, -1);
  hPhiDdiff->Add(hPhiDmc, hPhiD, -1);
  hRDdiff->Add(hRDmc, hRD, -1);

  hXOdiff->Write();
  hYOdiff->Write();
  hZOdiff->Write();
  hPhiOdiff->Write();
  hROdiff->Write();
  hXDdiff->Write();
  hYDdiff->Write();
  hZDdiff->Write();
  hPhiDdiff->Write();
  hRDdiff->Write();
  hXO->Write();
  hXOmc->Write();
  hYO->Write();
  hYOmc->Write();
  hZO->Write();
  hZOmc->Write();
  hPhiO->Write();
  hPhiOmc->Write();
  hRO->Write();
  hROmc->Write();
  hXD->Write();
  hXDmc->Write();
  hYD->Write();
  hYDmc->Write();
  hZD->Write();
  hZDmc->Write();
  hPhiD->Write();
  hPhiDmc->Write();
  hRD->Write();
  hRDmc->Write();
  dcaTupleZAxis->Write();
  dcaTupleZAxis_montecarlo->Write();
  deltaZProjMc->Write();
  deltaZProj->Write();
  outputfile->Close();
}
#endif