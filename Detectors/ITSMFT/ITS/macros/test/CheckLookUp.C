/// \file CheckDigits.C
/// \brief Simple macro to check ITSU clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TAxis.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>
#include <fstream>
#include <string>
#include "TStopwatch.h"

#include "MathUtils/Utils.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#endif

void CheckLookUp(std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim.root", std::string inputGeom = "O2geometry.root")
{
  using namespace o2::Base;
  using namespace o2::ITS;

  using o2::ITSMFT::BuildTopologyDictionary;
  using o2::ITSMFT::Cluster;
  using o2::ITSMFT::ClusterTopology;
  using o2::ITSMFT::Hit;
  using o2::ITSMFT::LookUp;
  using o2::ITSMFT::TopologyDictionary;

  LookUp finder("complete_dictionary.bin");
  TopologyDictionary dict;
  dict.ReadBinaryFile("complete_dictionary.bin");
  ofstream check_output("checkLU.txt");

  // Geometry
  o2::Base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot, o2::TransformType::L2G)); // request cached transforms

  // Hits
  TFile* file0 = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("ITSHit", &hitArray);

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  Int_t nevH = hitTree->GetEntries();   // hits are stored as one event per entry
  int ievC = 0, ievH = 0;
  int lastReadHitEv = -1;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);
    bool restart = false;
    restart = (ievC == 0) ? true : false;
    while (nc--) {
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[nc];
      int rowSpan = c.getPatternRowSpan();
      int columnSpan = c.getPatternColSpan();
      int nBytes = (rowSpan * columnSpan) >> 3;
      if (((rowSpan * columnSpan) % 8) != 0)
        nBytes++;
      unsigned char patt[Cluster::kMaxPatternBytes];
      c.getPattern(&patt[0], nBytes);
      ClusterTopology topology(rowSpan, columnSpan, patt);
      check_output << "input:" << endl
                   << endl;
      check_output << topology << endl;
      check_output << "output:" << endl
                   << endl;
      check_output << dict.GetPattern(finder.findGroupID(rowSpan, columnSpan, patt)) << endl;
      check_output << "********************************************************" << endl;
    }
  }
}
