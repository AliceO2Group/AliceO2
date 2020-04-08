/// \file CheckTopologies.C
/// Macros to test the generation of a dictionary of topologies. Three dictionaries are generated: one with signal-cluster only, one with noise-clusters only and one with all the clusters.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TAxis.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>
#include <fstream>
#include <string>

#include "MathUtils/Utils.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#endif

void CheckTopologies(std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim_HitsITS.root", std::string inputGeom = "")
{
  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::BuildTopologyDictionary;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::Hit;
  std::ofstream output_check("check_topologies.txt");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Hits
  TFile* fileH = nullptr;
  TTree* hitTree = nullptr;
  std::vector<Hit>* hitArray = nullptr;

  if (!hitfile.empty() && !gSystem->AccessPathName(hitfile.c_str())) {
    fileH = TFile::Open(hitfile.data());
    hitTree = (TTree*)fileH->Get("o2sim");
    hitTree->SetBranchAddress("ITSHit", &hitArray);
  }

  // Clusters
  TFile* FileCl = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)FileCl->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  if (hitTree && clusTree->GetBranch("ITSClusterMCTruth")) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
  }
  clusTree->GetEntry(0);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  Int_t nevH = 0;                       // hits are stored as one event per entry
  if (hitTree) {
    nevH = hitTree->GetEntries();
  }
  int ievC = 0, ievH = 0;
  int lastReadHitEv = -1;

  // Topologies dictionaries: 1) all clusters 2) signal clusters only 3) noise clusters only
  BuildTopologyDictionary completeDictionary;
  BuildTopologyDictionary signalDictionary;
  BuildTopologyDictionary noiseDictionary;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);

    auto pattIdx = patternsPtr->cbegin();
    for (int i = 0; i < nc; i++) {
      // cluster is in tracking coordinates always
      CompClusterExt& c = (*clusArr)[i];
      Int_t chipID = c.getSensorID();

      o2::itsmft::ClusterPattern pattern(pattIdx);
      ClusterTopology topology(pattern);
      output_check << "iEv: " << ievC << " / " << nevCl << " iCl: " << i << " / " << nc << std::endl;
      output_check << topology << std::endl;

      const auto locC = o2::itsmft::TopologyDictionary::getClusterCoordinates(c, pattern);
      float dx = 0, dz = 0;
      if (clusLabArr) {
        auto lab = (clusLabArr->getLabels(i))[0];
        int trID = lab.getTrackID();
        int ievH = lab.getEventID();
        Point3D<float> locH, locHsta;
        if (lab.isValid()) { // is this cluster from hit or noise ?
          Hit* p = nullptr;
          if (lastReadHitEv != ievH) {
            hitTree->GetEvent(ievH);
            lastReadHitEv = ievH;
          }
          for (auto& ptmp : *hitArray) {
            if (ptmp.GetDetectorID() != chipID)
              continue;
            if (ptmp.GetTrackID() != trID)
              continue;
            p = &ptmp;
            break;
          }
          if (!p) {
            printf("did not find hit (scanned HitEvs %d %d) for cluster of tr%d on chip %d\n", ievH, nevH, trID, chipID);
            locH.SetXYZ(0.f, 0.f, 0.f);
          } else {
            // mean local position of the hit
            locH = gman->getMatrixL2G(chipID) ^ (p->GetPos()); // inverse conversion from global to local
            locHsta = gman->getMatrixL2G(chipID) ^ (p->GetPosStart());
            locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
            // std::cout << "chip "<< p->GetDetectorID() << "  PposGlo " << p->GetPos() << std::endl;
            // std::cout << "chip "<< c->getSensorID() << "  PposLoc " << locH << std::endl;
            dx = locH.X() - locC.X();
            dz = locH.Z() - locC.Z();
          }
          signalDictionary.accountTopology(topology, dx, dz);
        } else {
          noiseDictionary.accountTopology(topology, dx, dz);
        }
      }
      completeDictionary.accountTopology(topology, dx, dz);
    }
  }

  auto dID = o2::detectors::DetID::ITS;

  completeDictionary.setThreshold(0.0001);
  completeDictionary.groupRareTopologies();
  completeDictionary.printDictionaryBinary(o2::base::NameConf::getDictionaryFileName(dID, "", ".bin"));
  completeDictionary.printDictionary(o2::base::NameConf::getDictionaryFileName(dID, "", ".txt"));
  completeDictionary.saveDictionaryRoot(o2::base::NameConf::getDictionaryFileName(dID, "", ".root"));

  TFile histogramOutput("histograms.root", "recreate");
  TCanvas* cComplete = new TCanvas("cComplete", "Distribution of all the topologies");
  cComplete->cd();
  cComplete->SetLogy();
  TH1F* hComplete = nullptr;
  o2::itsmft::TopologyDictionary::getTopologyDistribution(completeDictionary.getDictionary(), hComplete, "hComplete");
  hComplete->SetDirectory(0);
  hComplete->Draw("hist");
  hComplete->Write();
  cComplete->Write();

  TCanvas* cNoise = nullptr;
  TCanvas* cSignal = nullptr;
  TH1F* hNoise = nullptr;
  TH1F* hSignal = nullptr;

  if (clusLabArr) {
    noiseDictionary.setThreshold(0.0001);
    noiseDictionary.groupRareTopologies();
    noiseDictionary.printDictionaryBinary(o2::base::NameConf::getDictionaryFileName(dID, "noise", ".bin"));
    noiseDictionary.printDictionary(o2::base::NameConf::getDictionaryFileName(dID, "noise", ".txt"));
    noiseDictionary.saveDictionaryRoot(o2::base::NameConf::getDictionaryFileName(dID, "noise", ".root"));
    signalDictionary.setThreshold(0.0001);
    signalDictionary.groupRareTopologies();
    signalDictionary.printDictionaryBinary(o2::base::NameConf::getDictionaryFileName(dID, "signal", ".bin"));
    signalDictionary.printDictionary(o2::base::NameConf::getDictionaryFileName(dID, "signal", ".txt"));
    signalDictionary.saveDictionaryRoot(o2::base::NameConf::getDictionaryFileName(dID, "signal", ".root"));

    cNoise = new TCanvas("cNoise", "Distribution of noise topologies");
    cNoise->cd();
    cNoise->SetLogy();
    o2::itsmft::TopologyDictionary::getTopologyDistribution(noiseDictionary.getDictionary(), hNoise, "hNoise");
    hNoise->SetDirectory(0);
    hNoise->Draw("hist");
    hNoise->Write();
    cNoise->Write();

    cSignal = new TCanvas("cSignal", "cSignal");
    cSignal->cd();
    cSignal->SetLogy();
    o2::itsmft::TopologyDictionary::getTopologyDistribution(signalDictionary.getDictionary(), hSignal, "hSignal");
    hSignal->SetDirectory(0);
    hSignal->Draw("hist");
    hSignal->Write();
    cSignal->Write();
  }
}
