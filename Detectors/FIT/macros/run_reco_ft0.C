#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Rtypes.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TGeoManager.h>
#include <TFile.h>
#include <TTree.h>
#include <iostream>

#include "FairLogger.h"

#include "FT0Reconstruction/CollisionTimeRecoTask.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFT0/Digit.h"
#endif

void run_reco_ft0(std::string inpudDig = "ft0digits.root",
                  std::string outName = "o2reco_ft0.root",
                  std::string inputGRP = "o2sim_grp.root")
{

  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");

  // Setup timer
  TStopwatch timer;

  TFile* fdig = TFile::Open("ft0digits.root");
  std::cout << " Open digits file " << std::endl;
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  digTree->Print();
  std::vector<o2::ft0::Digit>* digits = new std::vector<o2::ft0::Digit>;
  digTree->SetBranchAddress("FT0Digit", &digits);
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  std::cout << "Found " << nevD << " events with digits " << std::endl;

  std::vector<o2::ft0::RecPoints>* recPointsP = new std::vector<o2::ft0::RecPoints>;

  TFile outFile(outName.c_str(), "recreate");
  TTree outTree("o2sim", "FT0RecPoints");
  outTree.Branch("FT0Cluster", &recPointsP);

  o2::ft0::CollisionTimeRecoTask recoFIT;
  timer.Start();
  for (int iEv = 0; iEv < nevD; iEv++) {
    digTree->GetEntry(iEv);
    recPointsP->resize(digits->size());
    for (size_t collID = 0; collID < digits->size(); ++collID)
      recoFIT.Process((*digits)[collID], (*recPointsP)[collID]);
    outTree.Fill();
  }

  outTree.Print();
  outFile.ls();
  timer.Stop();
  outTree.Write();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage = ctime / rtime;
  std::cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  std::cout << cpuUsage;
  std::cout << "</DartMeasurement>" << std::endl;
  std::cout << "Macro finished succesfully." << std::endl;

  std::cout << "Real time " << rtime << " s, CPU time " << ctime
            << "s" << std::endl
            << std::endl;
}
