#if !defined(__CINT__) || defined(__MAKECINT__)

#include <sstream>

#include <TStopwatch.h>
#include <TH2F.h>
#include <TCanvas.h>

#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"

#include <TTreeReader.h>
#include <TFile.h>
#include <TClonesArray.h>

#include "MFTReconstruction/Cluster.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

#include "ITSMFTSimulation/Hit.h"

#endif

void CheckClusters(Int_t nEvents = 1, Int_t nMuons = 1, std::string mcEngine = "TGeant3") {

  // macro/compare_cluster.C (TPC)

  // MFT local X (column) and Y (row)
  TH2F *hDifLocXcYr = new TH2F("hDifLocXcYr","hDifLocXcYr",100,-50.,+50.,100,-50.,+50.);
  // Global (ALCIE c.s.) X and Y
  TH2F *hDifGloXY = new TH2F("hDifGloXY","hDifGloXY",100,-50.,+50.,100,-50.,+50.);

  // Input files

  std::string treeName = "cbmsim";

  // clusters

  std::string filenameClus = "AliceO2_" + mcEngine + ".clus_" + std::to_string(nEvents) + "ev_" + std::to_string(nMuons) + "mu.root";

  // Output file name
  char fileout[100];
  sprintf(fileout, "macro.root");
  TString outFile = fileout;

  // Parameter file name
  char filepar[100];
  sprintf(filepar, "AliceO2_%s.params_%iev_%imu.root", mcEngine.c_str(), nEvents, nMuons);
  TString parFile = filepar;

  // Setup FairRoot analysis manager
  FairRunAna * fRun = new FairRunAna();
  FairFileSource *fFileSource = new FairFileSource(filenameClus);
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outFile);

  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(parFile);
  rtdb->setFirstInput(parInput1);

  // necessary to create the geometry interface for the MFT
  fRun->Init();

  // geometry 

  o2::MFT::Geometry *mftGeom = o2::MFT::Geometry::instance();
  mftGeom->build();

  o2::MFT::GeometryTGeo *mftTGeo = new o2::MFT::GeometryTGeo();
  mftTGeo->build(kFALSE);

  TFile *fileClus = TFile::Open(filenameClus.c_str());
  if (fileClus == NULL) {
    std::cout << "ERROR: Can't open file " << filenameClus << std::endl;
    return;
  }

  TTree* treeClus = (TTree*)fileClus->Get(treeName.c_str());
  if (treeClus == NULL) {
    std::cout << "ERROR: can't get tree " << treeName << std::endl;
    return;
  }

  TTreeReader readClus(treeName.c_str(), fileClus);
  TTreeReaderValue<TClonesArray> mClusters(readClus, "MFTClusters");

  // MC hits

  std::string filenameHits = "AliceO2_" + mcEngine + ".mc_" + std::to_string(nEvents) + "ev_" + std::to_string(nMuons) + "mu.root";

  TFile *fileHits = TFile::Open(filenameHits.c_str());
  if (fileHits == NULL) {
    std::cout << "ERROR: Can't open file " << filenameHits << std::endl;
    return;
  }

  TTree* treeHits = (TTree*)fileHits->Get(treeName.c_str());
  if (treeHits == NULL) {
    std::cout << "ERROR: can't get tree " << treeName << std::endl;
    return;
  }

  TTreeReader readHits(treeName.c_str(), fileHits);
  TTreeReaderValue<TClonesArray> mHits(readHits, "MFTHits");

  Double_t hitLocMFT[3], hitLocITS[3], hitGlobal[3];
  memset(hitLocMFT, 0, sizeof(Double_t) * 3);
  memset(hitLocITS, 0, sizeof(Double_t) * 3);
  memset(hitGlobal, 0, sizeof(Double_t) * 3);
  const TGeoHMatrix *matMFTtoITS = mftTGeo->getMatrixMFTtoITS();
  matMFTtoITS->Print();

  Int_t plane, half;
  // print transformation matrices
  /*
  for (Int_t i = 0; i < 920; i++) {
    plane = mftTGeo->getChipPlaneID(i);
    if (plane == 1) {
      const TGeoHMatrix* m = mftTGeo->getMatrixSensor(i);
      m->Print();
      break;
    }
  }
  */
  //return;
  
  // read hits
  while (readHits.Next()) {

    std::cout << readHits.GetCurrentEntry()+1 << " / " << readHits.GetEntries(false) << std::endl;

    int nHits = mHits->GetEntries();
    std::cout << nHits << " hits" << std::endl;
    
    // read clusters
    readClus.Next();
    int nClusters = mClusters->GetEntries();
    std::cout << nClusters << " clusters" << std::endl;
    
    // loop hits for event
    for (int i = 0; i < mHits->GetEntries(); i++) {

      o2::ITSMFT::Hit* mHit = dynamic_cast<o2::ITSMFT::Hit*>(mHits->At(i));

      //printf("Hit: %5d  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %5d  %5d \n",i,mHit->GetStartX(),mHit->GetStartY(),mHit->GetStartZ(),mHit->GetX(),mHit->GetY(),mHit->GetZ(),mHit->GetDetectorID(),mHit->GetTrackID());

      hitGlobal[0] = mHit->GetX();
      hitGlobal[1] = mHit->GetY();
      hitGlobal[2] = mHit->GetZ();

      // transform MC hit in local coordinates MFT and ITS

      const TGeoHMatrix* matSensor = mftTGeo->getMatrixSensor(mHit->GetDetectorID());
      const TGeoHMatrix* matSensorToITS = mftTGeo->getMatrixSensorToITS(mHit->GetDetectorID());
      //matSensor->Print();
      //matSensorToITS->Print();

      matSensor->MasterToLocal(hitGlobal,hitLocMFT);
      //printf("LocMFT:  %10.6f  %10.6f  %10.6f \n",hitLocMFT[0],hitLocMFT[1],hitLocMFT[2]);
      matSensorToITS->MasterToLocal(hitGlobal,hitLocITS);
      //printf("LocITS:  %10.6f  %10.6f  %10.6f \n",hitLocITS[0],hitLocITS[1],hitLocITS[2]);
      
      // loop clusters for event and for each hit
      for (int i = 0; i < mClusters->GetEntries(); i++) {

	o2::MFT::Cluster* mCluster = dynamic_cast<o2::MFT::Cluster*>(mClusters->At(i));

	//printf("ITS: %5d  %10.6f  %10.6f  %10.6f \n",i,mCluster->getX(),mCluster->getY(),mCluster->getZ());
	//printf("MFT: %5d  %10.6f  %10.6f  %10.6f \n",i,mCluster->getMFTLocalX(),mCluster->getMFTLocalY(),mCluster->getMFTLocalZ());
	//printf("Global: %5d  %10.6f  %10.6f  %10.6f  %2d  %2d  %2d \n",i,mCluster->getGlobalX(),mCluster->getGlobalY(),mCluster->getGlobalZ(),mCluster->getNx(),mCluster->getNz(),mCluster->getNPix());
	
	if (mCluster->getVolumeId() == mHit->GetDetectorID()) {

	  //printf("Global: %5d  %10.6f  %10.6f  %10.6f  %2d  %2d  %2d  %5d \n",i,mCluster->getGlobalX(),mCluster->getGlobalY(),mCluster->getGlobalZ(),mCluster->getNx(),mCluster->getNz(),mCluster->getNPix(),mCluster->getLabel(0));

	  if (mHit->GetTrackID() == mCluster->getLabel(0)) {

	    hDifLocXcYr->Fill(1.e+4*(mCluster->getMFTLocalX()-hitLocMFT[0]),
			      1.e+4*(mCluster->getMFTLocalY()-hitLocMFT[1]));
	    
	    hDifGloXY->Fill(1.e+4*(mCluster->getGlobalX()-hitGlobal[0]),
			    1.e+4*(mCluster->getGlobalY()-hitGlobal[1]));

	  } // select MC label

	} // select sensor
      
      } // end loop clusters
      
    } // end loop hits
    
  } // end loop events

  // read all clusters
  /*
  while (readClus.Next()) {
    
    std::cout << readClus.GetCurrentEntry()+1 << " / " << readClus.GetEntries(false) << std::endl;
    int nClusters = mClusters->GetEntries();
    std::cout << nClusters << " clusters" << std::endl;
    for (int i = 0; i < mClusters->GetEntries(); i++) {
      o2::MFT::Cluster* mCluster = dynamic_cast<o2::MFT::Cluster*>(mClusters->At(i));
      //printf("ITS: %5d  %10.6f  %10.6f  %10.6f \n",i,mCluster->getX(),mCluster->getY(),mCluster->getZ());
      //printf("MFT: %5d  %10.6f  %10.6f  %10.6f \n",i,mCluster->getMFTLocalX(),mCluster->getMFTLocalY(),mCluster->getMFTLocalZ());
      //printf("Global: %5d  %10.6f  %10.6f  %10.6f  %2d  %2d  %2d \n",i,mCluster->getGlobalX(),mCluster->getGlobalY(),mCluster->getGlobalZ(),mCluster->getNx(),mCluster->getNz(),mCluster->getNPix());
      
    }
    
  }
  */
  fileClus->Close();
  fileHits->Close();

  TCanvas *c1 = new TCanvas("c1","hDifXcYr",50,50,800,400);
  c1->Divide(2,1);
  c1->cd(1);
  hDifGloXY->Draw("COL2");
  c1->cd(2);
  hDifLocXcYr->Draw("COL2");

}


