#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>
#include <iostream>

#include "TROOT.h"
#include <TStopwatch.h>
#include "TCanvas.h"
#include "TH2.h"
//#include "DataFormatsParameters/GRPObject.h"
#include "FairFileSource.h"
#include <fairlogger/Logger.h>
#include "FairRunAna.h"
//#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "DataFormatsCPV/Cluster.h"
#include "CPVBase/Geometry.h"
#endif

void plot_clu_cpv(std::string inputfile = "cpvclusters.root", int ifirst = 0, int ilast = -1)
{
  // macros to plot CPV clusters

  // Clusters
  TFile* file0 = TFile::Open(inputfile.data());
  std::cout << " Open clusters file " << inputfile << std::endl;
  TTree* cluTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::cpv::Cluster>* mClustersArray = nullptr;
  cluTree->SetBranchAddress("CPVCluster", &mClustersArray);

  if (!mClustersArray) {
    std::cout << "CPV clusters not found in the file. Exiting ..." << std::endl;
    return;
  }

  TH1F* hClusterTotEnergy[5];
  TH1F* hClusterMaxEnergy[5];
  TH1F* hClusterSize[5];
  TH1F* hClusterSizeX[5];
  TH1F* hClusterSizeZ[5];

  TH2D* vMod[5][1000] = {0};
  int primLabels[5][1000];
  for (int mod = 2; mod < 5; mod++) {
    hClusterTotEnergy[mod] = new TH1F(Form("hClusterTotEnergy%d", mod),
                                      Form("Cluster Total Energy mod %d", mod), 10000, 0, 10000);
    hClusterMaxEnergy[mod] = new TH1F(Form("hClusterMaxEnergy%d", mod),
                                      Form("Cluster Max   Energy mod %d", mod), 10000, 0, 10000);
    hClusterSize[mod] = new TH1F(Form("hClusterSize%d", mod),
                                 Form("Cluster Size mod %d", mod), 100, 0, 100);
    hClusterSizeX[mod] = new TH1F(Form("hClusterSizeX%d", mod),
                                  Form("Cluster SizeX mod %d", mod), 100, 0, 100);
    hClusterSizeZ[mod] = new TH1F(Form("hClusterSizeZ%d", mod),
                                  Form("Cluster SizeZ mod %d", mod), 100, 0, 100);

    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -1;
  }

  int nEntries = cluTree->GetEntriesFast();
  if (ilast < 0)
    ilast = nEntries;
  if (ilast > nEntries)
    ilast = nEntries;

  for (int ievent = ifirst; ievent < ilast; ievent++) {
    cluTree->GetEvent(ievent);

    std::vector<o2::cpv::Cluster>::const_iterator it;
    std::cout << "I start cluster cycling (record #" << ievent << " in o2sim tree)" << std::endl;

    for (it = mClustersArray->begin(); it != mClustersArray->end(); it++) {
      float en = (*it).getEnergy();
      float posX, posZ;
      (*it).getLocalPosition(posX, posZ);
      int cluSize = (*it).getMultiplicity();
      int mod = (*it).getModule();
      if (!vMod[mod][0]) {
        gROOT->cd();
        vMod[mod][0] =
          new TH2D(Form("hMod%d_prim%d", mod, 0), Form("hMod%d_prim%d", mod, 0),
                   100, -100., 100., 100, -100., 100.);
      }
      vMod[mod][0]->Fill(posX, posZ, en);
      hClusterTotEnergy[mod]->Fill(en);
      hClusterSize[mod]->Fill(cluSize);
    }

    std::cout << "I finish cycling clusters" << std::endl;
  }
  TCanvas* c[5];
  TH2D* box = new TH2D("box", "CPV module", 100, -100., 100., 100, -100., 100.);
  TCanvas* cTotEn = new TCanvas();
  cTotEn->Divide(3, 1);
  TCanvas* cSize = new TCanvas();
  cSize->Divide(3, 1);

  for (int mod = 2; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("ClusterInMod%d", mod), Form("CPV clusters in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
    box->Draw();
    int j = 0;
    while (vMod[mod][j]) {
      vMod[mod][j]->SetLineColor(j + 1);
      if (j == 0)
        vMod[mod][j]->Draw("box");
      else
        vMod[mod][j]->Draw("boxsame");
      j++;
    }
    cTotEn->cd(mod - 1);
    hClusterTotEnergy[mod]->Draw();
    cSize->cd(mod - 1);
    hClusterSize[mod]->Draw();
  }
}
