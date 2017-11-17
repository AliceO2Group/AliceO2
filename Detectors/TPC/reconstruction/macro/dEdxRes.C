// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TPCReconstruction/Cluster.h"
#include "TPCBase/Mapper.h"
#include "TPCReconstruction/TrackTPC.h"
#include "TPCBase/CRU.h"

#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TSystem.h"
#include "TString.h"
#include "TProfile.h"
#include "TPaveText.h"
#include "TStyle.h"
#endif

using namespace o2::TPC;

struct EventHeader
{
  int run;
  float cherenkovValue;

};

void GetBinMinMax(const TH1 *hist, const float frac, int &bin1, int &bin2)
{
  const int binMax=hist->GetMaximumBin();
  const float contMax=hist->GetBinContent(binMax);
  bin1=binMax;
  bin2=binMax;
  while ( (bin1--)>binMax/3. ) if (hist->GetBinContent(bin1)<frac*contMax) break;
  while ( (bin2++)<binMax*3. ) if (hist->GetBinContent(bin2)<frac*contMax) break;
}

int dEdxRes(TString trackfile, float trLow = 0., float trHigh = .7)
{

  gStyle->SetOptStat(0);

  float electronmeanTot, electronsigmaTot, pionmeanTot, pionsigmaTot, electronmeanerr, electronsigmaerr, pionmeanerr, pionsigmaerr, electronmeanMax,
      electronsigmaMax, pionmeanMax, pionsigmaMax, electronmeanerrMax, electronsigmaerrMax, pionmeanerrMax, pionsigmaerrMax, pionres, pionresMax, electronres, electronresMax, separationpower, separationpowerMax;
  int runNr, CherenkovValue;

  electronmeanTot = electronsigmaTot = pionmeanTot = pionsigmaTot = electronmeanerr = electronsigmaerr = pionmeanerr = pionsigmaerr = electronmeanMax =
      electronsigmaMax = pionmeanMax = pionsigmaMax = electronmeanerrMax = electronsigmaerrMax = pionmeanerrMax = pionsigmaerrMax = pionres =
      pionresMax = electronres = electronresMax = separationpower = separationpowerMax = 0.;

  runNr = CherenkovValue = 0;

  TFile *TreeFile = TFile::Open(trackfile.Data());
  TTree *tree = (TTree*)gDirectory->Get("events");

  /// initialize histograms
  TH1F *hdEdxEleTot        = new TH1F("hdEdxEleTot", "; d#it{E}/d#it{x} Q_{tot} (a.u.); # counts", 300, 0, 600);
  TH1F *hdEdxPionTot       = new TH1F("hdEdxTot", "; d#it{E}/d#it{x} Q_{tot} (a.u.); # counts", 300, 0, 600);

  TH1F *hdEdxEleMax        = new TH1F("hdEdxEleMax", "; d#it{E}/d#it{x} Q_{max} (a.u.); # counts", 60, 0, 120);
  TH1F *hdEdxPionMax       = new TH1F("hdEdxMax", "; d#it{E}/d#it{x} Q_{max} (a.u.); # counts", 60, 0, 120);

  std::vector<TrackTPC> *vecEvent = nullptr;
  EventHeader Header;
  tree->SetBranchAddress("Tracks", &vecEvent);
  tree->SetBranchAddress("header", &Header);


  /// loop over events and apply cuts on number of tracks per event (1), number of clusters per track (nclCut) and separate pions from electrons with cherenkov value
  int nclCut = 40;
  int CherCutLow = 0.01;
  int CherCutHigh = 0.01;

  for (int iEv=0; iEv<tree->GetEntriesFast(); ++iEv){
    tree->GetEntry(iEv);

    runNr = Header.run;
    CherenkovValue = Header.cherenkovValue;

    int nTracks = vecEvent->size();
    //if (nTracks != 1) continue;

    for (auto& trackObject : *vecEvent){
      float dEdxTot = trackObject.getTruncatedMean(trLow,trHigh,1);
      float dEdxMax = trackObject.getTruncatedMean(trLow,trHigh,0);

      std::vector<Cluster> clCont;
      trackObject.getClusterVector(clCont);

      int ncl = clCont.size();
      //if (ncl < nclCut) continue;
      //if (CherenkovValue >= CherCutLow && CherenkovValue <= CherCutHigh) continue;

      if (CherenkovValue < CherCutLow){
        hdEdxPionTot->Fill(dEdxTot);
        hdEdxPionMax->Fill(dEdxMax);
      }
      if (CherenkovValue > CherCutHigh){
        hdEdxEleTot->Fill(dEdxTot);
        hdEdxEleMax->Fill(dEdxMax);
      }
    }
  }

  /// calculate and plot dE/dx for Qtot
  TCanvas *dEdxQ = new TCanvas();
  TF1 *pionfit = new TF1("pionfit","gaus",hdEdxPionTot->GetXaxis()->GetXmin(),hdEdxPionTot->GetXaxis()->GetXmax());
  TF1 *electronfit = new TF1("electronfit","gaus",hdEdxEleTot->GetXaxis()->GetXmin(),hdEdxEleTot->GetXaxis()->GetXmax());
  electronfit->SetLineColor(kRed);
  hdEdxEleTot->SetLineColor(kRed);
  pionfit->SetLineColor(kBlue);
  hdEdxPionTot->SetLineColor(kBlue);

  const float frac=0.2;
  int bin1=0,bin2=0;

  GetBinMinMax(hdEdxPionTot,frac,bin1,bin2);
  hdEdxPionTot->Fit(pionfit,"","",hdEdxPionTot->GetXaxis()->GetBinLowEdge(bin1),hdEdxPionTot->GetXaxis()->GetBinUpEdge(bin2));
  GetBinMinMax(hdEdxEleTot,frac,bin1,bin2);
  hdEdxEleTot->Fit(electronfit,"","",hdEdxEleTot->GetXaxis()->GetBinLowEdge(bin1),hdEdxEleTot->GetXaxis()->GetBinUpEdge(bin2));

  hdEdxPionTot->Draw();
  hdEdxEleTot->Draw("same");
  pionmeanTot = pionfit->GetParameter(1);
  pionsigmaTot = pionfit->GetParameter(2);
  electronmeanTot = electronfit->GetParameter(1);
  electronsigmaTot = electronfit->GetParameter(2);

  pionres = pionsigmaTot/pionmeanTot;
  electronres = electronsigmaTot/electronmeanTot;
  pionsigmaerr = pionfit->GetParError(2);
  pionmeanerr = pionfit->GetParError(1);
  electronsigmaerr = electronfit->GetParError(2);
  electronmeanerr = electronfit->GetParError(1);
  separationpower = 2*(electronmeanTot-pionmeanTot)/(pionsigmaTot+electronsigmaTot);

  TPaveText *pave1=new TPaveText(0.6,.7,.9,.9,"NDC");
  pave1->SetBorderSize(1);
  pave1->SetFillColor(10);
  pave1->AddText(Form("e: %.2f #pm %.2f (%.2f%%)",electronmeanTot,electronsigmaTot, electronsigmaTot/electronmeanTot*100));
  pave1->AddText(Form("#pi: %.2f #pm %.2f (%.2f%%)",pionmeanTot,pionsigmaTot,pionsigmaTot/pionmeanTot*100));
  pave1->AddText(Form("Separation: %.2f#sigma", TMath::Abs(electronmeanTot-pionmeanTot)/((electronsigmaTot+pionsigmaTot)/2.)));
  pave1->Draw("same");

  //dEdxQ->Print(Form("dEdxResQtot_%i.png", runNr));

/// calculate and plot dE/dx for Qmax

  TCanvas *dEdxQmax = new TCanvas();
  TF1 *pionfitMax = new TF1("pionfitMax","gaus",hdEdxPionMax->GetXaxis()->GetXmin(),hdEdxPionMax->GetXaxis()->GetXmax());
  TF1 *electronfitMax = new TF1("electronfitMax","gaus",hdEdxEleMax->GetXaxis()->GetXmin(),hdEdxEleMax->GetXaxis()->GetXmax());
  electronfitMax->SetLineColor(kRed);
  hdEdxEleMax->SetLineColor(kRed);
  pionfitMax->SetLineColor(kBlue);
  hdEdxPionMax->SetLineColor(kBlue);

  GetBinMinMax(hdEdxPionMax,frac,bin1,bin2);
  hdEdxPionMax->Fit(pionfitMax,"","",hdEdxPionMax->GetXaxis()->GetBinLowEdge(bin1),hdEdxPionMax->GetXaxis()->GetBinUpEdge(bin2));
  GetBinMinMax(hdEdxEleMax,frac,bin1,bin2);
  hdEdxEleMax->Fit(electronfitMax,"","",hdEdxEleMax->GetXaxis()->GetBinLowEdge(bin1),hdEdxEleMax->GetXaxis()->GetBinUpEdge(bin2));

  hdEdxPionMax->Draw();
  hdEdxEleMax->Draw("same");
  pionmeanMax = pionfitMax->GetParameter(1);
  pionsigmaMax = pionfitMax->GetParameter(2);
  electronmeanMax = electronfitMax->GetParameter(1);
  electronsigmaMax = electronfitMax->GetParameter(2);

  pionresMax = pionsigmaMax/pionmeanMax;
  electronresMax = electronsigmaMax/electronmeanMax;
  pionsigmaerrMax = pionfitMax->GetParError(2);
  pionmeanerrMax = pionfitMax->GetParError(1);
  electronsigmaerrMax = electronfitMax->GetParError(2);
  electronmeanerrMax = electronfitMax->GetParError(1);

  separationpowerMax = 2*(electronmeanMax-pionmeanMax)/(pionsigmaMax+electronsigmaMax);

  TPaveText *pave2=new TPaveText(0.6,.7,.9,.9,"NDC");
  pave2->SetBorderSize(1);
  pave2->SetFillColor(10);
  pave2->AddText(Form("e: %.2f #pm %.2f (%.2f%%)",electronmeanMax,electronsigmaMax, electronsigmaMax/electronmeanMax*100));
  pave2->AddText(Form("#pi: %.2f #pm %.2f (%.2f%%)",pionmeanMax,pionsigmaMax,pionsigmaMax/pionmeanMax*100));
  pave2->AddText(Form("Separation: %.2f#sigma", TMath::Abs(electronmeanMax-pionmeanMax)/((electronsigmaMax+pionsigmaMax)/2.)));
  pave2->Draw("same");

  //dEdxQmax->Print(Form("dEdxResQmax_%i.png", runNr));


return 0;
}
