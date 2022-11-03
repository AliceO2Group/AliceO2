#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TAxis.h>
#include <TStyle.h>
#include <TPaveStats.h>

#include <TStopwatch.h>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "DataFormatsFV0/Hit.h"
#include <fairlogger/Logger.h>
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsCommonDataFormats/DetID.h"

void AdjustStatBox(TH1* h, float x1ndc, float x2ndc, float y1ndc, float y2ndc)
{
  gPad->Update();
  TPaveStats* st = (TPaveStats*)h->FindObject("stats");
  if (st != NULL) {
    st->SetX1NDC(x1ndc);
    st->SetX2NDC(x2ndc);
    st->SetY1NDC(y1ndc);
    st->SetY2NDC(y2ndc);
    st->SetTextColor(h->GetLineColor());
  }
  gPad->Modified();
}

void InitHistoNames(std::vector<std::string>& vhName, std::vector<int>& vPdg)
{
  vPdg.push_back(22);  // gamma
  vPdg.push_back(11);  // e
  vPdg.push_back(211); // pi
  vPdg.push_back(0);   // other

  vhName.push_back("hElossVsDistance_Gamma");
  vhName.push_back("hElossVsDistance_E");
  vhName.push_back("hElossVsDistance_Pi");
  vhName.push_back("hElossVsDistance_Other");
  vhName.push_back("hElossVsEtot_Gamma");
  vhName.push_back("hElossVsEtot_E");
  vhName.push_back("hElossVsEtot_Pi");
  vhName.push_back("hElossVsEtot_Other");
  vhName.push_back("hElossDet");
  vhName.push_back("hEtotVsR");
  vhName.push_back("hEtotVsEloss");

  for (UInt_t ipdg = 0; ipdg < vPdg.size(); ipdg++) {
    std::stringstream ss;
    ss << vPdg.at(ipdg);
    vhName.at(ipdg) += ss.str();
    vhName.at(ipdg + 4) += ss.str();
  }
}

void readFV0Hits(std::string simPrefix = "o2sim", UInt_t rebin = 1)
{
  using namespace o2::detectors;
  std::string simFName(o2::base::DetectorNameConf::getHitsFileName(DetID::FV0, simPrefix));
  gStyle->SetOptStat("noumri");
  const int nCells = 40;

  // Init histos
  std::vector<std::string> vHistoNames;
  std::vector<int> vPdg;
  InitHistoNames(vHistoNames, vPdg);
  int el1 = 20, nEl = 2 * el1, et1 = 30, dist1 = 12, nDist = 2 * dist1;
  std::vector<TH2F*> vhElossVsDistance, vhElossVsEtot;
  for (UInt_t ih = 0; ih < 4; ih++) {
    vhElossVsDistance.push_back(new TH2F(vHistoNames.at(ih).c_str(), "", nEl, 0, el1, nDist, 0, dist1));
  }
  for (UInt_t ih = 4; ih < 8; ih++) {
    vhElossVsEtot.push_back(new TH2F(vHistoNames.at(ih).c_str(), "", nEl, 0, el1, 2000, 0, 20000));
  }
  TH2F* hElossDet = new TH2F(vHistoNames.at(8).c_str(), "", nEl, 0, el1, nCells, 0, nCells);
  TH2F* hEtotVsR = new TH2F(vHistoNames.at(9).c_str(), "", 30000, 0, 300, 80, 0, 80);
  TH2F* hEtotVsEloss = new TH2F(vHistoNames.at(10).c_str(), "", 30000, 0, 300, nEl, 0, el1);

  // Setup histo properties
  hElossDet->SetXTitle("Energy loss [MeV]");
  hElossDet->SetYTitle("Detector ID");
  hElossDet->SetZTitle("Counts");
  hEtotVsR->SetXTitle("Total energy at entrance [MeV]");
  hEtotVsR->SetYTitle("R [cm]");
  hEtotVsR->SetZTitle("Counts");
  hEtotVsEloss->SetXTitle("Total energy at entrance [MeV]");
  hEtotVsEloss->SetYTitle("Energy loss [MeV]");
  hEtotVsEloss->SetZTitle("Counts");
  for (UInt_t ih = 0; ih < vhElossVsDistance.size(); ih++) {
    TH2F* h = vhElossVsDistance.at(ih);
    std::stringstream ss;
    std::string hname = h->GetName();
    ss << "Energy loss [MeV] of " << hname.substr(hname.find_last_of('_') + 1);
    h->SetXTitle(ss.str().c_str());
    h->SetYTitle("Travel distance R [cm]");
    h->SetZTitle("Counts");
  }
  for (UInt_t ih = 0; ih < vhElossVsEtot.size(); ih++) {
    TH2F* h = vhElossVsEtot.at(ih);
    std::stringstream ss;
    std::string hname = h->GetName();
    ss << "Energy loss [MeV] in " << hname.substr(hname.find_last_of('_') + 1);
    h->SetXTitle(ss.str().c_str());
    h->SetYTitle("Total energy at entrance [MeV]");
    h->SetZTitle("Counts");
  }
  const float rmargin = 0.12, lmargin = 0.13, tmargin = 0.02, bmargin = 0.15;
  const float statX1 = 1. - rmargin, statX2 = statX1 - 0.18;
  const float statH = 0.3, statY1 = 1. - tmargin, statY2 = statY1 - statH;
  float fontsize = 0.06;
  std::vector<TH1*> vh;
  vh.push_back(hElossDet);
  vh.push_back(hEtotVsR);
  vh.push_back(hEtotVsEloss);
  vh.insert(vh.end(), vhElossVsDistance.begin(), vhElossVsDistance.end());
  vh.insert(vh.end(), vhElossVsEtot.begin(), vhElossVsEtot.end());
  for (UInt_t ih = 0; ih < vh.size(); ih++) {
    vh[ih]->SetDirectory(0);
    vh[ih]->GetXaxis()->SetTitleSize(fontsize);
    vh[ih]->GetYaxis()->SetTitleSize(fontsize);
    vh[ih]->GetZaxis()->SetTitleSize(fontsize);
    vh[ih]->GetXaxis()->SetLabelSize(fontsize * 0.8);
    vh[ih]->GetYaxis()->SetLabelSize(fontsize * 0.8);
    vh[ih]->GetZaxis()->SetLabelSize(fontsize * 0.8);
    vh[ih]->GetXaxis()->SetTitleOffset(1.0);
    vh[ih]->GetYaxis()->SetTitleOffset(1.0);
    vh[ih]->SetLineWidth(2);
  }

  std::unique_ptr<TFile> simFile(TFile::Open(simFName.c_str()));
  if (!simFile->IsOpen() || simFile->IsZombie()) {
    std::cout << "Failed to open input sim file " << simFName << std::endl;
    return;
  }

  TTree* simTree = (TTree*)simFile->Get("o2sim");
  if (!simTree) {
    std::cout << "Failed to get sim tree" << std::endl;
    return;
  }

  std::vector<o2::fv0::Hit>* hits = nullptr;
  simTree->SetBranchAddress("FV0Hit", &hits);

  UInt_t nEntries = simTree->GetEntries();
  for (UInt_t ient = 0; ient < nEntries; ient++) {
    simTree->GetEntry(ient);
    for (UInt_t ihit = 0; ihit < hits->size(); ihit++) {
      o2::fv0::Hit* hit = &(hits->at(ihit));
      hElossDet->Fill(hit->GetEnergyLoss() * 1e3, hit->GetDetectorID());
      float r = TMath::Sqrt(TMath::Power(hit->GetX(), 2) + TMath::Power(hit->GetY(), 2));
      hEtotVsR->Fill(hit->GetTotalEnergyAtEntrance() * 1e3, r);
      hEtotVsEloss->Fill(hit->GetTotalEnergyAtEntrance() * 1e3, hit->GetEnergyLoss() * 1e3);
      float distance = TMath::Sqrt(TMath::Power(hit->GetX() - hit->GetPosStart().X(), 2) +
                                   TMath::Power(hit->GetY() - hit->GetPosStart().Y(), 2) +
                                   TMath::Power(hit->GetZ() - hit->GetPosStart().Z(), 2));
      int apdg = abs(hit->GetParticlePdg());
      bool isFilled = false;
      for (UInt_t ipdg = 0; ipdg < vPdg.size() - 1; ipdg++) {
        if (apdg == vPdg.at(ipdg)) {
          vhElossVsDistance.at(ipdg)->Fill(hit->GetEnergyLoss() * 1e3, distance);
          vhElossVsEtot.at(ipdg)->Fill(hit->GetEnergyLoss() * 1e3, hit->GetTotalEnergyAtEntrance() * 1e3);
          isFilled = true;
        }
      }
      if (!isFilled) { // put all other particles to the last element
        vhElossVsDistance.at(vhElossVsDistance.size() - 1)->Fill(hit->GetEnergyLoss() * 1e3, distance);
        vhElossVsEtot.at(vhElossVsEtot.size() - 1)->Fill(hit->GetEnergyLoss() * 1e3, hit->GetTotalEnergyAtEntrance() * 1e3);
      }
    }
  }

  // Draw histos
  TCanvas* chit = new TCanvas("fv0hit", "fv0hit", 1800, 550);
  chit->Divide(3, 1);
  chit->cd(1);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hElossDet->Rebin2D(rebin, 1);
  hElossDet->Draw("colz");
  AdjustStatBox(hElossDet, statX1, statX2, statY1, statY2);
  chit->cd(2);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hEtotVsR->Rebin2D(rebin, rebin);
  hEtotVsR->Draw("colz");
  AdjustStatBox(hEtotVsR, statX1, statX2, statY1, statY2);
  chit->cd(3);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hEtotVsEloss->Rebin2D(rebin, rebin);
  hEtotVsEloss->Draw("colz");
  AdjustStatBox(hEtotVsEloss, statX1, statX2, statY1, statY2);

  TCanvas* celossdist = new TCanvas("fv0hit-elossDist", "fv0hit-elossDist", 1200, 930);
  celossdist->Divide(2, 2);
  for (UInt_t ih = 0; ih < vhElossVsDistance.size(); ih++) {
    celossdist->cd(ih + 1);
    gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
    gPad->SetLogz();
    vhElossVsDistance.at(ih)->Rebin2D(rebin, rebin);
    vhElossVsDistance.at(ih)->Draw("colz");
    AdjustStatBox(vhElossVsDistance.at(ih), statX1, statX2, statY1, statY2);
  }

  TCanvas* celosstot = new TCanvas("fv0hit-elossEtot", "fv0hit-elossEtot", 1200, 930);
  celosstot->Divide(2, 2);
  for (UInt_t ih = 0; ih < vhElossVsEtot.size(); ih++) {
    celosstot->cd(ih + 1);
    gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
    gPad->SetLogz();
    vhElossVsEtot.at(ih)->Rebin2D(rebin, rebin);
    vhElossVsEtot.at(ih)->Draw("colz");
    AdjustStatBox(vhElossVsEtot.at(ih), statX1, statX2, statY1, statY2);
  }

  // Save histos
  TFile* fout = new TFile("fv0hit-rawhistos.root", "RECREATE");
  for (UInt_t ih = 0; ih < vh.size(); ih++) {
    vh.at(ih)->Write();
  }
  fout->Close();
}

// Root files generated in the previous stage (readFV0Hits()) are used as inputs here
int compareFV0Hits(std::string simFName1 = "fv0hit-rawhistos.root", std::string simFName2 = "", std::string simFName3 = "")
{
  gStyle->SetOptStat("noumri");

  // Open files
  std::vector<TFile*> vf;
  TFile* f1 = new TFile(simFName1.c_str(), "OPEN");
  if (f1->IsOpen()) {
    vf.push_back(f1);
  } else {
    std::cout << "<E> Problem reading the first file: " << simFName1 << std::endl;
    return -1;
  }
  if (simFName2.size() > 0) {
    TFile* f2 = new TFile(simFName2.c_str(), "OPEN");
    if (f2->IsOpen()) {
      vf.push_back(f2);
    }
  }
  if (simFName3.size() > 0) {
    TFile* f3 = new TFile(simFName3.c_str(), "OPEN");
    if (f3->IsOpen()) {
      vf.push_back(f3);
    }
  }
  std::cout << "  <I> Number of accessible files: " << vf.size() << std::endl;

  // Open histos
  std::vector<std::string> vHistoNames;
  std::vector<int> vPdg;
  InitHistoNames(vHistoNames, vPdg);
  std::vector<TH1*> vh;
  UInt_t nFiles = vf.size();
  UInt_t nHistos = vHistoNames.size();
  for (UInt_t ifile = 0; ifile < nFiles; ifile++) {
    for (UInt_t ih = 0; ih < nHistos; ih++) {
      TH1* h = (TH1*)vf.at(ifile)->Get(vHistoNames.at(ih).c_str());
      if (h == nullptr) {
        std::cout << "<E> Problem reading histo: " << vHistoNames.at(ih);
        std::cout << " from file: " << ifile << std::endl;
        return -2;
      }
      vh.push_back(h);
    }
  }
  std::cout << "  <I> Read: " << vh.size() << " histos" << std::endl;

  const float rmargin = 0.12, lmargin = 0.13, tmargin = 0.02, bmargin = 0.15;
  const float statX1 = 1. - rmargin, statX2 = statX1 - 0.18;
  const float statH = 0.25, statY1 = 1. - tmargin, statY2 = statY1 - statH;

  // Draw side-by-side comparison of TH2's
  const float zoomLevelRms = 3;
  for (UInt_t ih = 8; ih < nHistos; ih++) {
    std::stringstream ss;
    ss << "fv0digi-cmp" << ih;
    TCanvas* c = new TCanvas(ss.str().c_str(), ss.str().c_str(), 1800, 500);
    c->Divide(3, 1);
    for (UInt_t ifile = 0; ifile < nFiles; ifile++) {
      c->cd(ifile + 1);
      gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
      gPad->SetLogz();
      TH1* h = vh.at(ifile * nHistos + ih);
      h->Draw("colz");
      AdjustStatBox(h, statX1, statX2, statY1, statY2);
    }
  }

  // Draw the comparison of TH1's
  Color_t col[3] = {kBlack, kRed, kBlue};
  TCanvas* c = new TCanvas("fv0digi-cmp-th1", "fv0digi-cmp-th1", 1800, 800);
  c->Divide(2, 2);
  for (UInt_t ifile = 0; ifile < nFiles; ifile++) {
    for (UInt_t ipdg = 0; ipdg < vPdg.size(); ipdg++) {
      TH2F* h2 = (TH2F*)vh.at(ifile * nHistos + ipdg);
      h2->SetLineColor(col[ifile]);
      h2->SetLineWidth(2);
      std::stringstream ss;
      ss << "p" << ifile;
      TH1D* he = h2->ProjectionX((ss.str() + "e_" + h2->GetName()).c_str());
      he->SetLineWidth(2);
      c->cd(ipdg + 1);
      gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
      gPad->SetLogy();
      he->Draw((ifile == 0) ? "" : "sames");
      AdjustStatBox(he, statX1, statX2, statY1 - statH * ifile, statY2 - statH * ifile);
    }
  }
  return 0;
}

#endif
