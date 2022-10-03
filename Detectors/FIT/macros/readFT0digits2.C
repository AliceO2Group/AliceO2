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
#include <TLatex.h>
#include <TColor.h>

#include <TStopwatch.h>
#include <memory>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsFT0/MCLabel.h"
#include <fairlogger/Logger.h>

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

void InitHistoNames(std::vector<std::string>& vhName)
{
  vhName.push_back("hTimeCharge");
  vhName.push_back("hTimeCh");
  vhName.push_back("hChargeCh");
  vhName.push_back("hNchannels");
}

const int nChannels = 220;

void readFT0Digits(std::string digiFName = "ft0digits.root", bool printAllToTerminal = false, UInt_t rebin = 1)
{
  gStyle->SetOptStat("noumri");
  // Settings for drawing, used with SetRangeUser()
  const float tMin = -30, tMax = 70, chargeMin = 0, chargeMax = 800;

  // Init histos
  std::vector<std::string> vHistoNames;
  InitHistoNames(vHistoNames);
  TH2F* hTimeCharge = new TH2F(vHistoNames.at(0).c_str(), "", 400, -200, 200, 4096, 0, 4096);
  TH2F* hTimeCh = new TH2F(vHistoNames.at(1).c_str(), "", 400, -200, 200, nChannels, 0, nChannels);
  TH2F* hChargeCh = new TH2F(vHistoNames.at(2).c_str(), "", 4096, 0, 4096, nChannels, 0, nChannels);
  TH1F* hNchannels = new TH1F(vHistoNames.at(3).c_str(), "", nChannels, 0, nChannels);
  TH1D* ht = nullptr; // Projection of the upper TH2F -> created later
  TH1D* hc = nullptr; // Projection of the upper TH2F -> created later

  std::unique_ptr<TFile> digiFile(TFile::Open(digiFName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(error) << "Failed to open input digits file " << digiFName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(error) << "Failed to get digits tree";
    return;
  }

  std::vector<o2::ft0::Digit> ft0BCData, *ft0BCDataPtr = &ft0BCData;
  std::vector<o2::ft0::ChannelData> ft0ChData, *ft0ChDataPtr = &ft0ChData;
  o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* labelsPtr = nullptr;

  digiTree->SetBranchAddress("FT0DIGITSBC", &ft0BCDataPtr);
  digiTree->SetBranchAddress("FT0DIGITSCH", &ft0ChDataPtr);
  if (digiTree->GetBranch("FV0DigitLabels")) {
    digiTree->SetBranchAddress("setProperBranchName", &labelsPtr);
  }

  UInt_t nEntries = digiTree->GetEntries();
  for (UInt_t ient = 0; ient < nEntries; ient++) {
    digiTree->GetEntry(ient);
    int nbc = ft0BCData.size();
    if (printAllToTerminal) {
      for (int ibc = 0; ibc < nbc; ibc++) {
        const auto& bcd = ft0BCData[ibc];
        bcd.printStream(std::cout);
        int chEnt = bcd.ref.getFirstEntry();
        for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
          const auto& chd = ft0ChData[chEnt++];
          chd.print();
        }

        if (labelsPtr) {
          const auto lbl = labelsPtr->getLabels(ibc);
          for (unsigned int lb = 0; lb < lbl.size(); lb++) {
            printf("Det%3d ", lbl[lb].getDetID());
            printf("Src%3d ", lbl[lb].getSourceID());
            lbl[lb].print();
          }
        }
      }
    }

    // Fill histos
    for (int ibc = 0; ibc < nbc; ibc++) {
      if ((nbc > 100) && (ibc % (nbc / 100) == 0)) {
        std::cout << "  Progress reading tree: " << ibc << "/" << nbc << " [";
        std::cout << 100.0 * ibc / nbc << "%]" << std::endl;
      }
      const auto& bcd = ft0BCData[ibc];
      // std::cout << ibc << " " << nbc << " - " << bcd.ir << std::endl;
      int chEnt = bcd.ref.getFirstEntry();
      int nchannels = 0;
      for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
        const auto& chd = ft0ChData[chEnt++];
        // std::cout << chd.ChId << "  " << chd.QTCAmpl << "  " << chd.CFDTime << std::endl;
        hTimeCharge->Fill(chd.CFDTime, chd.QTCAmpl);
        hTimeCh->Fill(chd.CFDTime, chd.ChId);
        hChargeCh->Fill(chd.QTCAmpl, chd.ChId);
        if (chd.QTCAmpl > 0) {
          nchannels++;
        }
      }
      hNchannels->Fill(nchannels);
    }
  }

  // Setup histo properties
  hTimeCharge->SetXTitle("Time [ch = 13.02 ps]");
  hTimeCharge->SetYTitle("Charge");
  hTimeCharge->SetZTitle("Counts");
  hTimeCh->SetXTitle("Time [ch = 13.02 ps]");
  hTimeCh->SetYTitle("Channel");
  hTimeCh->SetZTitle("Counts");
  hChargeCh->SetXTitle("Charge");
  hChargeCh->SetYTitle("Channel");
  hChargeCh->SetZTitle("Counts");
  hNchannels->SetXTitle("#Channels");
  hNchannels->SetYTitle("Counts");
  hNchannels->SetLineColor(kBlack);
  ht = hTimeCharge->ProjectionX("hTime_prX");
  hc = hTimeCharge->ProjectionY("hCharge_prY");
  const float rmargin = 0.12, lmargin = 0.12, tmargin = 0.02, bmargin = 0.15;
  const float statX1 = 1. - rmargin, statX2 = statX1 - 0.18;
  const float statH = 0.3, statY1 = 1. - tmargin, statY2 = statY1 - statH;
  float fontsize = 0.06;
  const UInt_t nHistos = 6;
  TH1* h[nHistos] = {hTimeCharge, hTimeCh, hChargeCh, hNchannels, ht, hc};
  for (UInt_t ih = 0; ih < nHistos; ih++) {
    h[ih]->SetDirectory(0);
    h[ih]->GetXaxis()->SetTitleSize(fontsize);
    h[ih]->GetYaxis()->SetTitleSize(fontsize);
    h[ih]->GetZaxis()->SetTitleSize(fontsize);
    h[ih]->GetXaxis()->SetLabelSize(fontsize * 0.8);
    h[ih]->GetYaxis()->SetLabelSize(fontsize * 0.8);
    h[ih]->GetZaxis()->SetLabelSize(fontsize * 0.8);
    h[ih]->GetXaxis()->SetTitleOffset(1.0);
    h[ih]->GetYaxis()->SetTitleOffset(1.1);
    h[ih]->SetLineWidth(2);
  }

  // Draw histos
  const float zoomLevelRms = 3;
  TCanvas* ctc = new TCanvas("ft0digi-timeCharge", "ft0digi-timeCharge", 1800, 900);
  ctc->Divide(2, 2);
  ctc->cd(1);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hTimeCh->Rebin2D(rebin, 1);
  hTimeCh->GetXaxis()->SetRangeUser(tMin, tMax);
  hTimeCh->Draw("colz");
  // AdjustStatBox(hTimeCh, statX1, statX2, statY1, statY2);
  ctc->cd(2);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hChargeCh->Rebin2D(rebin, 1);
  hChargeCh->GetXaxis()->SetRangeUser(chargeMin, chargeMax);
  hChargeCh->Draw("colz");
  // AdjustStatBox(hChargeCh, statX1, statX2, statY1, statY2);
  ctc->cd(3);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogy();
  ht->Rebin(rebin);
  ht->GetXaxis()->SetRangeUser(tMin, tMax);
  ht->Draw();
  // AdjustStatBox(ht, statX1, statX2, statY1, statY2);
  ctc->cd(4);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogy();
  hc->Rebin(rebin);
  hc->GetXaxis()->SetRangeUser(chargeMin, chargeMax);
  hc->Draw();
  // AdjustStatBox(hc, statX1, statX2, statY1, statY2);

  TCanvas* cmulti = new TCanvas("ft0digi-multi", "ft0digi-multi", 1800, 500);
  cmulti->Divide(2);
  cmulti->cd(1);
  {
    gPad->SetMargin(1.3 * lmargin, rmargin, bmargin, tmargin);
    hNchannels->Draw();
  }
  cmulti->cd(2);
  {
    gPad->SetMargin(1.3 * lmargin, rmargin, bmargin, tmargin);
    hTimeCharge->Rebin2D(rebin, rebin);
    hTimeCharge->GetXaxis()->SetRangeUser(tMin, tMax);
    hTimeCharge->GetYaxis()->SetRangeUser(chargeMin, chargeMax);
    hTimeCharge->Draw("colz");
    // AdjustStatBox(hTimeCharge, statX1, statX2, statY1, statY2);
  }

  // Save histos
  TFile* fout = new TFile("ft0digi-rawhistos.root", "RECREATE");
  hTimeCharge->Write();
  hTimeCh->Write();
  hChargeCh->Write();
  hNchannels->Write();
  fout->Close();
  std::cout << "Digits read" << std::endl;
}

void DrawTextNdc(std::string s, double x, double y, double size, Color_t col, Float_t tangle)
{
  TLatex* t = new TLatex(0, 0, s.c_str());
  t->SetTextAngle(tangle);
  t->SetNDC();
  t->SetX(x);
  t->SetY(y);
  t->SetTextSize(size);
  t->SetTextFont(42);
  t->SetTextColor(col);
  t->Draw();
}

void SetupPad(TH1* h, float fontSize, float lmargin, float rmargin, float tmargin, float bmargin, float xoffset, float yoffset)
{
  gPad->SetTopMargin(tmargin);
  gPad->SetBottomMargin(bmargin);
  gPad->SetLeftMargin(lmargin);
  gPad->SetRightMargin(rmargin);
  h->GetXaxis()->SetLabelSize(fontSize);
  h->GetXaxis()->SetTitleSize(fontSize);
  h->GetYaxis()->SetLabelSize(fontSize);
  h->GetYaxis()->SetTitleSize(fontSize);
  h->GetXaxis()->SetTitleOffset(xoffset);
  h->GetYaxis()->SetTitleOffset(yoffset);
}

// Root files generated in the previous stage (readFV0Digits()) are used as inputs here
int compareFT0Digits(std::string digiFName1 = "ft0digi-rawhistos.root", std::string digiFName2 = "", std::string digiFName3 = "")
{
  const int tmin = -100, tmax = 100, cmin = 0, cmax = 180;
  gStyle->SetOptStat("");
  std::vector<Color_t> vcol;
  vcol.push_back(kBlack);
  vcol.push_back(kRed);
  vcol.push_back(kBlue);

  // Open files
  std::vector<TFile*> vf;
  TFile* f1 = new TFile(digiFName1.c_str(), "OPEN");
  if (f1->IsOpen()) {
    vf.push_back(f1);
  } else {
    std::cout << "<E> Problem reading the first file: " << digiFName1 << std::endl;
    return -1;
  }
  if (digiFName2.size() > 0) {
    TFile* f2 = new TFile(digiFName2.c_str(), "OPEN");
    if (f2->IsOpen()) {
      vf.push_back(f2);
    }
  }
  if (digiFName3.size() > 0) {
    TFile* f3 = new TFile(digiFName3.c_str(), "OPEN");
    if (f3->IsOpen()) {
      vf.push_back(f3);
    }
  }
  std::cout << "  <I> Number of accessible files: " << vf.size() << std::endl;

  // Open histos
  std::vector<std::string> vHistoNames;
  InitHistoNames(vHistoNames);
  std::vector<TH1*> vh, vht, vhc;
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
  const float statH = 0.3, statY1 = 1. - tmargin, statY2 = statY1 - statH;

  // Draw the comparison of TH1's (sum of all channels)
  Color_t col[3] = {kBlack, kRed, kBlue};
  TCanvas* c = new TCanvas("ft0digi-cmp-th1", "ft0digi-cmp-th1", 1800, 500);
  c->Divide(2, 1);
  for (UInt_t ifile = 0; ifile < nFiles; ifile++) {
    TH2F* h2 = (TH2F*)vh.at(ifile * nHistos);
    h2->SetLineColor(col[ifile]);
    h2->SetLineWidth(2);
    std::stringstream ss;
    ss << "p" << ifile;
    TH1D* ht = h2->ProjectionX((ss.str() + "t_" + h2->GetName()).c_str());
    TH1D* hc = h2->ProjectionY((ss.str() + "c_" + h2->GetName()).c_str());
    hc->GetXaxis()->SetRangeUser(0, 100);
    c->cd(1);
    gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
    gPad->SetLogy();
    ht->SetLineWidth(3.5 - ifile);
    ht->Draw((ifile == 0) ? "" : "sames");
    AdjustStatBox(ht, statX1, statX2, statY1 - statH * ifile, statY2 - statH * ifile);
    c->cd(2);
    gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
    gPad->SetLogy();
    hc->SetLineWidth(3.5 - ifile);
    hc->Draw((ifile == 0) ? "" : "sames");
    AdjustStatBox(hc, statX1, statX2, statY1 - statH * ifile, statY2 - statH * ifile);
  }

  // Draw the comparison of time and amplitude channel by channel
  std::vector<std::string> vnamestat;
  vnamestat.push_back("htmean1");
  vnamestat.push_back("htrms1");
  vnamestat.push_back("hcmean1");
  vnamestat.push_back("hcrms1");
  vnamestat.push_back("htmean2");
  vnamestat.push_back("htrms2");
  vnamestat.push_back("hcmean2");
  vnamestat.push_back("hcrms2");
  vnamestat.push_back("htmean3");
  vnamestat.push_back("htrms3");
  vnamestat.push_back("hcmean3");
  vnamestat.push_back("hcrms3");
  std::vector<TH1D*> vhstat;
  for (unsigned int i = 0; i < vnamestat.size(); i++) {
    vhstat.push_back(new TH1D(vnamestat.at(i).c_str(), "", nChannels, 0, nChannels));
  }
  std::vector<TCanvas*> vcvs;
  for (unsigned int icvs = 0; icvs < 4; icvs++) {
    std::stringstream ssct, sscc;
    ssct << "ft0digi-cmp-time" << icvs;
    vcvs.push_back(new TCanvas(ssct.str().c_str(), ssct.str().c_str(), 1250, 820));
  }
  for (unsigned int icvs = 0; icvs < 4; icvs++) {
    std::stringstream sscc;
    sscc << "ft0digi-cmp-charge" << icvs;
    vcvs.push_back(new TCanvas(sscc.str().c_str(), sscc.str().c_str(), 1250, 820));
  }
  for (unsigned int icvs = 0; icvs < vcvs.size(); icvs++) {
    vcvs.at(icvs)->Divide(6, nChannels / 6 / 4 + 1);
  }
  for (UInt_t ifile = 0; ifile < nFiles; ifile++) {
    TH2F* h2t = (TH2F*)vh.at(ifile * nHistos + 1);
    h2t->GetXaxis()->SetRangeUser(tmin, tmax);
    TH2F* h2c = (TH2F*)vh.at(ifile * nHistos + 2);
    h2c->GetXaxis()->SetRangeUser(cmin, cmax);
    for (unsigned int ich = 0; ich < nChannels; ich++) {
      std::stringstream sst, ssc;
      vcvs.at(ich / 60)->cd((ich % 60) + 1);
      {
        sst << "t" << ifile << "_" << ich;
        TH1D* ht = h2t->ProjectionX(sst.str().c_str(), ich, ich);
        SetupPad(ht, 0.16, 0.01, 0, 0, 0.17, 1, 1);
        ht->SetLineColor(vcol.at(ifile));
        ht->Draw((ifile == 0) ? "" : "same");
        ht->GetXaxis()->SetRangeUser(tmin, tmax);
        vhstat.at(ifile * 4 + 0)->SetBinContent(ich + 1, ht->GetMean());
        vhstat.at(ifile * 4 + 1)->SetBinContent(ich + 1, ht->GetRMS());

        std::stringstream ss1, ss2;
        ss1 << "#mu = " << std::setprecision(2) << ht->GetMean();
        ss2 << "RMS = " << std::setprecision(2) << ht->GetRMS();
        if (ifile == 0) {
          std::stringstream ss;
          ss << ich;
          DrawTextNdc(ss.str(), 0.07, 0.75, 0.3, kBlack, 0);
        }
        DrawTextNdc(ss1.str(), 0.6, 0.85 - ifile * 0.3, 0.15, ht->GetLineColor(), 0);
        DrawTextNdc(ss2.str(), 0.6, 0.7 - ifile * 0.3, 0.15, ht->GetLineColor(), 0);
      }
      vcvs.at(ich / 60 + 4)->cd((ich % 60) + 1);
      {
        ssc << "c" << ifile << "_" << ich;
        TH1D* hc = h2c->ProjectionX(ssc.str().c_str(), ich, ich);
        SetupPad(hc, 0.16, 0.01, 0, 0, 0.17, 1, 1);
        hc->SetLineColor(vcol.at(ifile));
        hc->Draw((ifile == 0) ? "" : "same");
        hc->GetXaxis()->SetRangeUser(cmin, cmax);
        vhstat.at(ifile * 4 + 2)->SetBinContent(ich + 1, hc->GetMean());
        vhstat.at(ifile * 4 + 3)->SetBinContent(ich + 1, hc->GetRMS());

        std::stringstream ss1, ss2;
        ss1 << "#mu = " << std::setprecision(2) << hc->GetMean();
        ss2 << "RMS = " << std::setprecision(2) << hc->GetRMS();
        if (ifile == 0) {
          std::stringstream ss;
          ss << ich;
          DrawTextNdc(ss.str(), 0.07, 0.75, 0.3, kBlack, 0);
        }
        DrawTextNdc(ss1.str(), 0.6, 0.85 - ifile * 0.3, 0.15, hc->GetLineColor(), 0);
        DrawTextNdc(ss2.str(), 0.6, 0.7 - ifile * 0.3, 0.15, hc->GetLineColor(), 0);
      }
    }
  }

  // Draw the comparison of mean & rms for charge and time
  TCanvas* cstat = new TCanvas("ft0digi-cmp-stat", "ft0digi-cmp-stat", 1250, 820);
  cstat->Divide(2, 2);
  float ymin = 0, ymax = 2;
  for (int ipad = 0; ipad < 4; ipad++) {
    cstat->cd(ipad + 1);
    gPad->SetGrid(1, 1);

    //    vhstat.at(ipad)->SetLineColor(vcol.at(2));
    //    vhstat.at(ipad)->GetYaxis()->SetRangeUser(ymin, ymax);
    //    vhstat.at(ipad)->Draw();

    vhstat.at(ipad + 4)->SetLineColor(vcol.at(0));
    vhstat.at(ipad + 4)->Divide(vhstat.at(ipad + 0));
    vhstat.at(ipad + 4)->GetYaxis()->SetRangeUser(ymin, ymax);
    vhstat.at(ipad + 4)->Draw("same");

    vhstat.at(ipad + 8)->SetLineColor(vcol.at(1));
    vhstat.at(ipad + 8)->Divide(vhstat.at(ipad + 0));
    vhstat.at(ipad + 8)->GetYaxis()->SetRangeUser(ymin, ymax);
    vhstat.at(ipad + 8)->Draw("same");
  }
  return 0;
}

#endif
