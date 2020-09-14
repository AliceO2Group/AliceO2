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
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FV0Base/Constants.h"
#include "DataFormatsFV0/MCLabel.h"
#include "FairLogger.h"

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
  vhName.push_back("hMipsCh");
  vhName.push_back("hNchReal");
  vhName.push_back("hNchPrec");
}

void readFV0Digits(std::string digiFName = "fv0digits.root", bool printAllToTerminal = false, UInt_t rebin = 1)
{
  gStyle->SetOptStat("noumri");
  // Settings for drawing, used with SetRangeUser()
  const float tMin = -30, tMax = 70, chargeMin = 0, chargeMax = 800;
  const float chargePerMip = 1. / 16; // Assumes 1-MIP-peak is at 16th ADC channel
  const int nChannels = o2::fv0::Constants::nFv0Channels;

  // Init histos
  std::vector<std::string> vHistoNames;
  InitHistoNames(vHistoNames);
  TH2F* hTimeCharge = new TH2F(vHistoNames.at(0).c_str(), "", 400, -200, 200, 4096, 0, 4096);
  TH2F* hTimeCh = new TH2F(vHistoNames.at(1).c_str(), "", 400, -200, 200, nChannels, 0, nChannels);
  TH2F* hChargeCh = new TH2F(vHistoNames.at(2).c_str(), "", 4096, 0, 4096, nChannels, 0, nChannels);
  TH2F* hMipsCh = new TH2F(vHistoNames.at(3).c_str(), "", 256, 0, 256, nChannels, 0, nChannels);
  TH1F* hNchReal = new TH1F(vHistoNames.at(4).c_str(), "", 600, 0, 600);
  TH1F* hNchPrec = new TH1F(vHistoNames.at(5).c_str(), "", 600, 0, 600);
  TH1D* ht = nullptr; // Projection of the upper TH2F -> created later
  TH1D* hc = nullptr; // Projection of the upper TH2F -> created later

  std::unique_ptr<TFile> digiFile(TFile::Open(digiFName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(ERROR) << "Failed to open input digits file " << digiFName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(ERROR) << "Failed to get digits tree";
    return;
  }

  std::vector<o2::fv0::BCData> fv0BCData, *fv0BCDataPtr = &fv0BCData;
  std::vector<o2::fv0::ChannelData> fv0ChData, *fv0ChDataPtr = &fv0ChData;
  o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>* labelsPtr = nullptr;

  digiTree->SetBranchAddress("FV0DigitBC", &fv0BCDataPtr);
  digiTree->SetBranchAddress("FV0DigitCh", &fv0ChDataPtr);
  if (digiTree->GetBranch("FV0DigitLabels")) {
    digiTree->SetBranchAddress("FV0DigitLabels", &labelsPtr);
  }

  UInt_t nEntries = digiTree->GetEntries();
  for (UInt_t ient = 0; ient < nEntries; ient++) {
    digiTree->GetEntry(ient);
    int nbc = fv0BCData.size();
    if (printAllToTerminal) {
      for (int ibc = 0; ibc < nbc; ibc++) {
        const auto& bcd = fv0BCData[ibc];
        bcd.print();
        int chEnt = bcd.ref.getFirstEntry();
        for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
          const auto& chd = fv0ChData[chEnt++];
          chd.print();
        }

        if (labelsPtr) {
          const auto lbl = labelsPtr->getLabels(ibc);
          for (int lb = 0; lb < lbl.size(); lb++) {
            printf("Ch%3d ", lbl[lb].getChannel());
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
      const auto& bcd = fv0BCData[ibc];
      std::cout << ibc << " " << nbc << " - " << bcd.ir << std::endl;
      int chEnt = bcd.ref.getFirstEntry();
      int nchPrec = 0;
      int nchReal = 0;
      for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
        const auto& chd = fv0ChData[chEnt++];
        std::cout << chd.pmtNumber << "  " << chd.chargeAdc << "  " << chd.time << std::endl;
        hTimeCharge->Fill(chd.time, chd.chargeAdc);
        hTimeCh->Fill(chd.time, chd.pmtNumber);
        hChargeCh->Fill(chd.chargeAdc, chd.pmtNumber);
        float mips = chargePerMip * chd.chargeAdc;
        nchPrec += mips;
        if (mips < 0.4) {
          mips = 0;
        } else if (mips < 1.5) {
          mips = 1;
        }
        nchReal += round(mips);
        hMipsCh->Fill(round(mips), chd.pmtNumber);
      }
      hNchPrec->Fill(nchPrec);
      hNchReal->Fill(nchReal);
    }
  }

  // Setup histo properties
  hTimeCharge->SetXTitle("Time [ns]");
  hTimeCharge->SetYTitle("Charge");
  hTimeCharge->SetZTitle("Counts");
  hTimeCh->SetXTitle("Time [ns]");
  hTimeCh->SetYTitle("Channel");
  hTimeCh->SetZTitle("Counts");
  hChargeCh->SetXTitle("Charge");
  hChargeCh->SetYTitle("Channel");
  hChargeCh->SetZTitle("Counts");
  hMipsCh->SetXTitle("#MIPs");
  hMipsCh->SetYTitle("Channel");
  hMipsCh->SetZTitle("Counts");
  hNchReal->SetXTitle("#MIPs");
  hNchReal->SetYTitle("Counts");
  hNchPrec->SetXTitle(hNchReal->GetXaxis()->GetTitle());
  hNchPrec->SetYTitle(hNchReal->GetYaxis()->GetTitle());
  hNchReal->SetLineColor(kBlack);
  hNchPrec->SetLineColor(kRed);
  ht = hTimeCharge->ProjectionX("hTime_prX");
  hc = hTimeCharge->ProjectionY("hCharge_prY");
  const float rmargin = 0.12, lmargin = 0.12, tmargin = 0.02, bmargin = 0.15;
  const float statX1 = 1. - rmargin, statX2 = statX1 - 0.18;
  const float statH = 0.3, statY1 = 1. - tmargin, statY2 = statY1 - statH;
  float fontsize = 0.06;
  const UInt_t nHistos = 8;
  TH1* h[nHistos] = {hTimeCharge, hTimeCh, hChargeCh, hMipsCh, hNchReal, hNchPrec, ht, hc};
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
  TCanvas* ctc = new TCanvas("fv0digi-timeCharge", "fv0digi-timeCharge", 1800, 900);
  ctc->Divide(2, 2);
  ctc->cd(1);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hTimeCh->Rebin2D(rebin, 1);
  hTimeCh->GetXaxis()->SetRangeUser(tMin, tMax);
  hTimeCh->Draw("colz");
  AdjustStatBox(hTimeCh, statX1, statX2, statY1, statY2);
  ctc->cd(2);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogz();
  hChargeCh->Rebin2D(rebin, 1);
  hChargeCh->GetXaxis()->SetRangeUser(chargeMin, chargeMax);
  hChargeCh->Draw("colz");
  AdjustStatBox(hChargeCh, statX1, statX2, statY1, statY2);
  ctc->cd(3);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogy();
  ht->Rebin(rebin);
  ht->GetXaxis()->SetRangeUser(tMin, tMax);
  ht->Draw();
  AdjustStatBox(ht, statX1, statX2, statY1, statY2);
  ctc->cd(4);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  gPad->SetLogy();
  hc->Rebin(rebin);
  hc->GetXaxis()->SetRangeUser(chargeMin, chargeMax);
  hc->Draw();
  AdjustStatBox(hc, statX1, statX2, statY1, statY2);

  TCanvas* cmulti = new TCanvas("fv0digi-multi", "fv0digi-multi", 1800, 500);
  cmulti->Divide(3, 1);
  cmulti->cd(1);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  hMipsCh->Draw("colz");
  AdjustStatBox(hMipsCh, statX1, statX2, statY1, statY2);
  cmulti->cd(2);
  gPad->SetMargin(lmargin, rmargin, bmargin, tmargin);
  hNchPrec->Rebin(rebin);
  hNchReal->Rebin(rebin);
  hNchPrec->Draw();
  hNchReal->Draw("sames");
  AdjustStatBox(hNchPrec, statX1, statX2, statY1, statY2);
  AdjustStatBox(hNchReal, statX1, statX2, statY1 - statH, statY2 - statH);
  cmulti->cd(3);
  gPad->SetMargin(1.3 * lmargin, rmargin, bmargin, tmargin);
  hTimeCharge->Rebin2D(rebin, rebin);
  hTimeCharge->GetXaxis()->SetRangeUser(tMin, tMax);
  hTimeCharge->GetYaxis()->SetRangeUser(chargeMin, chargeMax);
  hTimeCharge->Draw("colz");
  AdjustStatBox(hTimeCharge, statX1, statX2, statY1, statY2);

  // Save histos
  TFile* fout = new TFile("fv0digi-rawhistos.root", "RECREATE");
  hTimeCharge->Write();
  hTimeCh->Write();
  hChargeCh->Write();
  hMipsCh->Write();
  hNchReal->Write();
  hNchPrec->Write();
  fout->Close();
  std::cout << "Digits read" << std::endl;
}

// Root files generated in the previous stage (readFV0Digits()) are used as inputs here
int compareFV0Digits(std::string digiFName1 = "fv0digi-rawhistos.root", std::string digiFName2 = "", std::string digiFName3 = "")
{
  gStyle->SetOptStat("noumri");

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
  const float statH = 0.3, statY1 = 1. - tmargin, statY2 = statY1 - statH;

  /*  // Draw side-by-side comparison of TH2's
  for (UInt_t ih = 0; ih < 4; ih++) {
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
*/
  // Draw the comparison of TH1's
  Color_t col[3] = {kBlack, kRed, kBlue};
  TCanvas* c = new TCanvas("fv0digi-cmp-th1", "fv0digi-cmp-th1", 1800, 500);
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
  return 0;
}

#endif
