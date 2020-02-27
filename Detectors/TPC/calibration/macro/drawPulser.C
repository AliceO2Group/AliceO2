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
#include "TROOT.h"
#include "TMath.h"
#include "TH2.h"
#include "TFile.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Painter.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TString.h"
#endif

TH1* GetBinInfoXY(int& binx, int& biny, float& bincx, float& bincy);

/// Add fec information to the active histogram title
void addFECInfo();

/// Open pedestalFile and retrieve noise and pedestal values
/// Draw then in separate canvases and add an executable to be able to add
/// FEC information to the title
void drawPulser(TString pulserFile)
{
  using namespace o2::tpc;
  TFile f(pulserFile);
  gROOT->cd();

  // ===| load pulser from file |===
  CalDet<float> dummy;
  CalDet<float>*t0 = nullptr, *width = nullptr, *qtot = nullptr;
  f.GetObject("T0", t0);
  f.GetObject("Width", width);
  f.GetObject("Qtot", qtot);

  // ===| loop over all ROCs |==================================================
  for (int iroc = 0; iroc < int(t0->getData().size()); ++iroc) {
    const auto& rocT0 = t0->getCalArray(iroc);
    const auto& rocWidth = width->getCalArray(iroc);
    const auto& rocQtot = qtot->getCalArray(iroc);

    // only draw if valid data
    if (!(std::abs(rocT0.getSum() + rocWidth.getSum() + rocQtot.getSum()) > 0)) {
      continue;
    }

    // ===| automatically set up ranges |=======================================
    const auto medianT0 = TMath::Median(rocT0.getData().size(), rocT0.getData().data());
    const auto medianWidth = TMath::Median(rocWidth.getData().size(), rocWidth.getData().data());
    const auto medianQtot = TMath::Median(rocQtot.getData().size(), rocQtot.getData().data());

    const float rangeT0 = 0.5;
    const float minT0 = medianT0 - rangeT0;
    const float maxT0 = medianT0 + rangeT0;

    const float rangeWidth = 0.1;
    const float minWidth = medianWidth - rangeWidth;
    const float maxWidth = medianWidth + rangeWidth;

    const float rangeQtot = 150;
    const float minQtot = medianQtot - 50;
    const float maxQtot = medianQtot + rangeQtot;

    // ===| histograms for t0, width and qtot |===
    auto hT0 = new TH1F(Form("hT0%02d", iroc), Form("T0 distribution ROC %02d;time bins (0.2 #mus)", iroc), 100, minT0, maxT0);
    auto hWidth = new TH1F(Form("hWidth%02d", iroc), Form("Width distribution ROC %02d;time bins (0.2 #mus)", iroc), 100, minWidth, maxWidth);
    auto hQtot = new TH1F(Form("hQtot%02d", iroc), Form("Qtot distribution ROC %02d;ADC counts", iroc), 100, minQtot, maxQtot);

    auto hT02D = painter::getHistogram2D(rocT0);
    hT02D->SetStats(0);
    hT02D->SetMinimum(minT0);
    hT02D->SetMaximum(maxT0);

    auto hWidth2D = painter::getHistogram2D(rocWidth);
    hWidth2D->SetStats(0);
    hWidth2D->SetMinimum(minWidth);
    hWidth2D->SetMaximum(maxWidth);

    auto hQtot2D = painter::getHistogram2D(rocQtot);
    hQtot2D->SetStats(0);
    hQtot2D->SetMinimum(minQtot);
    hQtot2D->SetMaximum(maxQtot);

    // ===| fill 1D histograms |===
    for (const auto& val : rocT0.getData()) {
      if (val > 0) {
        hT0->Fill(val);
      }
    }

    for (const auto& val : rocWidth.getData()) {
      if (val > 0) {
        hWidth->Fill(val);
      }
    }

    for (const auto& val : rocQtot.getData()) {
      if (val > 0) {
        hQtot->Fill(val);
      }
    }

    // ===| draw histograms |===

    auto cPulser = new TCanvas(Form("Pulser_Info_%02d", iroc), Form("Pulser Info ROC %02d", iroc));
    cPulser->Divide(3, 2);

    cPulser->cd(1);
    gPad->AddExec(Form("addFECInfoT0%02d", iroc), "addFECInfo()");
    hT02D->Draw("colz");
    hT02D->SetUniqueID(iroc);

    cPulser->cd(2);
    gPad->AddExec(Form("addFECInfoWidth%02d", iroc), "addFECInfo()");
    hWidth2D->Draw("colz");
    hWidth2D->SetUniqueID(iroc);

    cPulser->cd(3);
    gPad->AddExec(Form("addFECInfoQtot%02d", iroc), "addFECInfo()");
    hQtot2D->Draw("colz");
    hQtot2D->SetUniqueID(iroc);

    cPulser->cd(4);
    hT0->Draw();

    cPulser->cd(5);
    hWidth->Draw();

    cPulser->cd(6);
    hQtot->Draw();

    cPulser->SaveAs(Form("%s.png", cPulser->GetName()));
    cPulser->SaveAs(Form("%s.pdf", cPulser->GetName()));
  }
}

TH1* GetBinInfoXY(int& binx, int& biny, float& bincx, float& bincy)
{
  TObject* select = gPad->GetSelected();
  if (!select)
    return 0x0;
  if (!select->InheritsFrom("TH2")) {
    gPad->SetUniqueID(0);
    return 0x0;
  }

  TH1* h = (TH1*)select;
  gPad->GetCanvas()->FeedbackMode(kTRUE);

  const int px = gPad->GetEventX();
  const int py = gPad->GetEventY();
  const float xx = gPad->AbsPixeltoX(px);
  const float x = gPad->PadtoX(xx);
  const float yy = gPad->AbsPixeltoY(py);
  const float y = gPad->PadtoX(yy);
  binx = h->GetXaxis()->FindBin(x);
  biny = h->GetYaxis()->FindBin(y);
  bincx = h->GetXaxis()->GetBinCenter(binx);
  bincy = h->GetYaxis()->GetBinCenter(biny);
  //printf("binx, biny: %d %d\n",binx,biny);

  return h;
}

void addFECInfo()
{
  using namespace o2::tpc;
  const int event = gPad->GetEvent();
  if (event != 51) {
    return;
  }

  int binx, biny;
  float bincx, bincy;
  TH1* h = GetBinInfoXY(binx, biny, bincx, bincy);
  if (!h) {
    return;
  }

  const float binValue = h->GetBinContent(binx, biny);
  const int row = int(TMath::Floor(bincx));
  const int cpad = int(TMath::Floor(bincy));

  const auto& mapper = Mapper::instance();

  const int roc = h->GetUniqueID();
  if (roc < 0 || roc >= (int)ROC::MaxROC)
    return;
  if (row < 0 || row >= (int)mapper.getNumberOfRowsROC(roc))
    return;
  const int nPads = mapper.getNumberOfPadsInRowROC(roc, row);
  const int pad = cpad + nPads / 2;
  //printf("row %d, cpad %d, pad %d, nPads %d\n", row, cpad, pad, nPads);
  if (pad < 0 || pad >= (int)nPads) {
    return;
  }
  const int channel = mapper.getPadNumberInROC(PadROCPos(roc, row, pad));

  const auto& fecInfo = mapper.getFECInfo(PadROCPos(roc, row, pad));

  TString title("#splitline{#lower[.1]{#scale[.5]{");
  title += (roc / 18 % 2 == 0) ? "A" : "C";
  title += Form("%02d (%02d) row: %02d, pad: %03d, globalpad: %05d (in roc)}}}{#scale[.5]{FEC: %02d, Chip: %02d, Chn: %02d, Value: %.3f}}",
                roc % 18, roc, row, pad, channel, fecInfo.getIndex(), fecInfo.getSampaChip(), fecInfo.getSampaChannel(), binValue);

  h->SetTitle(title.Data());
}
