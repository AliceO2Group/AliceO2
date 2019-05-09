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
#endif

/// Helper to get active histogram and bin information
TH1* GetBinInfoXY(int& binx, int& biny, float& bincx, float& bincy);

/// Add fec information to the active histogram title
void addFECInfo();

/// Open pedestalFile and retrieve noise and pedestal values
/// Draw then in separate canvases and add an executable to be able to add
/// FEC information to the title
void drawNoiseAndPedestal(TString pedestalFile)
{
  using namespace o2::TPC;
  TFile f(pedestalFile);
  gROOT->cd();

  // ===| load noise and pedestal from file |===
  CalDet<float> dummy;
  CalDet<float>* pedestal = nullptr;
  CalDet<float>* noise = nullptr;
  f.GetObject("Pedestals", pedestal);
  f.GetObject("Noise", noise);

  // ===| loop over all ROCs |==================================================
  for (int iroc = 0; iroc < pedestal->getData().size(); ++iroc) {
    const auto& rocPedestal = pedestal->getCalArray(iroc);
    const auto& rocNoise = noise->getCalArray(iroc);

    // only draw if valid data
    if (!(std::abs(rocPedestal.getSum() + rocNoise.getSum()) > 0)) {
      continue;
    }

    // ===| histograms for noise and pedestal |===
    auto hPedestal = new TH1F(Form("hPedestal%02d", iroc), Form("Pedestal distribution ROC %02d;ADC value", iroc), 150, 0, 150);
    auto hNoise = new TH1F(Form("hNoise%02d", iroc), Form("Noise distribution ROC %02d;ADC value", iroc), 100, 0, 5);
    auto hPedestal2D = Painter::getHistogram2D(rocPedestal);
    hPedestal2D->SetStats(0);
    auto hNoise2D = Painter::getHistogram2D(rocNoise);
    hNoise2D->SetStats(0);

    // ===| fill 1D histograms |===
    for (const auto& val : rocPedestal.getData()) {
      if (val > 0) {
        hPedestal->Fill(val);
      }
    }

    for (const auto& val : rocNoise.getData()) {
      if (val > 0) {
        hNoise->Fill(val);
      }
    }

    // ===| draw histograms |===
    auto cPedestal = new TCanvas(Form("cPedestal%02d", iroc), Form("Pedestals ROC %02d", iroc));
    hPedestal->Draw();

    auto cNoise = new TCanvas(Form("cNoise%02d", iroc), Form("Noise ROC %02d", iroc));
    hNoise->Draw();

    auto cPedestal2D = new TCanvas(Form("cPedestal2D%0d", iroc), Form("Pedestals2D ROC %02d", iroc));
    cPedestal2D->AddExec(Form("addFECInfoPedestal%02d", iroc), "addFECInfo()");
    hPedestal2D->Draw("colz");
    hPedestal2D->SetUniqueID(iroc);

    auto cNoise2D = new TCanvas(Form("cNoise2D%02d", iroc), Form("Noise2D ROC %02d", iroc));
    cNoise2D->AddExec(Form("addFECInfoNoise%02d", iroc), "addFECInfo()");
    hNoise2D->Draw("colz");
    hNoise2D->SetUniqueID(iroc);
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
  using namespace o2::TPC;
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
