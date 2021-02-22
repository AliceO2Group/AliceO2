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
#include "TPCBase/Utils.h"
#include "TPCBase/CDBInterface.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TH1F.h"
#endif

/// Open pedestalFile and retrieve noise and pedestal values
/// Draw then in separate canvases and add an executable to be able to add
/// FEC information to the title
/// \param pedestalFile input file name
/// \param mode 0: one canvas per ROC 2D and 1D, 1: one canvas per full TPC, one canvas with all ROCs, for each noise and pedestal
/// \param outDir output directory to store plos in. Don't store if empty
/// \return Array with canvases
TObjArray* drawNoiseAndPedestal(std::string_view pedestalFile, int mode = 0, std::string_view outDir = "")
{
  if ((mode != 0) && (mode != 1)) {
    return 0x0;
  }

  TObjArray* arrCanvases = new TObjArray;
  arrCanvases->SetName("NoiseAndPedestals");

  using namespace o2::tpc;

  // ===| load noise and pedestal from file |===
  CalDet<float> dummy;
  const CalDet<float>* calPedestal = nullptr;
  const CalDet<float>* calNoise = nullptr;

  if (pedestalFile.find("cdb") != std::string::npos) {
    auto& cdb = CDBInterface::instance();
    if (pedestalFile == "cdb-test") {
      cdb.setURL("http://ccdb-test.cern.ch:8080");
    } else if (pedestalFile == "cdb-prod") {
      cdb.setURL("");
    }
    calPedestal = &cdb.getPedestals();
    calNoise = &cdb.getNoise();
  } else {
    TFile f(pedestalFile.data());
    gROOT->cd();
    f.GetObject("Pedestals", calPedestal);
    f.GetObject("Noise", calNoise);
  }

  // mode 1 handling
  if (mode == 1) {
    auto arrPedestals = painter::makeSummaryCanvases(*calPedestal, 120, 20.f, 140.f);
    auto arrNoise = painter::makeSummaryCanvases(*calNoise, 100, 0.f, 5.f);

    for (auto c : arrPedestals) {
      arrCanvases->Add(c);
    }

    for (auto c : arrNoise) {
      arrCanvases->Add(c);
    }

    if (outDir.size()) {
      utils::saveCanvases(*arrCanvases, outDir, "png,pdf", "NoiseAndPedestalCanvases.root");
    }

    return arrCanvases;
  }

  // ===| loop over all ROCs |==================================================
  for (size_t iroc = 0; iroc < calPedestal->getData().size(); ++iroc) {
    const auto& rocPedestal = calPedestal->getCalArray(iroc);
    const auto& rocNoise = calNoise->getCalArray(iroc);

    // only draw if valid data
    if (!(std::abs(rocPedestal.getSum() + rocNoise.getSum()) > 0)) {
      continue;
    }

    // ===| histograms for noise and pedestal |===
    auto hPedestal = new TH1F(Form("hPedestal%02zu", iroc), Form("Pedestal distribution ROC %02zu;ADC value", iroc), 150, 0, 150);
    auto hNoise = new TH1F(Form("hNoise%02zu", iroc), Form("Noise distribution ROC %02zu;ADC value", iroc), 100, 0, 5);
    auto hPedestal2D = painter::getHistogram2D(rocPedestal);
    hPedestal2D->SetStats(0);
    hPedestal2D->SetMinimum(20);
    hPedestal2D->SetMaximum(140);
    hPedestal2D->SetUniqueID(iroc);
    auto hNoise2D = painter::getHistogram2D(rocNoise);
    hNoise2D->SetStats(0);
    hNoise2D->SetMinimum(0);
    hNoise2D->SetMaximum(5);
    hNoise2D->SetUniqueID(iroc);

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
    auto cPedestal = new TCanvas(Form("cPedestal%02zu", iroc), Form("Pedestals ROC %02zu", iroc));
    hPedestal->Draw();

    auto cNoise = new TCanvas(Form("cNoise%02zu", iroc), Form("Noise ROC %02zu", iroc));
    hNoise->Draw();

    auto cPedestal2D = new TCanvas(Form("cPedestal2D%02zu", iroc), Form("Pedestals2D ROC %02zu", iroc));
    cPedestal2D->AddExec(Form("addFECInfoPedestal%02zu", iroc), "o2::tpc::utils::addFECInfo()");
    hPedestal2D->Draw("colz");

    auto cNoise2D = new TCanvas(Form("cNoise2D%02zu", iroc), Form("Noise2D ROC %02zu", iroc));
    cNoise2D->AddExec(Form("addFECInfoNoise%02zu", iroc), "o2::tpc::utils::addFECInfo()");
    hNoise2D->Draw("colz");

    arrCanvases->Add(cPedestal);
    arrCanvases->Add(cPedestal2D);
    arrCanvases->Add(cNoise);
    arrCanvases->Add(cNoise2D);
  }

  if (outDir.size()) {
    utils::saveCanvases(*arrCanvases, outDir, "png,pdf", "NoiseAndPedestalCanvases.root");
  }

  return arrCanvases;
}
