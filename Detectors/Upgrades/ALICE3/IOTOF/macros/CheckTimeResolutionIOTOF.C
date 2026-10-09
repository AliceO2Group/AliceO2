// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckTimeResolutionIOTOF.C
/// \brief Simple macro to check the time resolution of TF3 digits
///
/// For each digit with a valid MC label, the true time is computed as the
/// MC hit time (relative to the collision) plus the collision time taken from
/// the digitization context. The difference between the digit time and the
/// true time is the time residual, whose width is the time resolution.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLatex.h>
#include <TMath.h>
#include <TLegend.h>
#include <TLine.h>
#include <TProfile2D.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TStyle.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <unordered_map>
#include <utility>

#include "IOTOFBase/Segmentation.h"
#include "IOTOFBase/IOTOFBaseParam.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "IOTOFSimulation/DPLDigitizerParam.h"
#include "DataFormatsIOTOF/Digit.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"

#endif

#define ENABLE_UPGRADES

namespace
{
constexpr int kNLayers = 2;
constexpr int kNEtaRegions = 2;
const char* kLayerName[kNLayers] = {"ITOF", "OTOF"};
} // namespace

/// Fit the time residual distribution with a gaussian in +-2 RMS around the mean and report the resolution
void fitTimeResidual(TH1* h, const char* name, float expectedSigmaPs)
{
  if (!h || h->GetEntries() < 10) {
    Warning(name, "Not enough entries to fit the time residual distribution");
    return;
  }
  const double mean = h->GetMean(), rms = h->GetRMS();
  h->Fit("gaus", "QR", "", mean - 2 * rms, mean + 2 * rms);
  auto fit = h->GetFunction("gaus");
  if (!fit) {
    return;
  }
  fit->SetLineColor(kRed);
  Info(name, "mean(dt)=%.2f ps, RMS(dt)=%.2f ps, fitted sigma(dt)=%.2f +- %.2f ps, expected sigma(dt)=%.2f ps",
       mean, rms, fit->GetParameter(2), fit->GetParError(2), expectedSigmaPs);
}

/// Convert a TProfile2D filled with the "s" option to a TH2 with the spread of each bin
TH2F* profileToSpread(TProfile2D* prof, const char* name, const char* title)
{
  auto h = new TH2F(name, title,
                    prof->GetNbinsX(), prof->GetXaxis()->GetXmin(), prof->GetXaxis()->GetXmax(),
                    prof->GetNbinsY(), prof->GetYaxis()->GetXmin(), prof->GetYaxis()->GetXmax());
  for (int ix = 1; ix <= prof->GetNbinsX(); ++ix) {
    for (int iy = 1; iy <= prof->GetNbinsY(); ++iy) {
      if (prof->GetBinEntries(prof->GetBin(ix, iy)) < 2) {
        continue;
      }
      h->SetBinContent(ix, iy, prof->GetBinError(ix, iy));
    }
  }
  return h;
}

/// Set pad margins leaving room for the axis titles (and for the colour palette and z title of 2D histograms)
void setPadStyle(bool is2D)
{
  gPad->SetTicks(1, 1);
  gPad->SetLeftMargin(0.16);
  gPad->SetBottomMargin(0.12);
  gPad->SetTopMargin(0.08);
  gPad->SetRightMargin(is2D ? 0.2 : 0.05);
}

/// Draw a histogram alone on the canvas and print it as a new page of the given pdf file
void printPage(TCanvas* canv, const char* pdf, TH1* h, const char* opt = "", bool logy = false, bool logz = false)
{
  canv->Clear();
  canv->cd();
  const bool is2D = h->GetDimension() > 1;
  setPadStyle(is2D);
  gPad->SetLogy(logy);
  gPad->SetLogz(logz);
  h->GetYaxis()->SetTitleOffset(1.7);
  if (is2D) {
    h->GetZaxis()->SetTitleOffset(1.6);
  }
  h->Draw(opt);
  canv->Print(pdf, Form("Title:%s", h->GetName()));
}

/// Human readable name of the selected particle species
TString speciesLabel(int pdg)
{
  switch (std::abs(pdg)) {
    case 0:
      return "all particles";
    case 11:
      return "e^{#pm}";
    case 13:
      return "#mu^{#pm}";
    case 211:
      return "#pi^{#pm}";
    case 321:
      return "K^{#pm}";
    case 2212:
      return "p, #bar{p}";
    default:
      return Form("|PDG| = %d", std::abs(pdg));
  }
}

/// Draw the time spectra of ITOF and OTOF split in central and forward pseudorapidity regions, with the total in black.
/// The dashed lines mark the arrival time of a beta = 1 particle at eta = 0.
void drawTimeSpectra(TCanvas* canv, std::array<std::array<TH1F*, kNEtaRegions>, kNLayers>& hists, const std::array<float, kNLayers>& refTime,
                     const char* header, float etaCut)
{
  canv->Clear();
  canv->cd();
  setPadStyle(false);
  gPad->SetLogy(false);
  gPad->SetLogz(false);

  auto hTotal = (TH1F*)hists[0][0]->Clone(Form("%s_total", hists[0][0]->GetName()));
  hTotal->Reset();
  for (auto& layerHists : hists) {
    for (auto& h : layerHists) {
      hTotal->Add(h);
    }
  }
  hTotal->SetLineColor(kBlack);
  hTotal->SetFillStyle(0);
  hTotal->SetStats(0);
  hTotal->GetYaxis()->SetTitleOffset(1.7);
  const double peak = hTotal->GetMaximum();
  hTotal->SetMaximum(1.2 * peak);
  hTotal->Draw("hist");

  const int fillColor[kNLayers][kNEtaRegions] = {{kBlue, kRed}, {kBlue + 3, kRed + 3}};
  const int fillStyle[kNLayers] = {3004, 3005};
  auto leg = new TLegend(0.5, 0.55, 0.88, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetHeader(header);
  for (int ie = 0; ie < kNEtaRegions; ++ie) {
    for (int il = 0; il < kNLayers; ++il) {
      auto h = hists[il][ie];
      h->SetLineColor(fillColor[il][ie]);
      h->SetFillColor(fillColor[il][ie]);
      h->SetFillStyle(fillStyle[il]);
      h->SetStats(0);
      h->Draw("hist same");
      leg->AddEntry(h, Form("%s, |#eta| %s %.1f", kLayerName[il], ie == 0 ? "<" : ">", etaCut), "f");
    }
  }
  hTotal->Draw("hist same");
  leg->Draw();

  gPad->Update();
  for (int il = 0; il < kNLayers; ++il) {
    if (refTime[il] <= 0.f) {
      continue;
    }
    auto line = new TLine(refTime[il], gPad->GetUymin(), refTime[il], 1.03 * peak);
    line->SetLineStyle(2);
    line->SetLineColor(kGray + 2);
    line->Draw();
    auto txt = new TLatex(refTime[il], 1.05 * peak, kLayerName[il]);
    txt->SetTextAlign(21);
    txt->SetTextSize(0.035);
    txt->SetTextColor(kGray + 2);
    txt->Draw();
  }
}

void CheckTimeResolutionIOTOF(std::string digifile = "tf3digits.root", std::string hitfile = "o2sim_HitsTF3.root", std::string kinefile = "o2sim_Kine.root",
                              std::string inputGeom = "o2sim_geometry.root", std::string collContextFile = "collisioncontext.root",
                              int pdgSel = 211, float ptMin = 1.f, float ptMax = 10.f, float etaCut = 0.5f, float dtMaxPs = 1e4f,
                              std::string cfgStr = "IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;IOTOFBase.enableForwardTOF=false;IOTOFBase.enableBackwardTOF=false;")
{
  gStyle->SetPalette(55);
  gStyle->SetOptStat(0); // only the fitted histograms get a box, with the fit results
  gStyle->SetOptFit(1);

  using namespace o2::base;
  using namespace o2::iotof;

  using o2::iotof::Digit;
  using o2::itsmft::Hit;

  constexpr float sec2ns = 1e9f;
  constexpr float ns2ps = 1e3f;
  constexpr float cm2um = 1e4f;
  const float speedOfLightCmNs = TMath::C() * 1e-7; // m/s -> cm/ns

  o2::conf::ConfigurableParam::updateFromString(cfgStr);

  const auto& chipInfo = o2::iotof::ChipSpecificsParam::Instance();
  const auto& digiPars = o2::iotof::DPLDigitizerParam::Instance();
  auto seg = o2::iotof::Segmentation::Instance();

  // Expected resolution: gaussian smearing convoluted with the TDC quantisation (uniform, flooring adds a -tdcBin/2 bias)
  const float expectedSigmaPs = std::sqrt(digiPars.timeResolution * digiPars.timeResolution + digiPars.tdcBin * digiPars.tdcBin / 12.f) * ns2ps;
  const float dtRangePs = std::max(200.f, 8.f * expectedSigmaPs);
  const float tdcBinPs = digiPars.tdcBin * ns2ps;
  Info("CheckTimeResolutionIOTOF", "Nominal time resolution %.1f ps, TDC bin %.1f ps -> expected sigma(dt) = %.2f ps",
       digiPars.timeResolution * ns2ps, tdcBinPs, expectedSigmaPs);

  TFile* f = TFile::Open("CheckTimeResolution.root", "recreate");

  TNtuple* nt = new TNtuple("ntt", "digit time ntuple", "id:layer:x:y:z:xLoc:zLoc:dxPix:dzPix:eta:pt:pdg:tTrue:tDig:dt");

  // Histograms
  const float tMaxNs = 13.f; // time window (relative to the collision) for the time spectra
  const int nTdcBins = int(tMaxNs / digiPars.tdcBin);
  const float halfSizeRow = 0.5f * chipInfo.ActiveMatrixSizeRows();
  const float halfSizeCol = 0.5f * chipInfo.ActiveMatrixSizeCols();
  const float halfPitchRowUm = 0.5f * chipInfo.PitchRow * cm2um;
  const float halfPitchColUm = 0.5f * chipInfo.PitchCol * cm2um;

  std::array<TH1F*, kNLayers> hTTrue, hTDig, hDt;
  std::array<TH2F*, kNLayers> hTDigVsTTrue, hDtVsTTrue, hDtVsXLoc, hDtVsZLoc, hDtVsZGlo;
  std::array<TProfile2D*, kNLayers> pDtInPixel;
  std::array<std::array<TH1F*, kNEtaRegions>, kNLayers> hTdcSpectrumDig, hTdcSpectrumTrue;
  for (int il = 0; il < kNLayers; ++il) {
    const char* ln = kLayerName[il];
    hTTrue[il] = new TH1F(Form("h_tTrue_%s", ln), Form("%s: MC true time;t_{true} - t_{collision} (ns);Digits", ln), 260, 0, tMaxNs);
    hTDig[il] = new TH1F(Form("h_tDig_%s", ln), Form("%s: digitized time;t_{digit} - t_{collision} (ns);Digits", ln), 260, 0, tMaxNs);
    hTDigVsTTrue[il] = new TH2F(Form("h_tDig_vs_tTrue_%s", ln), Form("%s: digitized vs MC true time;t_{true} - t_{collision} (ns);t_{digit} - t_{collision} (ns);Digits", ln),
                                260, 0, tMaxNs, 260, 0, tMaxNs);
    hDt[il] = new TH1F(Form("h_dt_%s", ln), Form("%s: time residual;t_{digit} - t_{true} (ps);Digits", ln), 400, -dtRangePs, dtRangePs);
    hDtVsTTrue[il] = new TH2F(Form("h_dt_vs_tTrue_%s", ln), Form("%s: time residual vs MC true time;t_{true} - t_{collision} (ns);t_{digit} - t_{true} (ps);Digits", ln),
                              130, 0, tMaxNs, 200, -dtRangePs, dtRangePs);
    hDtVsXLoc[il] = new TH2F(Form("h_dt_vs_xLoc_%s", ln), Form("%s: time residual vs local x in the chip;x_{local} of the pixel (cm);t_{digit} - t_{true} (ps);Digits", ln),
                             100, -halfSizeRow, halfSizeRow, 200, -dtRangePs, dtRangePs);
    hDtVsZLoc[il] = new TH2F(Form("h_dt_vs_zLoc_%s", ln), Form("%s: time residual vs local z in the chip;z_{local} of the pixel (cm);t_{digit} - t_{true} (ps);Digits", ln),
                             100, -halfSizeCol, halfSizeCol, 200, -dtRangePs, dtRangePs);
    hDtVsZGlo[il] = new TH2F(Form("h_dt_vs_zGlo_%s", ln), Form("%s: time residual vs global z;z_{global} of the pixel (cm);t_{digit} - t_{true} (ps);Digits", ln),
                             200, -400, 400, 200, -dtRangePs, dtRangePs);
    pDtInPixel[il] = new TProfile2D(Form("p_dt_vs_inpixel_%s", ln), Form("%s: mean time residual vs position in the pixel;x_{hit} - x_{pixel} (#mum);z_{hit} - z_{pixel} (#mum);#LTt_{digit} - t_{true}#GT (ps)", ln),
                                    40, -halfPitchRowUm, halfPitchRowUm, 40, -halfPitchColUm, halfPitchColUm, "s");
    for (int ie = 0; ie < kNEtaRegions; ++ie) {
      const char* en = ie == 0 ? "central" : "forward";
      hTdcSpectrumDig[il][ie] = new TH1F(Form("h_tdc_digit_%s_%s", ln, en), Form("Digitized time;TDC (%g ps/bin);Entries", tdcBinPs), nTdcBins, 0, nTdcBins);
      hTdcSpectrumTrue[il][ie] = new TH1F(Form("h_tdc_true_%s_%s", ln, en), Form("MC true time;t_{true} - t_{collision} (%g ps/bin);Entries", tdcBinPs), nTdcBins, 0, nTdcBins);
    }
  }

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::iotof::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Collision context: (source, entry) -> collision time
  std::map<std::pair<int, int>, double> collisionTimeNS;
  auto* context = o2::steer::DigitizationContext::loadFromFile(collContextFile);
  if (context) {
    const auto& records = context->getEventRecords();
    const auto& parts = context->getEventParts();
    for (size_t iColl = 0; iColl < records.size() && iColl < parts.size(); ++iColl) {
      for (const auto& part : parts[iColl]) {
        collisionTimeNS[{part.sourceID, part.entryID}] = records[iColl].getTimeNS();
      }
    }
    Info("CheckTimeResolutionIOTOF", "Loaded %zu collisions from %s", records.size(), collContextFile.data());
  } else {
    Warning("CheckTimeResolutionIOTOF", "Could not load the collision context from %s: using the ROF BC as collision time (valid only in triggered mode)", collContextFile.data());
  }

  // Hits
  TFile* hitFile = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)hitFile->Get("o2sim");
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  std::vector<std::vector<o2::itsmft::Hit>*> hitArray(nevH, nullptr);

  std::vector<std::unordered_map<uint64_t, int>> mc2hitVec(nevH);

  // Kinematics (optional): used for the particle species, pT and eta selections
  TFile* kineFile = TFile::Open(kinefile.data());
  TTree* kineTree = (kineFile && !kineFile->IsZombie()) ? (TTree*)kineFile->Get("o2sim") : nullptr;
  std::vector<std::vector<o2::MCTrack>*> mcTrackArray(nevH, nullptr);
  if (!kineTree) {
    Warning("CheckTimeResolutionIOTOF", "Could not load the kinematics from %s: no species/pT selection, eta from the digit position", kinefile.data());
  }

  // Digits
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");

  std::vector<o2::iotof::Digit>* digArr{nullptr};
  std::vector<o2::itsmft::ROFRecord>* rofRecordsArr{nullptr};
  o2::dataformats::IOMCTruthContainerView* plabelsArr{nullptr};

  digTree->SetBranchAddress("TF3Digit", &digArr);
  digTree->SetBranchAddress("TF3DigitROF", &rofRecordsArr);
  digTree->SetBranchAddress("TF3DigitMCTruth", &plabelsArr);

  digTree->GetEntry(0);

  // Load all MC hit (and kinematics) events upfront and build the hit lookup map.
  for (int im = 0; im < nevH; ++im) {
    hitTree->SetBranchAddress("TF3Hit", &hitArray[im]);
    hitTree->GetEntry(im);
    auto& mc2hit = mc2hitVec[im];
    for (int ih = hitArray[im]->size(); ih--;) {
      const auto& hit = (*hitArray[im])[ih];
      uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
      mc2hit.emplace(key, ih);
    }
    if (kineTree && im < kineTree->GetEntries()) {
      kineTree->SetBranchAddress("MCTrack", &mcTrackArray[im]);
      kineTree->GetEntry(im);
    }
  }

  auto& rofArr = *rofRecordsArr;

  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels;
  plabelsArr->copyandflatten(labels);

  int nNoHit = 0, nNoCollision = 0, nOutliers = 0;
  std::array<double, kNLayers> sumRadius{0., 0.};
  std::array<long, kNLayers> nRadius{0, 0};

  // LOOP on : ROFRecord array
  for (unsigned int iROF = 0; iROF < rofArr.size(); ++iROF) {

    const unsigned int rofIndex = rofArr[iROF].getFirstEntry();
    const unsigned int rofNEntries = rofArr[iROF].getNEntries();
    const double rofTimeNS = rofArr[iROF].getBCData().bc2ns();

    // LOOP on : digits array
    for (unsigned int iDigit = rofIndex; iDigit < rofIndex + rofNEntries; iDigit++) {
      if (iDigit % 1000 == 0) {
        std::cout << "Reading digit " << iDigit << " / " << digArr->size() << std::endl;
      }

      const auto& digit = (*digArr)[iDigit];
      Int_t ix = digit.getRow(), iz = digit.getColumn();
      Int_t chipID = digit.getChipIndex();
      Int_t subDetID = gman->getIOTOFLayer(chipID);
      if (subDetID < 0 || subDetID >= kNLayers) {
        continue;
      }

      auto lab = (labels.getLabels(iDigit))[0];
      if (!lab.isValid() || lab.getSourceID() != 0) { // noise or not from the loaded hit file
        continue;
      }

      // collision time
      double tCollNS = rofTimeNS;
      if (context) {
        auto collEntry = collisionTimeNS.find({lab.getSourceID(), lab.getEventID()});
        if (collEntry == collisionTimeNS.end()) {
          nNoCollision++;
          continue;
        }
        tCollNS = collEntry->second;
      }

      // get MC info
      const int trID = lab.getTrackID();
      std::unordered_map<uint64_t, int>* mc2hit = &mc2hitVec[lab.getEventID()];
      uint64_t key = (uint64_t(trID) << 32) + chipID;
      auto hitEntry = mc2hit->find(key);
      if (hitEntry == mc2hit->end()) {
        LOG(error) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
        nNoHit++;
        continue;
      }
      Hit& hit = (*hitArray[lab.getEventID()])[hitEntry->second];

      // Local position of the digit (pixel center) and of the hit (mid point between start and end)
      Float_t xD = 0.f, zD = 0.f;
      seg->detectorToLocal(ix, iz, xD, zD, subDetID);
      o2::math_utils::Point3D<float> locD(xD, 0.f, zD);
      const auto gloD = gman->getMatrixL2G(chipID)(locD);

      auto xyzLocE = gman->getMatrixL2G(chipID) ^ (hit.GetPos());
      auto xyzLocS = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
      const float xH = 0.5f * (xyzLocS.X() + xyzLocE.X());
      const float zH = 0.5f * (xyzLocS.Z() + xyzLocE.Z());

      // Particle kinematics
      const float radius = std::hypot(gloD.X(), gloD.Y());
      float eta = -std::log(std::tan(0.5f * std::atan2(radius, gloD.Z())));
      float pt = -1.f;
      int pdg = 0;
      const auto* mcTracks = mcTrackArray[lab.getEventID()];
      if (mcTracks && trID >= 0 && trID < (int)mcTracks->size()) {
        const auto& mcTrack = (*mcTracks)[trID];
        eta = mcTrack.GetEta();
        pt = mcTrack.GetPt();
        pdg = mcTrack.GetPdgCode();
      }

      // Times (ns)
      const double tTrueNS = hit.GetTime() * sec2ns;       /// true time relative to the collision
      const double tDigNS = digit.getTime() - tCollNS;     /// digit time relative to the collision
      const float dtPs = (tDigNS - tTrueNS) * ns2ps;       /// time residual

      const float dxPixUm = (xH - xD) * cm2um;
      const float dzPixUm = (zH - zD) * cm2um;

      float ntVars[] = {float(chipID), float(subDetID), float(gloD.X()), float(gloD.Y()), float(gloD.Z()), xD, zD, dxPixUm, dzPixUm,
                        eta, pt, float(pdg), float(tTrueNS), float(tDigNS), dtPs};
      nt->Fill(ntVars);

      // Reject digits with an unphysical residual (e.g. a corrupted digit time): kept in the ntuple, excluded from the histograms
      if (std::abs(dtPs) > dtMaxPs) {
        nOutliers++;
        continue;
      }

      hTTrue[subDetID]->Fill(tTrueNS);
      hTDig[subDetID]->Fill(tDigNS);
      hTDigVsTTrue[subDetID]->Fill(tTrueNS, tDigNS);
      hDt[subDetID]->Fill(dtPs);
      hDtVsTTrue[subDetID]->Fill(tTrueNS, dtPs);
      hDtVsXLoc[subDetID]->Fill(xD, dtPs);
      hDtVsZLoc[subDetID]->Fill(zD, dtPs);
      hDtVsZGlo[subDetID]->Fill(gloD.Z(), dtPs);
      pDtInPixel[subDetID]->Fill(dxPixUm, dzPixUm, dtPs);
      sumRadius[subDetID] += radius;
      nRadius[subDetID]++;

      // Time spectra for the selected species in the given pT range
      const bool passSpecies = !kineTree || pdgSel == 0 || std::abs(pdg) == std::abs(pdgSel);
      const bool passPt = !kineTree || (pt > ptMin && pt < ptMax);
      if (passSpecies && passPt) {
        const int etaRegion = std::abs(eta) < etaCut ? 0 : 1;
        hTdcSpectrumDig[subDetID][etaRegion]->Fill(tDigNS / digiPars.tdcBin);
        hTdcSpectrumTrue[subDetID][etaRegion]->Fill(tTrueNS / digiPars.tdcBin);
      }

    } // end loop on digits array

  } // end loop on ROFRecords

  if (nNoHit || nNoCollision || nOutliers) {
    Warning("CheckTimeResolutionIOTOF", "Skipped digits: %d without matching hit, %d without matching collision, %d with |dt| > %g ps",
            nNoHit, nNoCollision, nOutliers, dtMaxPs);
  }

  // Arrival time of a beta = 1 particle at eta = 0, in TDC units
  std::array<float, kNLayers> refTimeTdc{0.f, 0.f};
  for (int il = 0; il < kNLayers; ++il) {
    if (nRadius[il]) {
      refTimeTdc[il] = sumRadius[il] / nRadius[il] / speedOfLightCmNs / digiPars.tdcBin;
    }
  }
  TString header = speciesLabel(kineTree ? pdgSel : 0);
  if (kineTree) {
    header += Form(", %g < #it{p}_{T} < %g GeV/#it{c}", ptMin, ptMax);
  }

  // Histograms created from here on must be written to the output file
  f->cd();

  // One plot per page: each pdf is opened with "[" and closed with "]"
  auto canv = new TCanvas("canv", "", 900, 800);
  auto openPdf = [&](const char* pdf) { canv->Print(Form("%s[", pdf)); };
  auto closePdf = [&](const char* pdf) { canv->Print(Form("%s]", pdf)); };

  // Time spectra split in layers and pseudorapidity regions: digitized time and MC true time
  const char* pdfSpectra = "tf3digits_time_spectra.pdf";
  openPdf(pdfSpectra);
  drawTimeSpectra(canv, hTdcSpectrumDig, refTimeTdc, header, etaCut);
  canv->Print(pdfSpectra, "Title:digitized time spectra");
  drawTimeSpectra(canv, hTdcSpectrumTrue, refTimeTdc, header, etaCut);
  canv->Print(pdfSpectra, "Title:MC true time spectra");
  closePdf(pdfSpectra);

  // Digitized and MC true time distributions
  const char* pdfTime = "tf3digits_time.pdf";
  openPdf(pdfTime);
  for (int il = 0; il < kNLayers; ++il) {
    printPage(canv, pdfTime, hTTrue[il], "", true);
    printPage(canv, pdfTime, hTDig[il], "", true);
    printPage(canv, pdfTime, hTDigVsTTrue[il], "colz", false, true);
  }
  closePdf(pdfTime);

  // Time residual distributions (digit time - true time)
  const char* pdfDt = "tf3digits_dt.pdf";
  openPdf(pdfDt);
  for (int il = 0; il < kNLayers; ++il) {
    fitTimeResidual(hDt[il], kLayerName[il], expectedSigmaPs);
    printPage(canv, pdfDt, hDt[il], "", true);
    printPage(canv, pdfDt, hDtVsTTrue[il], "colz");
    printPage(canv, pdfDt, hDtVsZGlo[il], "colz");
  }
  closePdf(pdfDt);

  // Time residual as a function of the position inside the pixel
  const char* pdfInPixel = "tf3digits_dt_inpixel.pdf";
  openPdf(pdfInPixel);
  for (int il = 0; il < kNLayers; ++il) {
    printPage(canv, pdfInPixel, pDtInPixel[il], "colz");
    auto hSpread = profileToSpread(pDtInPixel[il], Form("h_sigmadt_vs_inpixel_%s", kLayerName[il]),
                                   Form("%s: RMS of the time residual vs position in the pixel;x_{hit} - x_{pixel} (#mum);z_{hit} - z_{pixel} (#mum);RMS(t_{digit} - t_{true}) (ps)", kLayerName[il]));
    printPage(canv, pdfInPixel, hSpread, "colz");
  }
  closePdf(pdfInPixel);

  f->Write();
  f->Close();
}
