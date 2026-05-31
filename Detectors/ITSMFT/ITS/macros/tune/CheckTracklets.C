// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \file CheckTracklets.C
/// \brief Inspect ITS tune-mode tracklet diagnostics.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <TCanvas.h>
#include <TClass.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TKey.h>
#include <TLegend.h>
#include <TMath.h>
#include <TROOT.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TVirtualPad.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2D.h>
#endif

namespace
{
struct TransitionStats {
  long long all{0};
  long long ok{0};
  long long prim{0};
};

struct TransitionPlots {
  TH1D* hDPhiAll{nullptr};
  TH1D* hDPhiOk{nullptr};
  TH1D* hDeltaZEventOk{nullptr};
  TH1D* hDeltaZEventPrim{nullptr};
  TH1D* hDPhiTruthFraction{nullptr};
};

struct TransitionIterationPlots {
  TH1D* hPhiAll{nullptr};
  TH1D* hPhiOk{nullptr};
  TH1D* hTglAll{nullptr};
  TH1D* hTglOk{nullptr};
  TH1D* hDXYOk{nullptr};
  TH1D* hDXYPrim{nullptr};
  TH1D* hDZOk{nullptr};
  TH1D* hDZPrim{nullptr};
  TH1D* hDRAll{nullptr};
  TH1D* hDROk{nullptr};
  TH1D* hDZGeomAll{nullptr};
  TH1D* hDZGeomOk{nullptr};
  TH1D* hDPhiAll{nullptr};
  TH1D* hDPhiOk{nullptr};
  TH1D* hTglEventOk{nullptr};
  TH1D* hDeltaZEventOk{nullptr};
  TH1D* hDeltaZEventPrim{nullptr};
};

struct IterationData {
  int iteration{-1};
  TTree* tree{nullptr};
  std::map<std::pair<int, int>, TransitionStats> transitionStats;
  TH1D* hPhiAll{nullptr};
  TH1D* hPhiOk{nullptr};
  TH1D* hTglAll{nullptr};
  TH1D* hTglOk{nullptr};
  TH1D* hDXYOk{nullptr};
  TH1D* hDXYPrim{nullptr};
  TH1D* hDZOk{nullptr};
  TH1D* hDZPrim{nullptr};
  TH1D* hDRAll{nullptr};
  TH1D* hDROk{nullptr};
  TH1D* hDZGeomAll{nullptr};
  TH1D* hDZGeomOk{nullptr};
  TH1D* hDPhiAll{nullptr};
  TH1D* hDPhiOk{nullptr};
  TH1D* hTglEventOk{nullptr};
  TH1D* hDeltaZEventOk{nullptr};
  TH1D* hDeltaZEventPrim{nullptr};
  TH2D* hDXYVsPhi{nullptr};
  TH2D* hDZVsTgl{nullptr};
};

struct FormulaSet {
  std::unique_ptr<TTreeFormula> from;
  std::unique_ptr<TTreeFormula> to;
  std::unique_ptr<TTreeFormula> phi;
  std::unique_ptr<TTreeFormula> tgl;
  std::unique_ptr<TTreeFormula> dr;
  std::unique_ptr<TTreeFormula> dz;
  std::unique_ptr<TTreeFormula> dPhi;
  std::unique_ptr<TTreeFormula> ok;
  std::unique_ptr<TTreeFormula> prim;
  std::unique_ptr<TTreeFormula> dXY;
  std::unique_ptr<TTreeFormula> dZ;
  std::unique_ptr<TTreeFormula> tglEvent;
  std::unique_ptr<TTreeFormula> deltaZEvent;
};

bool parseIteration(const TString& name, int& iteration)
{
  if (!name.BeginsWith("trklt_")) {
    return false;
  }
  const TString suffix = name(6, name.Length() - 6);
  if (suffix.IsNull()) {
    return false;
  }
  for (int i = 0; i < suffix.Length(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  iteration = suffix.Atoi();
  return true;
}

std::unique_ptr<TTreeFormula> makeFormula(TTree* tree, const char* name, const char* expression)
{
  auto formula = std::make_unique<TTreeFormula>(name, expression, tree);
  if (formula->GetNdim() <= 0) {
    std::cerr << "CheckTracklets: missing or unreadable expression '" << expression << "' in tree '" << tree->GetName() << "'" << std::endl;
    return nullptr;
  }
  return formula;
}

bool makeFormulas(TTree* tree, FormulaSet& formulas)
{
  formulas.from = makeFormula(tree, "fromFormula", "from");
  formulas.to = makeFormula(tree, "toFormula", "to");
  formulas.phi = makeFormula(tree, "phiFormula", "trklt.phi");
  formulas.tgl = makeFormula(tree, "tglFormula", "trklt.tgl");
  formulas.dr = makeFormula(tree, "drFormula", "trklt.dr");
  formulas.dz = makeFormula(tree, "dzFormula", "trklt.dz");
  formulas.dPhi = makeFormula(tree, "dPhiFormula", "trklt.dPhi");
  formulas.ok = makeFormula(tree, "okFormula", "trklt.ok");
  formulas.prim = makeFormula(tree, "primFormula", "trklt.prim");
  formulas.dXY = makeFormula(tree, "dXYFormula", "trklt.dXY");
  formulas.dZ = makeFormula(tree, "dZFormula", "trklt.dZ");
  formulas.tglEvent = makeFormula(tree, "tglEventFormula", "trklt.tglEvent");
  formulas.deltaZEvent = makeFormula(tree, "deltaZEventFormula", "trklt.deltaZEvent");
  return formulas.from && formulas.to && formulas.phi && formulas.tgl && formulas.dr && formulas.dz && formulas.dPhi && formulas.ok && formulas.prim && formulas.dXY && formulas.dZ && formulas.tglEvent && formulas.deltaZEvent;
}

void updateFormulas(FormulaSet& formulas)
{
  formulas.from->UpdateFormulaLeaves();
  formulas.to->UpdateFormulaLeaves();
  formulas.phi->UpdateFormulaLeaves();
  formulas.tgl->UpdateFormulaLeaves();
  formulas.dr->UpdateFormulaLeaves();
  formulas.dz->UpdateFormulaLeaves();
  formulas.dPhi->UpdateFormulaLeaves();
  formulas.ok->UpdateFormulaLeaves();
  formulas.prim->UpdateFormulaLeaves();
  formulas.dXY->UpdateFormulaLeaves();
  formulas.dZ->UpdateFormulaLeaves();
  formulas.tglEvent->UpdateFormulaLeaves();
  formulas.deltaZEvent->UpdateFormulaLeaves();
}

void bookIterationHistograms(IterationData& data)
{
  const TString prefix = Form("iter%d", data.iteration);
  data.hPhiAll = new TH1D(Form("%s_phi_all", prefix.Data()), Form("Iteration %d;#phi;tracklets", data.iteration), 160, -TMath::Pi(), TMath::Pi());
  data.hPhiOk = new TH1D(Form("%s_phi_ok", prefix.Data()), Form("Iteration %d truth matched;#phi;tracklets", data.iteration), 160, -TMath::Pi(), TMath::Pi());
  data.hTglAll = new TH1D(Form("%s_tgl_all", prefix.Data()), Form("Iteration %d;tan(#lambda);tracklets", data.iteration), 200, -20., 20.);
  data.hTglOk = new TH1D(Form("%s_tgl_ok", prefix.Data()), Form("Iteration %d truth matched;tan(#lambda);tracklets", data.iteration), 200, -20., 20.);
  data.hDXYOk = new TH1D(Form("%s_dxy_ok", prefix.Data()), Form("Iteration %d truth matched;d_{XY} to true event (cm);tracklets", data.iteration), 200, 0., 3.);
  data.hDXYPrim = new TH1D(Form("%s_dxy_prim", prefix.Data()), Form("Iteration %d primary truth;d_{XY} to true event (cm);tracklets", data.iteration), 200, 0., 3.);
  data.hDZOk = new TH1D(Form("%s_dz_ok", prefix.Data()), Form("Iteration %d truth matched;d_{Z} to true event (cm);tracklets", data.iteration), 240, -30., 30.);
  data.hDZPrim = new TH1D(Form("%s_dz_prim", prefix.Data()), Form("Iteration %d primary truth;d_{Z} to true event (cm);tracklets", data.iteration), 240, -30., 30.);
  data.hDXYVsPhi = new TH2D(Form("%s_dxy_vs_phi", prefix.Data()), Form("Iteration %d truth matched;#phi;d_{XY} to true event (cm)", data.iteration), 120, -TMath::Pi(), TMath::Pi(), 120, 0., 3.);
  data.hDZVsTgl = new TH2D(Form("%s_dz_vs_tgl", prefix.Data()), Form("Iteration %d truth matched;tan(#lambda);d_{Z} to true event (cm)", data.iteration), 120, -20., 20., 120, -30., 30.);
  data.hDRAll = new TH1D(Form("%s_dr_all", prefix.Data()), Form("Iteration %d;#Deltar (cm);tracklets", data.iteration), 160, -5., 40.);
  data.hDROk = new TH1D(Form("%s_dr_ok", prefix.Data()), Form("Iteration %d truth matched;#Deltar (cm);tracklets", data.iteration), 160, -5., 40.);
  data.hDZGeomAll = new TH1D(Form("%s_dz_geom_all", prefix.Data()), Form("Iteration %d;#Deltaz (cm);tracklets", data.iteration), 240, -30., 30.);
  data.hDZGeomOk = new TH1D(Form("%s_dz_geom_ok", prefix.Data()), Form("Iteration %d truth matched;#Deltaz (cm);tracklets", data.iteration), 240, -30., 30.);
  data.hDPhiAll = new TH1D(Form("%s_dphi_all", prefix.Data()), Form("Iteration %d;#Delta#phi;tracklets", data.iteration), 160, 0., 0.2);
  data.hDPhiOk = new TH1D(Form("%s_dphi_ok", prefix.Data()), Form("Iteration %d truth matched;#Delta#phi;tracklets", data.iteration), 160, 0., 0.2);
  data.hTglEventOk = new TH1D(Form("%s_tgl_event_ok", prefix.Data()), Form("Iteration %d truth matched;tan(#lambda) from true event;tracklets", data.iteration), 200, -20., 20.);
  data.hDeltaZEventOk = new TH1D(Form("%s_delta_z_event_ok", prefix.Data()), Form("Iteration %d truth matched;#Deltaz from true event (cm);tracklets", data.iteration), 240, 0., 30.);
  data.hDeltaZEventPrim = new TH1D(Form("%s_delta_z_event_prim", prefix.Data()), Form("Iteration %d primary truth;#Deltaz from true event (cm);tracklets", data.iteration), 240, 0., 30.);

  for (auto* hist : {data.hPhiAll, data.hPhiOk, data.hTglAll, data.hTglOk, data.hDXYOk, data.hDXYPrim, data.hDZOk, data.hDZPrim, data.hDRAll, data.hDROk, data.hDZGeomAll, data.hDZGeomOk, data.hDPhiAll, data.hDPhiOk, data.hTglEventOk, data.hDeltaZEventOk, data.hDeltaZEventPrim}) {
    hist->Sumw2();
  }
}

TransitionPlots& getTransitionPlots(std::map<std::pair<int, int>, TransitionPlots>& plots, const std::pair<int, int>& transition)
{
  auto& transitionPlots = plots[transition];
  if (transitionPlots.hDPhiAll) {
    return transitionPlots;
  }

  const TString prefix = Form("L%d_L%d", transition.first, transition.second);
  transitionPlots.hDPhiAll = new TH1D(Form("%s_dphi_all", prefix.Data()), Form("L%d-L%d;#Delta#phi;tracklets", transition.first, transition.second), 160, 0., 0.2);
  transitionPlots.hDPhiOk = new TH1D(Form("%s_dphi_ok", prefix.Data()), Form("L%d-L%d truth matched;#Delta#phi;tracklets", transition.first, transition.second), 160, 0., 0.2);
  transitionPlots.hDeltaZEventOk = new TH1D(Form("%s_delta_z_event_ok", prefix.Data()), Form("L%d-L%d truth matched;#Deltaz from true event (cm);tracklets", transition.first, transition.second), 240, 0., 30.);
  transitionPlots.hDeltaZEventPrim = new TH1D(Form("%s_delta_z_event_prim", prefix.Data()), Form("L%d-L%d primary truth;#Deltaz from true event (cm);tracklets", transition.first, transition.second), 240, 0., 30.);

  for (auto* hist : {transitionPlots.hDPhiAll, transitionPlots.hDPhiOk, transitionPlots.hDeltaZEventOk, transitionPlots.hDeltaZEventPrim}) {
    hist->Sumw2();
  }
  return transitionPlots;
}

TransitionIterationPlots& getTransitionIterationPlots(std::map<std::pair<int, int>, std::map<int, TransitionIterationPlots>>& plots, const std::pair<int, int>& transition, const int iteration)
{
  auto& iterationPlots = plots[transition][iteration];
  if (iterationPlots.hPhiAll) {
    return iterationPlots;
  }

  const TString prefix = Form("L%d_L%d_iter%d", transition.first, transition.second, iteration);
  const TString title = Form("L%d-L%d iteration %d", transition.first, transition.second, iteration);
  iterationPlots.hPhiAll = new TH1D(Form("%s_phi_all", prefix.Data()), Form("%s;#phi;tracklets", title.Data()), 160, -TMath::Pi(), TMath::Pi());
  iterationPlots.hPhiOk = new TH1D(Form("%s_phi_ok", prefix.Data()), Form("%s truth matched;#phi;tracklets", title.Data()), 160, -TMath::Pi(), TMath::Pi());
  iterationPlots.hTglAll = new TH1D(Form("%s_tgl_all", prefix.Data()), Form("%s;tan(#lambda);tracklets", title.Data()), 200, -20., 20.);
  iterationPlots.hTglOk = new TH1D(Form("%s_tgl_ok", prefix.Data()), Form("%s truth matched;tan(#lambda);tracklets", title.Data()), 200, -20., 20.);
  iterationPlots.hDXYOk = new TH1D(Form("%s_dxy_ok", prefix.Data()), Form("%s truth matched;d_{XY} to true event (cm);tracklets", title.Data()), 200, 0., 3.);
  iterationPlots.hDXYPrim = new TH1D(Form("%s_dxy_prim", prefix.Data()), Form("%s primary truth;d_{XY} to true event (cm);tracklets", title.Data()), 200, 0., 3.);
  iterationPlots.hDZOk = new TH1D(Form("%s_dz_ok", prefix.Data()), Form("%s truth matched;d_{Z} to true event (cm);tracklets", title.Data()), 240, -30., 30.);
  iterationPlots.hDZPrim = new TH1D(Form("%s_dz_prim", prefix.Data()), Form("%s primary truth;d_{Z} to true event (cm);tracklets", title.Data()), 240, -30., 30.);
  iterationPlots.hDRAll = new TH1D(Form("%s_dr_all", prefix.Data()), Form("%s;#Deltar (cm);tracklets", title.Data()), 160, -5., 40.);
  iterationPlots.hDROk = new TH1D(Form("%s_dr_ok", prefix.Data()), Form("%s truth matched;#Deltar (cm);tracklets", title.Data()), 160, -5., 40.);
  iterationPlots.hDZGeomAll = new TH1D(Form("%s_dz_geom_all", prefix.Data()), Form("%s;#Deltaz (cm);tracklets", title.Data()), 240, -30., 30.);
  iterationPlots.hDZGeomOk = new TH1D(Form("%s_dz_geom_ok", prefix.Data()), Form("%s truth matched;#Deltaz (cm);tracklets", title.Data()), 240, -30., 30.);
  iterationPlots.hDPhiAll = new TH1D(Form("%s_dphi_all", prefix.Data()), Form("%s;#Delta#phi;tracklets", title.Data()), 160, 0., 0.2);
  iterationPlots.hDPhiOk = new TH1D(Form("%s_dphi_ok", prefix.Data()), Form("%s truth matched;#Delta#phi;tracklets", title.Data()), 160, 0., 0.2);
  iterationPlots.hTglEventOk = new TH1D(Form("%s_tgl_event_ok", prefix.Data()), Form("%s truth matched;tan(#lambda) from true event;tracklets", title.Data()), 200, -20., 20.);
  iterationPlots.hDeltaZEventOk = new TH1D(Form("%s_delta_z_event_ok", prefix.Data()), Form("%s truth matched;#Deltaz from true event (cm);tracklets", title.Data()), 240, 0., 30.);
  iterationPlots.hDeltaZEventPrim = new TH1D(Form("%s_delta_z_event_prim", prefix.Data()), Form("%s primary truth;#Deltaz from true event (cm);tracklets", title.Data()), 240, 0., 30.);

  for (auto* hist : {iterationPlots.hPhiAll, iterationPlots.hPhiOk, iterationPlots.hTglAll, iterationPlots.hTglOk, iterationPlots.hDXYOk, iterationPlots.hDXYPrim, iterationPlots.hDZOk, iterationPlots.hDZPrim, iterationPlots.hDRAll, iterationPlots.hDROk, iterationPlots.hDZGeomAll, iterationPlots.hDZGeomOk, iterationPlots.hDPhiAll, iterationPlots.hDPhiOk, iterationPlots.hTglEventOk, iterationPlots.hDeltaZEventOk, iterationPlots.hDeltaZEventPrim}) {
    hist->Sumw2();
  }
  return iterationPlots;
}

void setLine(TH1* hist, int color, int style = 1)
{
  hist->SetLineColor(color);
  hist->SetMarkerColor(color);
  hist->SetLineStyle(style);
  hist->SetLineWidth(2);
}

TH1D* normalizedClone(const TH1D* source, const char* name)
{
  auto* clone = static_cast<TH1D*>(source->Clone(name));
  clone->SetDirectory(nullptr);
  if (clone->Integral() > 0.) {
    clone->Scale(1. / clone->Integral());
  }
  return clone;
}

void drawPair(TH1D* all, TH1D* selected, const char* allLabel, const char* selectedLabel)
{
  setLine(all, kBlack);
  setLine(selected, kRed + 1);
  all->Draw("hist");
  selected->Draw("hist same");
  auto* legend = new TLegend(0.58, 0.74, 0.88, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(all, Form("%s: %.0f", allLabel, all->GetEntries()), "l");
  legend->AddEntry(selected, Form("%s: %.0f", selectedLabel, selected->GetEntries()), "l");
  legend->Draw();
}

void drawSingle(TH1D* hist, int color)
{
  setLine(hist, color);
  hist->Draw("hist");
}

void drawIterationBreakdown(const std::map<int, TransitionIterationPlots>& plots, TH1D* TransitionIterationPlots::* member, const char* title)
{
  static const std::array<int, 10> colors = {kBlack, kRed + 1, kBlue + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2, kViolet + 5, kAzure + 7, kGray + 2};
  bool first = true;
  auto* legend = new TLegend(0.62, 0.62, 0.88, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);

  size_t i = 0;
  for (const auto& [iteration, iterationPlots] : plots) {
    auto* hist = iterationPlots.*member;
    setLine(hist, colors[i % colors.size()]);
    hist->SetTitle(title);
    hist->Draw(first ? "hist" : "hist same");
    first = false;
    legend->AddEntry(hist, Form("iteration %d: %.0f", iteration, hist->GetEntries()), "l");
    ++i;
  }
  legend->Draw();
}

void drawOverlay(const std::vector<IterationData>& iterations, bool drawDXY)
{
  static const std::array<int, 10> colors = {kBlack, kRed + 1, kBlue + 1, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2, kViolet + 5, kAzure + 7, kGray + 2};
  bool first = true;
  auto* legend = new TLegend(0.62, 0.62, 0.88, 0.88);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);

  for (size_t i = 0; i < iterations.size(); ++i) {
    const auto* source = drawDXY ? iterations[i].hDXYOk : iterations[i].hDZOk;
    auto* hist = normalizedClone(source, Form("%s_norm", source->GetName()));
    setLine(hist, colors[i % colors.size()]);
    hist->SetTitle(drawDXY ? "Truth matched d_{XY};d_{XY} to true event (cm);normalized tracklets" : "Truth matched d_{Z};d_{Z} to true event (cm);normalized tracklets");
    hist->Draw(first ? "hist" : "hist same");
    first = false;
    legend->AddEntry(hist, Form("iteration %d", iterations[i].iteration), "l");
  }
  legend->Draw();
}

void writeIterationHistograms(TDirectory* directory, const IterationData& data)
{
  directory->cd();
  data.hPhiAll->Write();
  data.hPhiOk->Write();
  data.hTglAll->Write();
  data.hTglOk->Write();
  data.hDXYOk->Write();
  data.hDXYPrim->Write();
  data.hDZOk->Write();
  data.hDZPrim->Write();
  data.hDRAll->Write();
  data.hDROk->Write();
  data.hDZGeomAll->Write();
  data.hDZGeomOk->Write();
  data.hDPhiAll->Write();
  data.hDPhiOk->Write();
  data.hTglEventOk->Write();
  data.hDeltaZEventOk->Write();
  data.hDeltaZEventPrim->Write();
  data.hDXYVsPhi->Write();
  data.hDZVsTgl->Write();
}

void writeTransitionIterationHistograms(TDirectory* directory, const TransitionIterationPlots& plots)
{
  directory->cd();
  plots.hPhiAll->Write();
  plots.hPhiOk->Write();
  plots.hTglAll->Write();
  plots.hTglOk->Write();
  plots.hDXYOk->Write();
  plots.hDXYPrim->Write();
  plots.hDZOk->Write();
  plots.hDZPrim->Write();
  plots.hDRAll->Write();
  plots.hDROk->Write();
  plots.hDZGeomAll->Write();
  plots.hDZGeomOk->Write();
  plots.hDPhiAll->Write();
  plots.hDPhiOk->Write();
  plots.hTglEventOk->Write();
  plots.hDeltaZEventOk->Write();
  plots.hDeltaZEventPrim->Write();
}

} // namespace

void CheckTracklets(std::string inputFile = "its_tune.root",
                    std::string outputRoot = "CheckTracklets.root",
                    std::string outputPdf = "CheckTracklets.pdf")
{
  TH1::AddDirectory(kFALSE);
  gStyle->SetOptStat(10);

  std::unique_ptr<TFile> input(TFile::Open(inputFile.c_str(), "READ"));
  if (!input || input->IsZombie()) {
    std::cerr << "CheckTracklets: cannot open input file '" << inputFile << "'" << std::endl;
    return;
  }

  std::vector<IterationData> iterations;
  TIter nextKey(input->GetListOfKeys());
  while (auto* key = static_cast<TKey*>(nextKey())) {
    const TString name = key->GetName();
    int iteration = -1;
    if (!parseIteration(name, iteration)) {
      continue;
    }
    auto* klass = gROOT->GetClass(key->GetClassName());
    if (!klass || !klass->InheritsFrom(TTree::Class())) {
      continue;
    }
    auto* tree = input->Get<TTree>(name);
    if (!tree) {
      continue;
    }
    iterations.push_back({iteration, tree});
  }

  std::sort(iterations.begin(), iterations.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.iteration < rhs.iteration;
  });

  if (iterations.empty()) {
    std::cerr << "CheckTracklets: no trklt_<iteration> trees found in '" << inputFile << "'" << std::endl;
    return;
  }

  std::set<std::pair<int, int>> transitions;
  std::map<std::pair<int, int>, TransitionPlots> transitionPlots;
  std::map<std::pair<int, int>, std::map<int, TransitionIterationPlots>> transitionIterationPlots;
  for (auto& data : iterations) {
    FormulaSet formulas;
    if (!makeFormulas(data.tree, formulas)) {
      return;
    }
    bookIterationHistograms(data);
    const auto entries = data.tree->GetEntries();
    for (Long64_t entry = 0; entry < entries; ++entry) {
      data.tree->LoadTree(entry);
      data.tree->GetEntry(entry);
      updateFormulas(formulas);

      const int from = static_cast<int>(formulas.from->EvalInstance());
      const int to = static_cast<int>(formulas.to->EvalInstance());
      const float phi = static_cast<float>(formulas.phi->EvalInstance());
      const float tgl = static_cast<float>(formulas.tgl->EvalInstance());
      const float dr = static_cast<float>(formulas.dr->EvalInstance());
      const float dz = static_cast<float>(formulas.dz->EvalInstance());
      const float dPhi = static_cast<float>(formulas.dPhi->EvalInstance());
      const bool ok = formulas.ok->EvalInstance() != 0.;
      const bool prim = formulas.prim->EvalInstance() != 0.;
      const float dXY = static_cast<float>(formulas.dXY->EvalInstance());
      const float dZ = static_cast<float>(formulas.dZ->EvalInstance());
      const float tglEvent = static_cast<float>(formulas.tglEvent->EvalInstance());
      const float deltaZEvent = static_cast<float>(formulas.deltaZEvent->EvalInstance());

      const auto transition = std::make_pair(from, to);
      auto& stats = data.transitionStats[transition];
      auto& plots = getTransitionPlots(transitionPlots, transition);
      auto& perIterationPlots = getTransitionIterationPlots(transitionIterationPlots, transition, data.iteration);
      ++stats.all;
      transitions.insert(transition);

      data.hPhiAll->Fill(phi);
      data.hTglAll->Fill(tgl);
      data.hDRAll->Fill(dr);
      data.hDZGeomAll->Fill(dz);
      data.hDPhiAll->Fill(dPhi);
      plots.hDPhiAll->Fill(dPhi);
      perIterationPlots.hPhiAll->Fill(phi);
      perIterationPlots.hTglAll->Fill(tgl);
      perIterationPlots.hDRAll->Fill(dr);
      perIterationPlots.hDZGeomAll->Fill(dz);
      perIterationPlots.hDPhiAll->Fill(dPhi);

      if (!ok) {
        continue;
      }
      ++stats.ok;
      data.hPhiOk->Fill(phi);
      data.hTglOk->Fill(tgl);
      data.hDXYOk->Fill(dXY);
      data.hDZOk->Fill(dZ);
      data.hDROk->Fill(dr);
      data.hDZGeomOk->Fill(dz);
      data.hDPhiOk->Fill(dPhi);
      data.hTglEventOk->Fill(tglEvent);
      data.hDeltaZEventOk->Fill(deltaZEvent);
      data.hDXYVsPhi->Fill(phi, dXY);
      data.hDZVsTgl->Fill(tgl, dZ);
      plots.hDPhiOk->Fill(dPhi);
      plots.hDeltaZEventOk->Fill(deltaZEvent);
      perIterationPlots.hPhiOk->Fill(phi);
      perIterationPlots.hTglOk->Fill(tgl);
      perIterationPlots.hDXYOk->Fill(dXY);
      perIterationPlots.hDZOk->Fill(dZ);
      perIterationPlots.hDROk->Fill(dr);
      perIterationPlots.hDZGeomOk->Fill(dz);
      perIterationPlots.hDPhiOk->Fill(dPhi);
      perIterationPlots.hTglEventOk->Fill(tglEvent);
      perIterationPlots.hDeltaZEventOk->Fill(deltaZEvent);
      if (prim) {
        ++stats.prim;
        data.hDXYPrim->Fill(dXY);
        data.hDZPrim->Fill(dZ);
        data.hDeltaZEventPrim->Fill(deltaZEvent);
        plots.hDeltaZEventPrim->Fill(deltaZEvent);
        perIterationPlots.hDXYPrim->Fill(dXY);
        perIterationPlots.hDZPrim->Fill(dZ);
        perIterationPlots.hDeltaZEventPrim->Fill(deltaZEvent);
      }
    }
  }

  if (transitions.empty()) {
    std::cerr << "CheckTracklets: no tracklet entries found in '" << inputFile << "'" << std::endl;
    return;
  }

  auto* hCounts = new TH2D("hTrackletCounts", "Tracklet counts;iteration;transition", iterations.size(), 0, iterations.size(), transitions.size(), 0, transitions.size());
  auto* hTruthFraction = new TH2D("hTruthMatchedFraction", "Truth matched fraction;iteration;transition", iterations.size(), 0, iterations.size(), transitions.size(), 0, transitions.size());
  auto* hPrimaryFraction = new TH2D("hPrimaryFraction", "Primary fraction among truth matched;iteration;transition", iterations.size(), 0, iterations.size(), transitions.size(), 0, transitions.size());

  int ybin = 1;
  for (const auto& transition : transitions) {
    const auto label = Form("L%d-L%d", transition.first, transition.second);
    hCounts->GetYaxis()->SetBinLabel(ybin, label);
    hTruthFraction->GetYaxis()->SetBinLabel(ybin, label);
    hPrimaryFraction->GetYaxis()->SetBinLabel(ybin, label);
    ++ybin;
  }

  for (size_t i = 0; i < iterations.size(); ++i) {
    const int xbin = static_cast<int>(i) + 1;
    const auto label = Form("%d", iterations[i].iteration);
    hCounts->GetXaxis()->SetBinLabel(xbin, label);
    hTruthFraction->GetXaxis()->SetBinLabel(xbin, label);
    hPrimaryFraction->GetXaxis()->SetBinLabel(xbin, label);

    ybin = 1;
    for (const auto& transition : transitions) {
      const auto found = iterations[i].transitionStats.find(transition);
      if (found != iterations[i].transitionStats.end()) {
        const auto& stats = found->second;
        hCounts->SetBinContent(xbin, ybin, stats.all);
        if (stats.all > 0) {
          hTruthFraction->SetBinContent(xbin, ybin, static_cast<double>(stats.ok) / static_cast<double>(stats.all));
        }
        if (stats.ok > 0) {
          hPrimaryFraction->SetBinContent(xbin, ybin, static_cast<double>(stats.prim) / static_cast<double>(stats.ok));
        }
      }
      ++ybin;
    }
  }

  for (auto& [transition, plots] : transitionPlots) {
    plots.hDPhiTruthFraction = static_cast<TH1D*>(plots.hDPhiOk->Clone(Form("L%d_L%d_dphi_truth_fraction", transition.first, transition.second)));
    plots.hDPhiTruthFraction->SetTitle(Form("L%d-L%d truth matched fraction;#Delta#phi;ok / all", transition.first, transition.second));
    plots.hDPhiTruthFraction->Divide(plots.hDPhiOk, plots.hDPhiAll, 1., 1., "B");
    plots.hDPhiTruthFraction->SetMinimum(0.);
    plots.hDPhiTruthFraction->SetMaximum(1.05);
  }

  std::unique_ptr<TFile> output(TFile::Open(outputRoot.c_str(), "RECREATE"));
  if (!output || output->IsZombie()) {
    std::cerr << "CheckTracklets: cannot create output file '" << outputRoot << "'" << std::endl;
    return;
  }

  auto* summaryDir = output->mkdir("summary");
  summaryDir->cd();
  hCounts->Write();
  hTruthFraction->Write();
  hPrimaryFraction->Write();
  for (const auto& [transition, plots] : transitionPlots) {
    auto* transitionDir = summaryDir->mkdir(Form("transition_L%d_L%d", transition.first, transition.second));
    transitionDir->cd();
    plots.hDPhiAll->Write();
    plots.hDPhiOk->Write();
    plots.hDPhiTruthFraction->Write();
    plots.hDeltaZEventOk->Write();
    plots.hDeltaZEventPrim->Write();
    const auto transitionIterations = transitionIterationPlots.find(transition);
    if (transitionIterations != transitionIterationPlots.end()) {
      auto* iterationsDir = transitionDir->mkdir("iterations");
      for (const auto& [iteration, iterationPlots] : transitionIterations->second) {
        auto* iterationDir = iterationsDir->mkdir(Form("iteration_%d", iteration));
        writeTransitionIterationHistograms(iterationDir, iterationPlots);
      }
    }
  }
  for (const auto& data : iterations) {
    auto* iterDir = output->mkdir(Form("iteration_%d", data.iteration));
    writeIterationHistograms(iterDir, data);
  }
  output->Close();

  TCanvas canvas("cCheckTracklets", "ITS tune tracklets", 1800, 1100);
  canvas.Print((outputPdf + "[").c_str());

  canvas.Clear();
  canvas.Divide(2, 2);
  canvas.cd(1);
  gPad->SetRightMargin(0.16);
  hCounts->Draw("colz text");
  canvas.cd(2);
  gPad->SetRightMargin(0.16);
  hTruthFraction->SetMinimum(0.);
  hTruthFraction->SetMaximum(1.);
  hTruthFraction->Draw("colz text");
  canvas.cd(3);
  gPad->SetRightMargin(0.16);
  hPrimaryFraction->SetMinimum(0.);
  hPrimaryFraction->SetMaximum(1.);
  hPrimaryFraction->Draw("colz text");
  canvas.cd(4);
  drawOverlay(iterations, true);
  canvas.Print(outputPdf.c_str());

  canvas.Clear();
  canvas.Divide(2, 1);
  canvas.cd(1);
  drawOverlay(iterations, true);
  canvas.cd(2);
  drawOverlay(iterations, false);
  canvas.Print(outputPdf.c_str());

  for (const auto& [transition, plots] : transitionPlots) {
    canvas.Clear();
    canvas.Divide(2, 1);
    canvas.cd(1);
    drawSingle(plots.hDPhiTruthFraction, kBlue + 1);
    canvas.cd(2);
    drawPair(plots.hDeltaZEventOk, plots.hDeltaZEventPrim, "truth", "primary");
    canvas.Print(outputPdf.c_str());
  }

  for (const auto& [transition, plots] : transitionIterationPlots) {
    canvas.Clear();
    canvas.Divide(3, 2);
    canvas.cd(1);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hPhiOk, Form("L%d-L%d truth matched by iteration;#phi;tracklets", transition.first, transition.second));
    canvas.cd(2);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hTglOk, Form("L%d-L%d truth matched by iteration;tan(#lambda);tracklets", transition.first, transition.second));
    canvas.cd(3);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDXYOk, Form("L%d-L%d truth matched by iteration;d_{XY} to true event (cm);tracklets", transition.first, transition.second));
    canvas.cd(4);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDZOk, Form("L%d-L%d truth matched by iteration;d_{Z} to true event (cm);tracklets", transition.first, transition.second));
    canvas.cd(5);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDPhiOk, Form("L%d-L%d truth matched by iteration;#Delta#phi;tracklets", transition.first, transition.second));
    canvas.cd(6);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDeltaZEventOk, Form("L%d-L%d truth matched by iteration;#Deltaz from true event (cm);tracklets", transition.first, transition.second));
    canvas.Print(outputPdf.c_str());

    canvas.Clear();
    canvas.Divide(3, 2);
    canvas.cd(1);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDRAll, Form("L%d-L%d all tracklets by iteration;#Deltar (cm);tracklets", transition.first, transition.second));
    canvas.cd(2);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDROk, Form("L%d-L%d truth matched by iteration;#Deltar (cm);tracklets", transition.first, transition.second));
    canvas.cd(3);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDZGeomAll, Form("L%d-L%d all tracklets by iteration;#Deltaz (cm);tracklets", transition.first, transition.second));
    canvas.cd(4);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDZGeomOk, Form("L%d-L%d truth matched by iteration;#Deltaz (cm);tracklets", transition.first, transition.second));
    canvas.cd(5);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hDPhiAll, Form("L%d-L%d all tracklets by iteration;#Delta#phi;tracklets", transition.first, transition.second));
    canvas.cd(6);
    drawIterationBreakdown(plots, &TransitionIterationPlots::hTglEventOk, Form("L%d-L%d truth matched by iteration;tan(#lambda) from true event;tracklets", transition.first, transition.second));
    canvas.Print(outputPdf.c_str());
  }

  for (auto& data : iterations) {
    canvas.Clear();
    canvas.Divide(3, 2);
    canvas.cd(1);
    drawPair(data.hPhiAll, data.hPhiOk, "all", "truth");
    canvas.cd(2);
    drawPair(data.hTglAll, data.hTglOk, "all", "truth");
    canvas.cd(3);
    drawPair(data.hDXYOk, data.hDXYPrim, "truth", "primary");
    canvas.cd(4);
    drawPair(data.hDZOk, data.hDZPrim, "truth", "primary");
    canvas.cd(5);
    gPad->SetRightMargin(0.16);
    data.hDXYVsPhi->Draw("colz");
    canvas.cd(6);
    gPad->SetRightMargin(0.16);
    data.hDZVsTgl->Draw("colz");
    canvas.Print(outputPdf.c_str());

    canvas.Clear();
    canvas.Divide(3, 2);
    canvas.cd(1);
    drawPair(data.hDRAll, data.hDROk, "all", "truth");
    canvas.cd(2);
    drawPair(data.hDZGeomAll, data.hDZGeomOk, "all", "truth");
    canvas.cd(3);
    drawPair(data.hDPhiAll, data.hDPhiOk, "all", "truth");
    canvas.cd(4);
    drawPair(data.hDeltaZEventOk, data.hDeltaZEventPrim, "truth", "primary");
    canvas.cd(5);
    drawSingle(data.hTglEventOk, kBlue + 1);
    canvas.Print(outputPdf.c_str());
  }

  canvas.Print((outputPdf + "]").c_str());

  std::cout << "CheckTracklets: wrote " << outputRoot << " and " << outputPdf << std::endl;
}
