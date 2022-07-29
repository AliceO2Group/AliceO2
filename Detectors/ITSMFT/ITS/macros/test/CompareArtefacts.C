#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TMath.h"
#include "TString.h"
#include "TStyle.h"

#include <string>
#include <vector>
#endif

using std::string;
using std::vector;

void CompareArtefacts(const string cpu_file = "artefacts_tf.root", const string gpu_file = "artefacts_tf_gpu.root")
{
  gStyle->SetOptStat(0);

  auto f_cpu = TFile::Open(cpu_file.data(), "r");
  auto f_gpu = TFile::Open(gpu_file.data(), "r");

  auto tree_cpu_tracklets = (TTree*)f_cpu->Get("tracklets");
  auto tree_gpu_tracklets = (TTree*)f_gpu->Get("tracklets");

  auto hist_cpu_tracklets0_phi = new TH1F("hist_cpu_tracklets0_phi", "hist_cpu_tracklets0_phi", 100, -TMath::Pi() - 1, TMath::Pi() + 1);
  auto hist_gpu_tracklets0_phi = new TH1F("hist_gpu_tracklets0_phi", "hist_gpu_tracklets0_phi", 100, -TMath::Pi() - 1, TMath::Pi() + 1);
  auto hist_cpu_tracklets0_tanL = new TH1F("hist_cpu_tracklets0_tanL", "hist_cpu_tracklets0_tanL", 100, -80, 80);
  auto hist_gpu_tracklets0_tanL = new TH1F("hist_gpu_tracklets0_tanL", "hist_gpu_tracklets0_tanL", 100, -80, 80);
  auto hist_cpu_tracklets0_firstClusterIndex = new TH1F("hist_cpu_tracklets0_firstClusterIndex", "hist_cpu_tracklets0_firstClusterIndex", 200, 0, 600);
  auto hist_gpu_tracklets0_firstClusterIndex = new TH1F("hist_gpu_tracklets0_firstClusterIndex", "hist_gpu_tracklets0_firstClusterIndex", 200, 0, 600);

  auto hist_cpu_tracklets1_phi = new TH1F("hist_cpu_tracklets1_phi", "hist_cpu_tracklets1_phi", 100, -TMath::Pi() - 1, TMath::Pi() + 1);
  auto hist_gpu_tracklets1_phi = new TH1F("hist_gpu_tracklets1_phi", "hist_gpu_tracklets1_phi", 100, -TMath::Pi() - 1, TMath::Pi() + 1);
  auto hist_cpu_tracklets1_tanL = new TH1F("hist_cpu_tracklets1_tanL", "hist_cpu_tracklets1_tanL", 100, -80, 80);
  auto hist_gpu_tracklets1_tanL = new TH1F("hist_gpu_tracklets1_tanL", "hist_gpu_tracklets1_tanL", 100, -80, 80);
  auto hist_cpu_tracklets1_firstClusterIndex = new TH1F("hist_cpu_tracklets1_firstClusterIndex", "hist_cpu_tracklets1_firstClusterIndex", 200, 0, 600);
  auto hist_gpu_tracklets1_firstClusterIndex = new TH1F("hist_gpu_tracklets1_firstClusterIndex", "hist_gpu_tracklets1_firstClusterIndex", 200, 0, 600);

  hist_cpu_tracklets0_phi->SetLineColor(kRed);
  hist_gpu_tracklets0_phi->SetLineColor(kBlue);
  hist_cpu_tracklets0_tanL->SetLineColor(kRed);
  hist_gpu_tracklets0_tanL->SetLineColor(kBlue);
  hist_cpu_tracklets0_firstClusterIndex->SetLineColor(kRed);
  hist_gpu_tracklets0_firstClusterIndex->SetLineColor(kBlue);

  hist_cpu_tracklets1_phi->SetLineColor(kRed);
  hist_gpu_tracklets1_phi->SetLineColor(kBlue);
  hist_cpu_tracklets1_tanL->SetLineColor(kRed);
  hist_gpu_tracklets1_tanL->SetLineColor(kBlue);
  hist_cpu_tracklets1_firstClusterIndex->SetLineColor(kRed);
  hist_gpu_tracklets1_firstClusterIndex->SetLineColor(kBlue);

  auto c1 = new TCanvas("c1", "c1", 800, 800);
  c1->cd();

  tree_cpu_tracklets->Draw("Tracklets0.phi >> hist_cpu_tracklets0_phi");
  tree_gpu_tracklets->Draw("Tracklets0.phi >> hist_gpu_tracklets0_phi");

  hist_gpu_tracklets0_phi->Draw();
  hist_cpu_tracklets0_phi->Draw("same");

  auto legend = new TLegend(0.4, 0.4, 0.2, 0.2);
  legend->SetHeader("Tracklets0 #varphi", "C");
  legend->AddEntry(hist_cpu_tracklets0_phi, Form("CPU: %1.f", hist_cpu_tracklets0_phi->GetEntries()), "l");
  legend->AddEntry(hist_gpu_tracklets0_phi, Form("GPU: %1.f", hist_gpu_tracklets0_phi->GetEntries()), "l");

  legend->Draw();

  // // Tan(L)
  auto c2 = new TCanvas("c2", "c2", 800, 800);
  c2->cd();

  tree_cpu_tracklets->Draw("Tracklets0.tanLambda >> hist_cpu_tracklets0_tanL");
  tree_gpu_tracklets->Draw("Tracklets0.tanLambda >> hist_gpu_tracklets0_tanL");

  hist_gpu_tracklets0_tanL->Draw();
  hist_cpu_tracklets0_tanL->Draw("same");

  auto legend2 = new TLegend(0.4, 0.4, 0.2, 0.2);
  legend2->SetHeader("Tracklets0 tan(#lambda)", "C");
  legend2->AddEntry(hist_cpu_tracklets0_tanL, Form("CPU: %1.f", hist_cpu_tracklets0_tanL->GetEntries()), "l");
  legend2->AddEntry(hist_gpu_tracklets0_tanL, Form("GPU: %1.f", hist_gpu_tracklets0_tanL->GetEntries()), "l");

  legend2->Draw();

  // first cluster index
  auto c3 = new TCanvas("c3", "c3", 800, 800);
  c3->cd();

  tree_cpu_tracklets->Draw("Tracklets0.firstClusterIndex >> hist_cpu_tracklets0_firstClusterIndex");
  tree_gpu_tracklets->Draw("Tracklets0.firstClusterIndex >> hist_gpu_tracklets0_firstClusterIndex");
  auto hist_cpu_tracklets0_firstClusterIndex_clone = (TH1F*)hist_cpu_tracklets0_firstClusterIndex->Clone("hist_cpu_tracklets0_firstClusterIndex_clone");
  hist_cpu_tracklets0_firstClusterIndex_clone->Add(hist_gpu_tracklets0_firstClusterIndex, -1);
  hist_cpu_tracklets0_firstClusterIndex_clone->SetLineColor(kBlack);

  hist_gpu_tracklets0_firstClusterIndex->Draw();
  hist_cpu_tracklets0_firstClusterIndex->Draw("same");
  hist_cpu_tracklets0_firstClusterIndex_clone->Draw("same");

  auto legend3 = new TLegend(0.4, 0.4, 0.2, 0.2);
  legend3->SetHeader("Tracklets0 first cluster index", "C");
  legend3->AddEntry(hist_cpu_tracklets0_firstClusterIndex, Form("CPU: %1.f", hist_cpu_tracklets0_firstClusterIndex->GetEntries()), "l");
  legend3->AddEntry(hist_gpu_tracklets0_firstClusterIndex, Form("GPU: %1.f", hist_gpu_tracklets0_firstClusterIndex->GetEntries()), "l");

  legend3->Draw();

  // // Tracklets1
  auto c4 = new TCanvas("c4", "c4", 800, 800);
  c4->cd();

  tree_cpu_tracklets->Draw("Tracklets1.phi >> hist_cpu_tracklets1_phi");
  tree_gpu_tracklets->Draw("Tracklets1.phi >> hist_gpu_tracklets1_phi");

  hist_cpu_tracklets1_phi->Draw();
  hist_gpu_tracklets1_phi->Draw("same");

  auto legend4 = new TLegend(0.4, 0.4, 0.2, 0.2);
  legend4->SetHeader("Tracklets1 #varphi", "C");
  legend4->AddEntry(hist_cpu_tracklets1_phi, Form("CPU: %1.f", hist_cpu_tracklets1_phi->GetEntries()), "l");
  legend4->AddEntry(hist_gpu_tracklets1_phi, Form("GPU: %1.f", hist_gpu_tracklets1_phi->GetEntries()), "l");

  legend4->Draw();

  // // Tan(L)
  auto c5 = new TCanvas("c5", "c5", 800, 800);
  c5->cd();

  tree_cpu_tracklets->Draw("Tracklets1.tanLambda >> hist_cpu_tracklets1_tanL");
  tree_gpu_tracklets->Draw("Tracklets1.tanLambda >> hist_gpu_tracklets1_tanL");

  hist_cpu_tracklets1_tanL->Draw();
  hist_gpu_tracklets1_tanL->Draw("same");

  auto legend5 = new TLegend(0.4, 0.4, 0.2, 0.2);
  legend5->SetHeader("Tracklets1 tan(#lambda)", "C");
  legend5->AddEntry(hist_cpu_tracklets1_tanL, Form("CPU: %1.f", hist_cpu_tracklets1_tanL->GetEntries()), "l");
  legend5->AddEntry(hist_gpu_tracklets1_tanL, Form("GPU: %1.f", hist_gpu_tracklets1_tanL->GetEntries()), "l");

  legend5->Draw();

  // first cluster index
  auto c6 = new TCanvas("c6", "c6", 800, 800);
  c6->cd();

  tree_cpu_tracklets->Draw("Tracklets1.firstClusterIndex >> hist_cpu_tracklets1_firstClusterIndex");
  tree_gpu_tracklets->Draw("Tracklets1.firstClusterIndex >> hist_gpu_tracklets1_firstClusterIndex");
  auto hist_cpu_tracklets1_firstClusterIndex_clone = (TH1F*)hist_cpu_tracklets1_firstClusterIndex->Clone("hist_cpu_tracklets1_firstClusterIndex_clone");
  hist_cpu_tracklets1_firstClusterIndex_clone->Add(hist_gpu_tracklets1_firstClusterIndex, -1);
  hist_cpu_tracklets1_firstClusterIndex_clone->SetLineColor(kBlack);

  hist_cpu_tracklets1_firstClusterIndex->Draw();
  hist_gpu_tracklets1_firstClusterIndex->Draw("same");
  hist_cpu_tracklets1_firstClusterIndex_clone->Draw("same");
  auto legend6 = new TLegend(0.4, 0.4, 0.2, 0.2);

  legend6->SetHeader("Tracklets1 first cluster index", "C");
  legend6->AddEntry(hist_cpu_tracklets1_firstClusterIndex, Form("CPU: %1.f", hist_cpu_tracklets1_firstClusterIndex->GetEntries()), "l");
  legend6->AddEntry(hist_gpu_tracklets1_firstClusterIndex, Form("GPU: %1.f", hist_gpu_tracklets1_firstClusterIndex->GetEntries()), "l");

  legend6->Draw();
}