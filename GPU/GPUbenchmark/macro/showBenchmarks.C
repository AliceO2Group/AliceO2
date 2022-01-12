#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TString.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>
#include <TKey.h>
#include <TBranch.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#endif

int nBins{500};
float minHist{0.f}, maxHist{1e4};
void showBenchmarks(const bool times = false, const TString fileName = "0_benchmark_result.root")
{
  auto f = TFile::Open(fileName.Data(), "UPDATE");
  std::unordered_map<std::string, TTree*> um_trees;
  std::vector<std::vector<TH1F*>> histograms;
  std::vector<TGraphErrors*> results;
  std::vector<std::string> tests = {"read", "write", "copy"};
  std::vector<std::string> types = {"char", "int", "unsigned_long"};
  std::vector<std::string> modes = {"seq", "conc"};
  std::vector<std::string> patterns = {"SB", "MB"};

  for (auto&& keyAsObj : *f->GetListOfKeys()) {
    auto tName = ((TKey*)keyAsObj)->GetName();
    um_trees[tName] = (TTree*)f->Get(tName);
  }
  std::cout << "Found " << um_trees.size() << " trees.\n";

  // Main loop
  for (auto& keyPair : um_trees) {
    for (auto& test : tests) {
      if (keyPair.first.find(test) != std::string::npos) {
        for (auto& mode : modes) {
          if (keyPair.first.find(mode) != std::string::npos) {
            for (auto& type : types) {
              if (keyPair.first.find(type) != std::string::npos) {
                for (auto& pattern : patterns) {
                  if (keyPair.first.find(pattern) != std::string::npos) {
                    if ((keyPair.first.find("TP") == std::string::npos)) {
                      if (times) {
                        // Single Tree entry, we know test, type, mode, pattern
                        std::vector<float>* measures = 0;
                        TBranch* elapsed;
                        keyPair.second->SetBranchAddress("elapsed", &measures, &elapsed);
                        elapsed->GetEntry(keyPair.second->LoadTree(0));
                        auto nChunk = measures->size();
                        histograms.emplace_back(nChunk);
                        for (int iHist{0}; iHist < (int)nChunk; ++iHist) {
                          histograms.back()[iHist] = new TH1F(Form("Chunk_%d_%s", iHist, keyPair.first.c_str()), Form("Chunk_%d_%s;ms", iHist, keyPair.first.c_str()), 1000, 0, 1e4);
                        }
                        for (size_t iEntry(0); iEntry < (size_t)keyPair.second->GetEntriesFast(); ++iEntry) {
                          auto tentry = keyPair.second->LoadTree(iEntry);
                          elapsed->GetEntry(tentry);
                          for (int iHist{0}; iHist < (int)nChunk; ++iHist) {
                            histograms.back()[iHist]->Fill((*measures)[iHist]);
                          }
                        }

                        std::vector<float> xCoord(nChunk), exCoord(nChunk), yCoord(nChunk), eyCoord(nChunk);

                        for (size_t i{0}; i < nChunk; ++i) {
                          xCoord[i] = (float)i;
                          yCoord[i] = histograms.back()[i]->GetMean();
                          eyCoord[i] = histograms.back()[i]->GetRMS();
                          exCoord[i] = 0.f;
                        }
                        TCanvas* c = new TCanvas(Form("c%s", keyPair.first.c_str()), Form("%s", keyPair.first.c_str()));
                        c->cd();
                        TGraphErrors* g = new TGraphErrors(nChunk, xCoord.data(), yCoord.data(), exCoord.data(), eyCoord.data());
                        g->GetYaxis()->SetRangeUser(0, 5000);
                        g->GetXaxis()->SetRangeUser(-2.f, nChunk);
                        g->SetTitle(Form("%s, N_{test}=%d;chunk_id;elapsed (s)", keyPair.first.c_str(), (int)keyPair.second->GetEntriesFast()));
                        g->SetFillColor(kBlue);
                        g->SetFillStyle(3001);
                        g->Draw("AB");
                      }
                    } else { // TP plots //
                      // Single Tree entry, we know test, type, mode, pattern
                      std::vector<float>* measures = 0;
                      TBranch* throughput;
                      keyPair.second->SetBranchAddress("throughput", &measures, &throughput);
                      throughput->GetEntry(keyPair.second->LoadTree(0));
                      auto nChunk = measures->size();
                      histograms.emplace_back(nChunk);
                      for (int iHist{0}; iHist < (int)nChunk; ++iHist) {
                        histograms.back()[iHist] = new TH1F(Form("Chunk_%d_%s", iHist, keyPair.first.c_str()), Form("Chunk_%d_%s;GB/s", iHist, keyPair.first.c_str()), 1000, 0, 1e3);
                      }
                      for (size_t iEntry(0); iEntry < (size_t)keyPair.second->GetEntriesFast(); ++iEntry) {
                        auto tentry = keyPair.second->LoadTree(iEntry);
                        throughput->GetEntry(tentry);
                        for (int iHist{0}; iHist < (int)nChunk; ++iHist) {
                          histograms.back()[iHist]->Fill((*measures)[iHist]);
                        }
                      }

                      std::vector<float> xCoord(nChunk), exCoord(nChunk), yCoord(nChunk), eyCoord(nChunk);

                      for (size_t i{0}; i < nChunk; ++i) {
                        xCoord[i] = (float)i;
                        yCoord[i] = histograms.back()[i]->GetMean();
                        eyCoord[i] = histograms.back()[i]->GetRMS();
                        exCoord[i] = 0.f;
                      }
                      TCanvas* c = new TCanvas(Form("c%s", keyPair.first.c_str()), Form("%s", keyPair.first.c_str()));
                      c->cd();
                      TGraphErrors* g = new TGraphErrors(nChunk, xCoord.data(), yCoord.data(), exCoord.data(), eyCoord.data());
                      g->GetYaxis()->SetRangeUser(0, 150);
                      g->GetXaxis()->SetRangeUser(-2.f, nChunk);
                      g->SetTitle(Form("%s, N_{test}=%d;chunk_id;throughput (GB/s)", keyPair.first.c_str(), (int)keyPair.second->GetEntriesFast()));
                      g->SetFillColor(kBlue);
                      g->SetFillStyle(3001);
                      g->Draw("AB");
                      g->Write();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  f->Close();
}