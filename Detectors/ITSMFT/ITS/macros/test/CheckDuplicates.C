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

/// \file CheckDuplicates.C
/// \brief Macro to look for duplicate tracks across a ROF windows

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TStyle.h>
#include <TGaxis.h>

#include "DataFormatsITS/TrackITS.h"
#endif
#include "DataFormatsITSMFT/ROFRecord.h"

void CheckDuplicates(TString tracfile = "./o2trac_its.root", TString output = ".", TString rootSave = "", int rofStart = 0, int rofEnd = -1, int windowSize = 2, bool includeSame = 1)
{
  gErrorIgnoreLevel = kWarning;

  // Obtain data from files:
  TFile* file1 = TFile::Open(tracfile);
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  // Extract track information
  std::vector<o2::its::TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);
  // Extract information on which ROFs contain which tracks
  std::vector<o2::itsmft::ROFRecord>* rofArr = nullptr;
  recTree->SetBranchAddress("ITSTracksROF", &rofArr);

  // Specify plot parameters:
  int phiBins{101}, etaBins{101};                                                                                        // number of bins for dPhi and dEta axes
  double_t phiMin{-2 * TMath::Pi()}, phiMax{2 * TMath::Pi()}, etaMin{-4}, etaMax{4};                                     // dPhi and dEta axes limits
  double_t phiLim{0.01}, etaLim{0.01};                                                                                   // limits for determining duplicate status
  auto h1 = new TH2D("h1", "", phiBins, phiMin, phiMax, etaBins, etaMin, etaMax);                                        // Distribution of all comparisons over dPhi and dEta
  auto h2 = new TH2D("h2", "All pairs; #||{#Delta#phi} [rad]; #||{#Delta#eta}", phiBins, 0, phiMax, etaBins, 0, etaMax); // Distribution of all comparisons over |dPhi| and |dEta|
  auto h3 = new TH2D("dupl", "Duplicates; #Delta#phi; #Delta#eta", 100, -1 * phiLim, phiLim, 100, -1 * etaLim, etaLim);  // Distribution of duplicates pairs over dPhi and dEta
  auto h4 = new TH2D("h4", "Duplicates; avg. #phi; avg. #eta", 50, 0, phiMax, 50, etaMin, etaMax);                       // Distribution of duplicate pairs over avg. phi and avg. eta

  int comps{0}, dupl{0}, zeroes{0}, totalROFs{0}; // Collect summary information
  double_t eta1, eta2, phi1, phi2, dPhi, dEta, avgPhi, avgEta;
  int rofLow, rofHigh, rof1Low, rof1High, rof2Low, rof2High;
  int rofFrameEnd = rofEnd; // In case frames have different numbers of ROFs, store the final ROF index separately for each frame inside rofFrameEnd
  int totalFrames = recTree->GetEntriesFast();
  for (int frame = 0; frame < totalFrames; frame++) { // Loop over timeframes
    LOGP(info, "Proceeding to frame {} of {}", frame + 1, totalFrames);
    recTree->GetEvent(frame);
    LOGP(info, "{} ROFs in frame {}/{}", rofArr->size(), frame + 1, totalFrames);
    totalROFs += rofArr->size();
    if (rofEnd >= (int long)rofArr->size() || rofEnd < 0) { // Check whether rofEnd is greater than the total number of ROFs in this dataset, or negative; if so, set to maximum
      LOGP(warn, "Frame {}/{}: currently set upper ROF limit {} out of bounds; set to maximum {} instead.", frame + 1, totalFrames, rofEnd, rofArr->size() - 1);
      rofFrameEnd = rofArr->size() - 1;
    }

    // First loop: for each pair of ROFs in the first window, compare the tracks within them
    for (int rof1 = rofStart; rof1 < rofStart + windowSize - 1; rof1++) {
      for (int rof2 = rof1 + 1; rof2 <= rofStart + windowSize - 1; rof2++) {
        rof1Low = rofArr->at(rof1).getEntry().getFirstEntry();
        rof1High = rofArr->at(rof1).getEntry().getEntriesBound();
        rof2Low = rofArr->at(rof2).getEntry().getFirstEntry();
        rof2High = rofArr->at(rof2).getEntry().getEntriesBound();
        for (int iTrack1 = rof1Low; iTrack1 < rof1High; ++iTrack1) {
          for (int iTrack2 = rof2Low; iTrack2 < rof2High; ++iTrack2) {
            phi1 = recArr->at(iTrack1).getPhi();
            phi2 = recArr->at(iTrack2).getPhi();
            eta1 = recArr->at(iTrack1).getEta();
            eta2 = recArr->at(iTrack2).getEta();

            dPhi = phi1 - phi2;
            dEta = eta1 - eta2;
            avgPhi = (phi1 + phi2) / 2;
            avgEta = (eta1 + eta2) / 2;
            h1->Fill(dPhi, dEta);
            h2->Fill(abs(dPhi), abs(dEta));

            if (abs(dPhi) < phiLim && abs(dEta) < etaLim) {
              if (dPhi == 0 && dEta == 0) {
                LOGP(warning, "Zero found; ROFs: {}, {}. Tracks: {}, {}. dPhi=dEta=0. Not included in plots.", rof1, rof2, iTrack1, iTrack2);
                zeroes++;
              } else {
                LOGP(info, "Possible duplicate found. ROFs: {}, {}. Tracks: {}, {}. dPhi={}, dEta={}.", rof1, rof2, iTrack1, iTrack2, dPhi, dEta);
                dupl++;
                h3->Fill(dPhi, dEta);
                h4->Fill(avgPhi, avgEta);
              }
            }
            ++comps;
          }
        }
      }
    }
    LOGP(info, "Frame {}/{}: initial ROF window {}-{} complete. {} possible duplicates out of {} total comparisons.", frame + 1, totalFrames, rofStart, rofStart + windowSize - 1, dupl, comps);

    // Next loops: for each new window, pair the last ROF with the other ROFs and run comparisons.  We avoid other ROF pairs since they have already been done
    for (int rofWLow = rofStart + 1; rofWLow <= rofFrameEnd - windowSize + 1; rofWLow++) {
      int rof2 = rofWLow + windowSize - 1;
      for (int rof1 = rofWLow; rof1 < rof2; rof1++) {
        rof1Low = rofArr->at(rof1).getEntry().getFirstEntry();
        rof1High = rofArr->at(rof1).getEntry().getEntriesBound();
        rof2Low = rofArr->at(rof2).getEntry().getFirstEntry();
        rof2High = rofArr->at(rof2).getEntry().getEntriesBound();
        for (int iTrack1 = rof1Low; iTrack1 < rof1High; ++iTrack1) {
          for (int iTrack2 = rof2Low; iTrack2 < rof2High; ++iTrack2) {
            phi1 = recArr->at(iTrack1).getPhi();
            phi2 = recArr->at(iTrack2).getPhi();
            eta1 = recArr->at(iTrack1).getEta();
            eta2 = recArr->at(iTrack2).getEta();

            dPhi = phi1 - phi2;
            dEta = eta1 - eta2;
            avgPhi = (phi1 + phi2) / 2;
            avgEta = (eta1 + eta2) / 2;
            h1->Fill(dPhi, dEta);
            h2->Fill(abs(dPhi), abs(dEta));

            if (abs(dPhi) < phiLim && abs(dEta) < etaLim) {
              if (dPhi == 0 && dEta == 0) {
                LOGP(warning, "Zero found; ROFs: {}, {}. Tracks: {}, {}. dPhi=dEta=0. Not included in plots.", rof1, rof2, iTrack1, iTrack2);
                zeroes++;
              } else {
                LOGP(info, "Possible duplicate found. ROFs: {}, {}. Tracks: {}, {}. dPhi={}, dEta={}.", rof1, rof2, iTrack1, iTrack2, dPhi, dEta);
                dupl++;
                h3->Fill(dPhi, dEta);
                h4->Fill(avgPhi, avgEta);
              }
            }
            ++comps;
          }
        }
      }
    }
    LOGP(info, "Frame {}/{}: inter-ROF analysis complete. {} possible duplicates out of {} total comparisons.", frame + 1, totalFrames, dupl, comps);

    if (includeSame) { // If includeSame == 1, then we also run comparisons between tracks in the same ROF.  Otherwise, skip
      for (int rof = rofStart; rof <= rofFrameEnd; rof++) {
        rofLow = rofArr->at(rof).getEntry().getFirstEntry();
        rofHigh = rofArr->at(rof).getEntry().getEntriesBound();
        for (int iTrack1 = rofLow; iTrack1 < rofHigh; ++iTrack1) {
          for (int iTrack2 = iTrack1 + 1; iTrack2 < rofHigh; ++iTrack2) {
            phi1 = recArr->at(iTrack1).getPhi();
            phi2 = recArr->at(iTrack2).getPhi();
            eta1 = recArr->at(iTrack1).getEta();
            eta2 = recArr->at(iTrack2).getEta();

            dPhi = phi1 - phi2;
            dEta = eta1 - eta2;
            avgPhi = (phi1 + phi2) / 2;
            avgEta = (eta1 + eta2) / 2;
            h1->Fill(dPhi, dEta);
            h2->Fill(abs(dPhi), abs(dEta));

            if (abs(dPhi) < phiLim && abs(dEta) < etaLim) {
              if (dPhi == 0 && dEta == 0) {
                LOGP(warning, "Zero found; ROFs: {}, {}. Tracks: {}, {}. dPhi=dEta=0. Not included in plots.", rof, rof, iTrack1, iTrack2);
                zeroes++;
              } else {
                LOGP(info, "Possible duplicate found. ROFs: {}, {}. Tracks: {}, {}. dPhi={}, dEta={}.", rof, rof, iTrack1, iTrack2, dPhi, dEta);
                dupl++;
                h3->Fill(dPhi, dEta);
                h4->Fill(avgPhi, avgEta);
              }
            }
            ++comps;
          }
        }
      }
      LOGP(info, "Frame {}/{}: intra-ROF analysis complete. {} possible duplicates out of {} total comparisons.", frame + 1, totalFrames, dupl, comps);
    } else
      LOGP(info, "Frame {}/{}: intra-ROF analysis not included as per parameters (includeSame={})", frame + 1, totalFrames, includeSame);
  }
  // NOTE: comparisons between ROFs in different frames are intentionally left out.  Only if reconstruction was performed on contiguous CTFs do comparisons between
  // ROFs in different frames make sense.

  TString prefix = ""; // insert any more information you'd like to see on the plot title here, like run numbers
  h1->SetTitle(Form("%s/%i ROFs/window size %i;#Delta#phi [rad];#Delta#eta", prefix.Data(), totalROFs, windowSize));

  gStyle->SetOptStat("ne");
  auto c1 = new TCanvas("c1", "", 0, 0, 1200, 800);
  h1->SetStats(1);
  h1->GetYaxis()->SetTitleOffset(1.4);
  h1->GetXaxis()->SetTitleOffset(1.4);
  h1->GetXaxis()->CenterTitle(true);
  h1->GetYaxis()->CenterTitle(true);
  h1->Draw("lego");

  auto c2 = new TCanvas("c2", "", 0, 0, 1200, 800);
  h2->SetStats(0);
  h2->Draw("COLZ");

  auto c3 = new TCanvas("c3", "", 0, 0, 1200, 800);
  h3->SetStats(1);
  h3->GetXaxis()->SetMaxDigits(1);
  h3->GetXaxis()->CenterTitle(true);
  h3->GetYaxis()->CenterTitle(true);
  TGaxis::SetExponentOffset(0, -0.05, "x");
  h3->GetYaxis()->SetMaxDigits(1);
  h3->GetYaxis()->SetTitleOffset(0.8);
  h3->Draw("COLZ");

  auto c4 = new TCanvas("c4", "", 0, 0, 1200, 800);
  h4->SetStats(0);
  h4->Draw("COLZ");

  auto c5 = new TCanvas("c5", "", 0, 0, 1200, 800);
  TH1D* h5 = h4->ProjectionX("pPhi");
  h5->SetTitle("Duplicate distribution over #phi");
  c5->SetLogy();
  h5->SetStats(0);
  h5->Draw();

  auto c6 = new TCanvas("c6", "", 0, 0, 1200, 800);
  TH1D* h6 = h4->ProjectionY("pEta");
  h6->SetTitle("Duplicate distribution over #eta");
  c6->SetLogy();
  h6->SetStats(0);
  h6->Draw();

  if (rootSave.Length() > 0) {
    TFile output(Form("%s/projections.root", rootSave.Data()), "RECREATE");
    h5->Write();
    h6->Write();
    output.Close();
  }

  TString filename = Form("%s/plots.pdf", output.Data());
  c1->Print(filename + "[");
  c1->Print(filename);
  c2->Print(filename);
  c3->Print(filename);
  c4->Print(filename);
  c5->Print(filename);
  c6->Print(filename);
  c6->Print(filename + "]");
  LOGP(info, "Summary:");
  LOGP(info, "Zeroes: {}; duplicates: {}; total comparisons: {}.", zeroes, dupl, comps);
}
