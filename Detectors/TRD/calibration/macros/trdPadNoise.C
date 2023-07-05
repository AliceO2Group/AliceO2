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

// macro with examples for analysing noise data

#if !defined(__CLING__) || defined(__ROOTCLING__)
// ROOT header
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TH1.h>
#include <TH2.h>
#include <TLine.h>
#include <TText.h>

// O2 header
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "DataFormatsTRD/NoiseCalibration.h"
#include "Framework/Logger.h"
#include "TRDCalibration/CalibratorNoise.h"

#include <vector>
#include <string>
#endif

using namespace o2::trd;
using namespace o2::trd::constants;

const float almostZero{0.001f};

void sortDigits()
{
  // reorder the digits for quicker visualization of signal from specific channels
  // does not work with very large digit files and requires quite some RAM...
  TChain chain("o2sim");
  chain.AddFile("trddigits.root");
  std::vector<Digit> digits, *digitInPtr{&digits};
  chain.SetBranchAddress("TRDDigit", &digitInPtr);
  LOGP(info, "Total number of entries in the tree: {}", chain.GetEntries());
  auto fOut = new TFile("digitsSorted.root", "recreate");
  auto tree = new TTree("digits", "TRD digits sorted by pad");
  std::vector<Digit> digitsOut, *digitsOutPtr{&digitsOut};
  tree->Branch("digi", &digitsOutPtr);
  std::vector<std::vector<Digit>> digitsPerChannel;
  digitsPerChannel.resize(NCHANNELSTOTAL);
  for (int iEntry = 0; iEntry < chain.GetEntries(); ++iEntry) {
    LOGP(info, "Checking entry {} of {}", iEntry, chain.GetEntries());
    chain.GetEntry(iEntry);
    for (const auto& digit : digits) {
      int indexGlobal = HelperMethods::getGlobalChannelIndex(digit.getDetector(), digit.getROB(), digit.getMCM(), digit.getChannel());
      digitsPerChannel[indexGlobal].push_back(digit);
    }
  }
  for (int i = 0; i < NCHANNELSTOTAL; ++i) {
    digitsOut = digitsPerChannel[i];
    tree->Fill();
  }
  tree->Write();
  delete tree;
  fOut->Close();
  delete fOut;
}

void drawGrid(bool stack2 = false)
{
  auto line = new TLine();
  line->SetLineWidth(2);
  int maxRow = stack2 ? 12 : 16;
  for (int i = 0; i < maxRow; ++i) {
    float xStart = -0.5, xEnd = 167.5;
    float y = 0.5 + i;
    line->DrawLine(xStart, y, xEnd, y);
  }
  for (int i = 0; i < 8; ++i) {
    float yStart = -0.5, yEnd = maxRow - 0.5;
    float x = 20.5 + 21 * i;
    line->DrawLine(x, yStart, x, yEnd);
  }
}

void plotChamberwise()
{
  auto fPads = TFile::Open("padInfo.root");
  auto tree = (TTree*)fPads->Get("padValues");
  ChannelInfoDetailed channelInfo, *channelInfoPtr{&channelInfo};
  tree->SetBranchAddress("padInfo", &channelInfoPtr);
  auto ccdbObject = (o2::trd::ChannelInfoContainer*)fPads->Get("ccdbObject");
  auto fOut = TFile::Open("hists.root", "recreate");
  std::vector<TH2F*> histsMean;
  std::vector<TH2F*> histsRMS;
  std::vector<TH2F*> histsFlags;
  for (int iDet = 0; iDet < 540; ++iDet) {
    int nRows = (HelperMethods::getStack(iDet) == 2) ? 12 : 16;
    histsMean.push_back(new TH2F(Form("mean_det_%i", iDet), Form("%i_%i_%i;channel;row", HelperMethods::getSector(iDet), HelperMethods::getStack(iDet), HelperMethods::getLayer(iDet)), 168, -0.5, 167.5, nRows, -0.5, nRows - 0.5));
    histsMean.back()->SetStats(0);
    histsRMS.push_back(new TH2F(Form("rms_det_%i", iDet), Form("%i_%i_%i;channel;row", HelperMethods::getSector(iDet), HelperMethods::getStack(iDet), HelperMethods::getLayer(iDet)), 168, -0.5, 167.5, nRows, -0.5, nRows - 0.5));
    histsRMS.back()->SetStats(0);
    histsFlags.push_back(new TH2F(Form("flags_det_%i", iDet), Form("%i_%i_%i;channel;row", HelperMethods::getSector(iDet), HelperMethods::getStack(iDet), HelperMethods::getLayer(iDet)), 168, -0.5, 167.5, nRows, -0.5, nRows - 0.5));
    histsFlags.back()->SetStats(0);
  }
  for (int idx = 0; idx < ccdbObject->getData().size(); ++idx) {
    const auto& channelInfo = ccdbObject->getData()[idx];
    if (channelInfo.isDummy()) {
      continue;
    }
    int det, rob, mcm, channel;
    HelperMethods::getPositionFromGlobalChannelIndex(idx, det, rob, mcm, channel);
    int channelGlb = HelperMethods::getChannelIndexInColumn(rob, mcm, channel);
    int bin = histsMean[0]->FindBin(channelGlb, HelperMethods::getPadRowFromMCM(rob, mcm));
    histsMean[det]->SetBinContent(bin, channelInfo.getMean());
    histsRMS[det]->SetBinContent(bin, channelInfo.getRMS());
    uint8_t flags = 1;
    // TODO change flags to see chamber-wise which channels fullfil given criteria
    histsFlags[det]->SetBinContent(bin, flags);
  }
  for (int iDet = 0; iDet < 540; ++iDet) {
    histsMean[iDet]->Write();
    delete histsMean[iDet];
    histsRMS[iDet]->Write();
    delete histsRMS[iDet];
    histsFlags[iDet]->Write();
    delete histsFlags[iDet];
  }
  fOut->Close();
  delete fOut;
}

void plotNoisyChannels()
{
  auto fDigits = TFile::Open("digitsSorted.root");
  auto treeDigits = (TTree*)fDigits->Get("digits");
  std::vector<Digit> digits, *digitsPtr{&digits};
  treeDigits->SetBranchAddress("digi", &digitsPtr);
  auto fPads = TFile::Open("padInfo.root");
  auto ccdbObject = (o2::trd::ChannelInfoContainer*)fPads->Get("ccdbObject");
  auto tree = (TTree*)fPads->Get("padValues");
  ChannelInfoDetailed channelInfo, *channelInfoPtr{&channelInfo};
  tree->SetBranchAddress("padInfo", &channelInfoPtr);

  auto text = new TText();
  auto c = new TCanvas("c1", "c1", 1400, 700);
  c->Divide(2, 1);
  auto hPH = new TH2F("ph", ";time bin;ADC", 30, -0.5, 29.5, 1024, -0.5, 1023.5);
  auto hADC = new TH1F("adc", ";ADC;counts", 1024, -0.5, 1023.5);
  for (int idx = 0; idx < ccdbObject->getData().size(); ++idx) {
    const auto& channel = ccdbObject->getData()[idx];
    if (channel.isDummy()) {
      continue;
    }
    if (channel.getMean() > 10.4f ||
        channel.getMean() < 8.8f ||
        channel.getRMS() > 5.f ||
        (channel.getRMS() < 0.2f && channel.getRMS() > almostZero)) {
      hPH->Reset();
      hADC->Reset();
      tree->GetEntry(idx);
      treeDigits->GetEntry(idx);
      for (const auto& d : digits) {
        for (int iTb = 0; iTb < 30; ++iTb) {
          hPH->Fill(iTb, d.getADC()[iTb]);
          hADC->Fill(d.getADC()[iTb]);
        }
      }
      c->cd(1);
      hPH->Draw("colz");
      c->cd(2);
      hADC->Draw();
      text->DrawTextNDC(0., 0.92, Form("%2i_%i_%i, row %i, col %i, isShared(%i)", channelInfo.sec, channelInfo.stack, channelInfo.layer, channelInfo.row, channelInfo.col, channelInfo.isShared));
      c->SaveAs(Form("plots/%i/channel_%i.png", channelInfo.det, channelInfo.indexGlb));
    }
  }
}

void trdPadNoise(int maxEntry = 100)
{
  // prepare to read the TRD digits
  TChain chain("o2sim");
  chain.AddFile("trddigits.root");
  std::vector<Digit> digits, *digitInPtr{&digits};
  chain.SetBranchAddress("TRDDigit", &digitInPtr);
  LOGP(info, "Total number of entries in the tree: {}", chain.GetEntries());

  // calibrator
  CalibratorNoise calib;

  // here we will store the output
  auto fOut = new TFile("padInfo.root", "recreate");
  auto tree = new TTree("padValues", "Mean and RMS ADC information per pad");
  ChannelInfoDetailed channelInfo;
  tree->Branch("padInfo", &channelInfo);

  LOG(info) << "Looping over all provided digits and filling channel information";
  for (int iEntry = 0; iEntry < chain.GetEntries(); ++iEntry) {
    if (iEntry >= maxEntry) {
      LOGP(info, "Reached maximum number of entries ({}). Skipping the rest of the input digits", maxEntry);
      break;
    }
    chain.GetEntry(iEntry); // for each TimeFrame there is one tree entry
    calib.process(digits);
  }
  LOG(info) << "Done reading the input from the digits";
  calib.collectChannelInfo();

  // now we just need to store the output in the TTree, so we loop over each channel,
  // copy the object from the vector into channelInfo and call TTree::Fill()
  int countFilledChannels = 0;
  for (const auto& info : calib.getInternalChannelInfos()) {
    channelInfo = info;
    if (info.nEntries > 0) {
      ++countFilledChannels;
    }
    tree->Fill();
  }
  ChannelInfoContainer ccdbObject = calib.getCcdbObject();
  fOut->WriteObjectAny(&ccdbObject, "o2::trd::ChannelInfoContainer", "ccdbObject");

  LOGP(info, "Got information for {} out of {} channels", countFilledChannels, NCHANNELSTOTAL);
  tree->Write();
  delete tree;
  fOut->Close();
}
