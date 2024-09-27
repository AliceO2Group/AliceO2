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

#include <fmt/format.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <string_view>
#include <filesystem>

#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGButtonGroup.h"
#include "TQObject.h"
#include "TH2Poly.h"
#include "TPolyMarker.h"
#include "TLine.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TObjArray.h"
#include "TROOT.h"
#include "TMath.h"
#include "TApplication.h"

#include <fairlogger/Logger.h>

#include "GPUTPCGeometry.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"
#include "DataFormatsTPC/Constants.h"

#include "TPCMonitor/SimpleEventDisplayGUI.h"

using namespace o2::tpc;
namespace fs = std::filesystem;

//__________________________________________________________________________
void SimpleEventDisplayGUI::monitorGui()
{
  float xsize = 160;
  float ysize = 25;
  float yoffset = 10;
  float ysize_dist = 2;
  float mainx = xsize + 2 * 10;
  float mainy = 335;
  int ycount = 0;
  float currentY = yoffset + ycount * (ysize_dist + ysize);

  auto nextY = [&ycount, &currentY, yoffset, ysize, ysize_dist]() {
    ++ycount;
    currentY = yoffset + ycount * (ysize_dist + ysize);
  };

  TGMainFrame* mFrameMain = new TGMainFrame(gClient->GetRoot(), mainx, mainy, kMainFrame | kVerticalFrame);
  mFrameMain->SetLayoutBroken(kTRUE);
  mFrameMain->SetCleanup(kDeepCleanup);

  TGCompositeFrame* mContRight = new TGCompositeFrame(mFrameMain, xsize + 5, mainy, kVerticalFrame | kFixedWidth | kFitHeight);
  mFrameMain->AddFrame(mContRight, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY | kLHintsExpandX, 3, 5, 3, 3));

  //---------------------------
  TGTextButton* mFrameNextEvent = new TGTextButton(mContRight, "&Next Event");
  mContRight->AddFrame(mFrameNextEvent, new TGLayoutHints(kLHintsExpandX));

  mFrameNextEvent->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "next(=-1)");
  mFrameNextEvent->SetTextColor(200);
  mFrameNextEvent->SetToolTipText("Go to next event");
  mFrameNextEvent->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  nextY();

  //---------------------------
  TGTextButton* mFramePreviousEvent = new TGTextButton(mContRight, "&Previous Event");
  mContRight->AddFrame(mFramePreviousEvent, new TGLayoutHints(kLHintsExpandX));
  if (mRunMode == RunMode::Online) {
    mFramePreviousEvent->SetState(kButtonDisabled);
  }

  mFramePreviousEvent->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "next(=-2)");
  mFramePreviousEvent->SetTextColor(200);
  mFramePreviousEvent->SetToolTipText("Go to previous event");
  mFramePreviousEvent->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  nextY();

  //---------------------------

  TGTextButton* mGoToEvent = new TGTextButton(mContRight, "&Go to event");
  mContRight->AddFrame(mGoToEvent, new TGLayoutHints(kLHintsNormal));

  mGoToEvent->SetTextColor(200);
  mGoToEvent->SetToolTipText("Go to event");
  mGoToEvent->MoveResize(10, currentY, 0.65 * xsize, (unsigned int)ysize);
  mGoToEvent->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "callEventNumber()");

  //
  auto* ftbuf = new TGTextBuffer(10);
  ftbuf->AddText(0, "0");
  mEventNumber = new TGTextEntry(mContRight, ftbuf);
  mContRight->AddFrame(mEventNumber, new TGLayoutHints(kFitHeight));
  mEventNumber->MoveResize(0.7 * xsize, currentY, 0.3 * xsize, (unsigned int)ysize);
  mEventNumber->SetAlignment(kTextRight);
  nextY();

  //---------------------------
  TGTextButton* mApplySignalThreshold = new TGTextButton(mContRight, "&Apply Threshold");
  mContRight->AddFrame(mApplySignalThreshold, new TGLayoutHints(kLHintsNormal));

  mApplySignalThreshold->SetTextColor(200);
  mApplySignalThreshold->SetToolTipText("Apply Threshold");
  mApplySignalThreshold->MoveResize(10, currentY, 0.65 * xsize, (unsigned int)ysize);
  mApplySignalThreshold->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "applySignalThreshold()");

  auto* signalThresholdBuf = new TGTextBuffer(10);
  signalThresholdBuf->AddText(0, "0");
  mSignalThresholdValue = new TGTextEntry(mContRight, signalThresholdBuf);
  mSignalThresholdValue->MoveResize(0.7 * xsize, currentY, 0.3 * xsize, (unsigned int)ysize);
  mSignalThresholdValue->SetAlignment(kTextRight);
  mSignalThresholdValue->Connect("ReturnPressed()", "o2::tpc::SimpleEventDisplayGUI", this, "applySignalThreshold()");
  nextY();

  //---------------------------
  mCheckSingleTB = new TGCheckButton(mContRight, "One TB");
  mContRight->AddFrame(mCheckSingleTB, new TGLayoutHints(kLHintsExpandX));

  mCheckSingleTB->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "toggleSingleTimeBin()");
  mCheckSingleTB->SetTextColor(200);
  mCheckSingleTB->SetToolTipText("Show single time bin");
  mCheckSingleTB->MoveResize(10, currentY, 0.5 * xsize, (unsigned int)ysize);
  mCheckSingleTB->SetDown(0);

  mSelTimeBin = new TGNumberEntry(mContRight, mEvDisp.getFirstTimeBin(), 6, 999, TGNumberFormat::kNESInteger,
                                  TGNumberFormat::kNEAPositive,
                                  TGNumberFormat::kNELLimitMinMax,
                                  mEvDisp.getFirstTimeBin(), mEvDisp.getLastTimeBin());

  mSelTimeBin->MoveResize(0.55 * xsize, currentY, 0.45 * xsize, (unsigned int)ysize);
  mSelTimeBin->Connect("ValueSet(Long_t)", "o2::tpc::SimpleEventDisplayGUI", this, "selectTimeBin()");
  (mSelTimeBin->GetNumberEntry())->Connect("ReturnPressed()", "o2::tpc::SimpleEventDisplayGUI", this, "selectTimeBin()");
  nextY();

  //---------------------------
  mCheckFFT = new TGCheckButton(mContRight, "Show FFT");
  mContRight->AddFrame(mCheckFFT, new TGLayoutHints(kLHintsExpandX));

  mCheckFFT->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "toggleFFT()");
  mCheckFFT->SetTextColor(200);
  mCheckFFT->SetToolTipText("Switch on FFT calculation");
  mCheckFFT->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  mCheckFFT->SetDown(0);
  toggleFFT();
  nextY();

  //---------------------------
  mCheckOccupancy = new TGCheckButton(mContRight, "Show Occupancy");
  mContRight->AddFrame(mCheckOccupancy, new TGLayoutHints(kLHintsExpandX));

  mCheckOccupancy->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "toggleOccupancy()");
  mCheckOccupancy->SetTextColor(200);
  mCheckOccupancy->SetToolTipText("Switch on Occupancy calculation");
  mCheckOccupancy->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  mCheckOccupancy->SetDown(0);
  toggleOccupancy();
  nextY();

  //---------------------------
  mCheckPadTime = new TGCheckButton(mContRight, "Show PadTime");
  mContRight->AddFrame(mCheckPadTime, new TGLayoutHints(kLHintsExpandX));

  mCheckPadTime->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "togglePadTime()");
  mCheckPadTime->SetTextColor(200);
  mCheckPadTime->SetToolTipText("Switch on PadTime calculation");
  mCheckPadTime->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  mCheckPadTime->SetDown(0);
  nextY();

  //---------------------------
  mCheckShowClusters = new TGCheckButton(mContRight, "Overlay clusters");
  mContRight->AddFrame(mCheckShowClusters, new TGLayoutHints(kLHintsExpandX));

  mCheckShowClusters->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "toggleClusters()");
  mCheckShowClusters->SetTextColor(200);
  mCheckShowClusters->SetToolTipText("Switch on ShowClusters calculation");
  mCheckShowClusters->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  mCheckShowClusters->SetDown(0);
  mCheckShowClusters->SetEnabled(kFALSE);

  nextY();

  //---------------------------
  mFlagGroup = new TGVButtonGroup(mContRight, "Cl Flags");
  auto hframe = new TGHorizontalFrame(mFlagGroup);
  const std::string flagTips[NCheckClFlags] = {"Golden", "Split Pad", "Split Time", "Edge", "Single Pad and/or Time"};
  for (int iCheck = 0; iCheck < NCheckClFlags; ++iCheck) {
    mCheckClFlags[iCheck] = new TGCheckButton(hframe, "", 10000 + iCheck);
    mCheckClFlags[iCheck]->SetToolTipText(flagTips[iCheck].data());
    mCheckClFlags[iCheck]->SetDown(1);
    mCheckClFlags[iCheck]->SetEnabled(kFALSE);
    mCheckClFlags[iCheck]->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "showClusters(=-1,-1)");
    hframe->AddFrame(mCheckClFlags[iCheck], new TGLayoutHints(kLHintsExpandX));
  }
  mFlagGroup->AddFrame(hframe, new TGLayoutHints(kLHintsExpandX));
  mFlagGroup->Show();
  mFlagGroup->MoveResize(10, currentY, xsize, (unsigned int)2 * ysize);
  mContRight->AddFrame(mFlagGroup, new TGLayoutHints(kLHintsExpandX));
  mFlagGroup->SetState(kFALSE);
  nextY();
  nextY();

  //---------------------------
  TGTextButton* mFrameExit = new TGTextButton(mContRight, "Exit ROOT");
  mContRight->AddFrame(mFrameExit, new TGLayoutHints(kLHintsExpandX));

  mFrameExit->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "exitRoot()");
  mFrameExit->SetTextColor(200);
  mFrameExit->SetToolTipText("Exit the ROOT process");
  mFrameExit->MoveResize(10, currentY, xsize, (unsigned int)ysize);
  nextY();

  //---------------------------
  mFrameMain->MapSubwindows();
  mFrameMain->MapWindow();
  mFrameMain->SetWindowName("OM");
  mFrameMain->MoveResize(50, 50, (unsigned int)mainx, (unsigned int)currentY + 20);
  mFrameMain->Move(4 * 400 + 10, 10);
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::toggleFFT()
{
  if (mCheckFFT->IsDown()) {
    if (gROOT->GetListOfCanvases()->FindObject("SigIFFT")) {
      return;
    }
    // FFT canvas
    const int w = 400;
    const int h = 400;
    const int hOff = 60;
    const int vOff = 2;
    new TCanvas("SigIFFT", "SigIFFT", -3 * (w + vOff), 0 * h, w, h);
    new TCanvas("SigOFFT", "SigOFFT", -3 * (w + vOff), 1 * h + hOff, w, h);
  } else {
    delete gROOT->GetListOfCanvases()->FindObject("SigIFFT");
    delete gROOT->GetListOfCanvases()->FindObject("SigOFFT");
    delete mHFFTI;
    delete mHFFTO;
    mHFFTI = mHFFTO = nullptr;
  }
}

//______________________________________________________________________________
void SimpleEventDisplayGUI::initOccupancyHists()
{
  const int w = 400;
  const int h = 400;
  const int hOff = 60;
  const int vOff = 2;
  TCanvas* c = nullptr;

  if (mShowSides) {
    // histograms and canvases for occupancy values A-Side
    c = new TCanvas("OccupancyValsA", "OccupancyValsA", 0 * w - 1, 0 * h, w, h);

    c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
               "o2::tpc::SimpleEventDisplayGUI", this,
               "selectSectorExec(int,int,int,TObject*)");

    mHOccupancyA = new TH2F("hOccupancyValsA", "Occupancy Values Side A;x (cm);y (cm)", 330, -250, 250, 330, -250, 250);
    mHOccupancyA->SetStats(kFALSE);
    mHOccupancyA->SetUniqueID(0); // A-Side
    mHOccupancyA->Draw("colz");
    painter::drawSectorsXY(Side::A);

    // histograms and canvases for occupancy values C-Side
    c = new TCanvas("OccupancyValsC", "OccupancyValsC", 0 * w - 1, 1 * h + hOff, w, h);

    c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
               "o2::tpc::SimpleEventDisplayGUI", this,
               "selectSectorExec(int,int,int,TObject*)");
    mHOccupancyC = new TH2F("hOccupancyValsC", "Occupancy Values Side C;x (cm);y (cm)", 330, -250, 250, 330, -250, 250);
    mHOccupancyC->SetStats(kFALSE);
    mHOccupancyC->SetUniqueID(1); // C-Side
    mHOccupancyC->Draw("colz");
    painter::drawSectorsXY(Side::C);
  }

  // histograms and canvases for occupancy values IROC
  c = new TCanvas("OccupancyValsI", "OccupancyValsI", -1 * (w + vOff), 0 * h, w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mHOccupancyIROC = new TH2F("hOccupancyValsIROC", "Occupancy Values IROC;row;pad", 63, 0, 63, 108, -54, 54);
  mHOccupancyIROC->SetDirectory(nullptr);
  mHOccupancyIROC->SetStats(kFALSE);
  mHOccupancyIROC->Draw("colz");

  // histograms and canvases for occupancy values OROC
  c = new TCanvas("OccupancyValsO", "OccupancyValsO", -1 * (w + vOff), 1 * h + hOff, w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mHOccupancyOROC = new TH2F("hOccupancyValsOROC", "Occupancy Values OROC;row;pad", 89, 0, 89, 140, -70, 70);
  mHOccupancyOROC->SetDirectory(nullptr);
  mHOccupancyOROC->SetStats(kFALSE);
  mHOccupancyOROC->Draw("colz");

  fillHists(0, Occupancy);
}

void SimpleEventDisplayGUI::deleteOccupancyHists()
{
  delete gROOT->GetListOfCanvases()->FindObject("OccupancyValsA");
  delete mHOccupancyA;
  mHOccupancyA = nullptr;

  delete gROOT->GetListOfCanvases()->FindObject("OccupancyValsC");
  delete mHOccupancyC;
  mHOccupancyC = nullptr;

  delete gROOT->GetListOfCanvases()->FindObject("OccupancyValsO");
  delete mHOccupancyOROC;
  mHOccupancyOROC = nullptr;

  delete gROOT->GetListOfCanvases()->FindObject("OccupancyValsI");
  delete mHOccupancyIROC;
  mHOccupancyIROC = nullptr;

  delete gROOT->GetListOfCanvases()->FindObject("hOccupancyValsA");
  delete gROOT->GetListOfCanvases()->FindObject("hOccupancyValsC");
  delete gROOT->GetListOfCanvases()->FindObject("hOccupancyValsIROC");
  delete gROOT->GetListOfCanvases()->FindObject("hOccupancyValsOROC");
}

void SimpleEventDisplayGUI::initPadTimeHists()
{
  // histograms and canvases for pad vs. time values IROC
  const int w = 400;
  const int h = 400;
  const int hOff = 60;
  const int vOff = 4;
  TCanvas* c = nullptr;

  const Int_t firstTimeBin = mEvDisp.getFirstTimeBin();
  const Int_t lastTimeBin = mEvDisp.getLastTimeBin();
  const Int_t nTimeBins = mEvDisp.getLastTimeBin() - mEvDisp.getFirstTimeBin();

  c = new TCanvas("PadTimeValsI", "PadTimeValsI", -3 * (w + vOff), 0 * h, w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mHPadTimeIROC = new TH2F("hPadTimeValsI", "PadTime Values IROC;time bin;pad", nTimeBins, firstTimeBin, lastTimeBin, 108, -54, 54);
  // mHPadTimeIROC->SetDirectory(nullptr);
  mHPadTimeIROC->SetStats(kFALSE);
  mHPadTimeIROC->Draw("colz");
  if (!mClustersIROC) {
    mClustersIROC = new TPolyMarker;
    mClustersIROC->SetMarkerSize(1);
    mClustersIROC->SetMarkerStyle(29);
    mClustersIROC->SetMarkerColor(kMagenta);
    mClustersIROC->Draw();
  }

  // histograms and canvases for pad vs. time values OROC
  c = new TCanvas("PadTimeValsO", "PadTimeValsO", -3 * (w + vOff), 1 * h + hOff, w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mHPadTimeOROC = new TH2F("hPadTimeValsO", "PadTime Values OROC;time bin;pad", nTimeBins, firstTimeBin, lastTimeBin, 140, -70, 70);
  // mHPadTimeOROC->SetDirectory(nullptr);
  mHPadTimeOROC->SetStats(kFALSE);
  mHPadTimeOROC->Draw("colz");
  if (!mClustersOROC) {
    mClustersOROC = new TPolyMarker;
    mClustersOROC->SetMarkerSize(1);
    mClustersOROC->SetMarkerStyle(29);
    mClustersOROC->SetMarkerColor(kMagenta);
    mClustersOROC->Draw();
  }
}

void SimpleEventDisplayGUI::deletePadTimeHists()
{
  delete gROOT->GetListOfCanvases()->FindObject("PadTimeValsO");
  delete mHPadTimeOROC;
  mHPadTimeOROC = nullptr;

  delete gROOT->GetListOfCanvases()->FindObject("PadTimeValsI");
  delete mHPadTimeIROC;
  mHPadTimeIROC = nullptr;

  delete gROOT->GetListOfCanvases()->FindObject("hPadTimeValsIROC");
  delete gROOT->GetListOfCanvases()->FindObject("hPadTimeValsOROC");
}

void SimpleEventDisplayGUI::initSingleTBHists()
{
  // histograms and canvases for pad vs. time values IROC
  const int w = 400;
  const int h = 400;
  const int hOff = 60;
  const int vOff = 4;
  TCanvas* c = nullptr;

  const Int_t firstTimeBin = mEvDisp.getFirstTimeBin();
  const Int_t lastTimeBin = mEvDisp.getLastTimeBin();
  const Int_t nTimeBins = mEvDisp.getLastTimeBin() - mEvDisp.getFirstTimeBin();

  c = new TCanvas("SingleTB", "SingleTB", -3 * (w + vOff), 1 * h + hOff, 1.8 * w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mSectorPolyTimeBin = painter::makeSectorHist("hSingleTB");
  // mHPadTimeIROC->SetDirectory(nullptr);
  mSectorPolyTimeBin->SetStats(kFALSE);
  mSectorPolyTimeBin->Draw("colz");

  if (!mClustersRowPad) {
    mClustersRowPad = new TPolyMarker;
    mClustersRowPad->SetMarkerSize(1);
    mClustersRowPad->SetMarkerStyle(29);
    mClustersRowPad->SetMarkerColor(kMagenta);
  }
  mClustersRowPad->Draw();
}

void SimpleEventDisplayGUI::deleteSingleTBHists()
{
  delete gROOT->GetListOfCanvases()->FindObject("SingleTB");
  delete mSectorPolyTimeBin;
  mSectorPolyTimeBin = nullptr;
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::togglePadTime()
{
  if (mCheckPadTime->IsDown()) {
    initPadTimeHists();
    mCheckShowClusters->SetEnabled(kTRUE);
  } else {
    deletePadTimeHists();
    mCheckShowClusters->SetEnabled(kFALSE);
  }
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::toggleOccupancy()
{
  if (mCheckOccupancy->IsDown()) {
    initOccupancyHists();
  } else {
    deleteOccupancyHists();
  }
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::toggleSingleTimeBin()
{
  if (mCheckSingleTB->IsDown()) {
    initSingleTBHists();
    selectTimeBin();
  } else {
    deleteSingleTBHists();
    if (mClustersRowPad) {
      mClustersRowPad->SetPolyMarker(0);
    }
  }
}

//______________________________________________________________________________
void SimpleEventDisplayGUI::toggleClusters()
{
  if (mCheckShowClusters->IsDown()) {
    if (mTPCclusterReader.getTreeSize() > 0) {
      return;
    }
    fs::path p{mInputFileInfo};
    std::string clusterFile = fmt::format("{}/tpc-native-clusters.root", p.parent_path().c_str());
    if (!fs::exists(clusterFile)) {
      LOGP(warn, "Clusters file '{}' does not exist, trying local 'tpc-native-clusters.root'", clusterFile);
      clusterFile = "tpc-native-clusters.root";
      if (!fs::exists(clusterFile)) {
        LOGP(error, "Clusters file '{}' does not exist, can't load clusters", clusterFile);
        return;
      }
    }
    LOGP(info, "loading clusters from file '{}'", clusterFile);
    mTPCclusterReader.init(clusterFile.data());
    gROOT->cd();
    const auto presentEventNumber = mEvDisp.getPresentEventNumber();
    fillClusters(presentEventNumber);
    mFlagGroup->SetState(kTRUE);
    for (int iCheck = 0; iCheck < NCheckClFlags; ++iCheck) {
      mCheckClFlags[iCheck]->SetEnabled(kTRUE);
    }
  } else {
    if (mClustersIROC) {
      mClustersIROC->SetPolyMarker(0);
      mClustersOROC->SetPolyMarker(0);
    }
    mFlagGroup->SetState(kFALSE);
    for (int iCheck = 0; iCheck < NCheckClFlags; ++iCheck) {
      mCheckClFlags[iCheck]->SetEnabled(kFALSE);
    }
  }
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::exitRoot()
{
  mStop = true;
  gApplication->Terminate();
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::update(TString clist)
{
  std::unique_ptr<TObjArray> arr(clist.Tokenize(";"));
  for (auto o : *arr) {
    auto c = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(o->GetName());
    if (c) {
      c->Modified();
      c->Update();
    }
  }
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::resetHists(int type, HistogramType histogramType)
{
  if (!type) {
    switch (histogramType) {
      case MaxValues:
        if (mHMaxA) {
          mHMaxA->Reset();
        }
        if (mHMaxC) {
          mHMaxC->Reset();
        }
        break;
      case Occupancy:
        if (mHOccupancyA) {
          mHOccupancyA->Reset();
        }
        if (mHOccupancyC) {
          mHOccupancyC->Reset();
        }
        break;
      default:
        break;
    }
  }
  switch (histogramType) {
    case MaxValues:
      if (mHMaxIROC) {
        mHMaxIROC->Reset();
      }
      if (mHMaxOROC) {
        mHMaxOROC->Reset();
      }
      break;
    case Occupancy:
      if (mHOccupancyIROC) {
        mHOccupancyIROC->Reset();
      }
      if (mHOccupancyOROC) {
        mHOccupancyOROC->Reset();
      }
      break;
    default:
      break;
  }
}

//__________________________________________________________________________
TH1* SimpleEventDisplayGUI::getBinInfoXY(int& binx, int& biny, float& bincx, float& bincy)
{
  auto pad = (TPad*)gTQSender;
  TObject* select = pad->GetSelected();

  if (!select) {
    return nullptr;
  }

  if (!select->InheritsFrom("TH2")) {
    pad->SetUniqueID(0);
    return nullptr;
  }

  TH1* h = (TH1*)select;
  pad->GetCanvas()->FeedbackMode(kTRUE);

  const int px = pad->GetEventX();
  const int py = pad->GetEventY();
  const float xx = pad->AbsPixeltoX(px);
  const float x = pad->PadtoX(xx);
  const float yy = pad->AbsPixeltoY(py);
  const float y = pad->PadtoX(yy);

  if (h->InheritsFrom(TH2Poly::Class())) {
    auto hPoly = (TH2Poly*)h;
    binx = hPoly->GetXaxis()->FindBin(x);
    biny = hPoly->GetYaxis()->FindBin(y);
    bincx = hPoly->GetXaxis()->GetBinCenter(binx);
    bincy = hPoly->GetYaxis()->GetBinCenter(biny);
    binx = biny = hPoly->FindBin(x, y);
  } else {
    binx = h->GetXaxis()->FindBin(x);
    biny = h->GetYaxis()->FindBin(y);
    bincx = h->GetXaxis()->GetBinCenter(binx);
    bincy = h->GetYaxis()->GetBinCenter(biny);
  }

  return h;
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::drawPadSignal(int event, int x, int y, TObject* o)
{
  //
  // type: name of canvas
  //

  // fmt::print("o: {}, type: {}, name: {}\n", (void*)o, o ? o->IsA()->GetName() : "", o ? o->GetName() : "");
  if (!o) {
    return;
  }

  // check if an event was alreay loaded
  if (!mEvDisp.getNumberOfProcessedEvents()) {
    return;
  }

  // return if mouse is pressed to allow looking at one pad
  if (event != 51) {
    return;
  }

  int binx, biny;
  float bincx, bincy;
  TH1* h = getBinInfoXY(binx, biny, bincx, bincy);
  if (!h) {
    return;
  }
  // fmt::print("binx {}, biny {}, cx {}, cy {}\n", binx, biny, bincx, bincy);

  const auto& mapper = Mapper::instance();
  // int roc = h->GetUniqueID();
  int roc = mSelectedSector;
  TString type;
  const std::string_view objectName(o->GetName());

  // for standard row vs cpad histo, is overwritte in case of TH2Poly sector histo below
  int row = int(TMath::Floor(bincx));

  // find pad and channel
  int pad = -1;

  if (objectName == "hMaxValsIROC" || objectName == "hOccupancyValsIROC") {
    type = "SigI";
  } else if (objectName == "hMaxValsOROC" || objectName == "hOccupancyValsOROC") {
    type = "SigO";
    roc += 36;
  } else if (objectName == "hPadTimeValsI") {
    type = "SigI";
    row = h->GetUniqueID();
  } else if (objectName == "hPadTimeValsO") {
    type = "SigO";
    row = h->GetUniqueID();
    roc += 36;
  } else if (objectName == "hSingleTB") {
    type = "SigI";
    const auto padPosSec = mapper.padPos(binx - 1);
    pad = padPosSec.getPad();
    row = padPosSec.getRow();
    if (bincx > 133) {
      type = "SigO";
      roc += 36;
      row -= mapper.getNumberOfRowsROC(0);
    }
    // fmt::print("roc {}, row {}, pad {}\n", roc, row, pad);
  } else {
    return;
  }
  const int nPads = mapper.getNumberOfPadsInRowROC(roc, row);
  if (pad == -1) {
    const int cpad = int(TMath::Floor(bincy));
    pad = cpad + nPads / 2;
  }

  if (pad < 0 || pad >= (int)nPads) {
    return;
  }
  if (row < 0 || row >= (int)mapper.getNumberOfRowsROC(roc)) {
    return;
  }
  if (roc < 0 || roc >= (int)ROC::MaxROC) {
    return;
  }
  const TString rocType = (roc < 36) ? "I" : "O";

  // draw requested pad signal

  TH1*& hFFT = (roc < 36) ? mHFFTI : mHFFTO;

  TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(type);
  TCanvas* cFFT = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(Form("%sFFT", type.Data()));
  if (c) {
    c->Clear();
    c->cd();
    TH1D* h2 = mEvDisp.makePadSignals(roc, row, pad);
    if (h2) {
      h2->Draw();
      h2->SetStats(0);

      if (cFFT) {
        const bool init = (hFFT == nullptr);
        const double maxTime = h2->GetNbinsX() * 200.e-6;
        hFFT = h2->FFT(hFFT, "MAG M");
        if (hFFT) {
          hFFT->SetStats(0);
          const auto nbinsx = hFFT->GetNbinsX();
          auto xax = hFFT->GetXaxis();
          xax->SetRange(2, nbinsx / 2);
          if (init) {
            xax->Set(nbinsx, xax->GetXmin() / maxTime, xax->GetXmax() / maxTime);
            hFFT->SetNameTitle(Form("hFFT_%sROC", rocType.Data()), "FFT magnitude;frequency (kHz);amplitude");
          }
          hFFT->Scale(2. / (nbinsx - 1));
          cFFT->cd();
          hFFT->Draw();
        }
      }
    }
    if (mCheckSingleTB) {
      TLine l;
      l.SetLineColor(kRed);
      const auto timeBin = mSelTimeBin->GetNumberEntry()->GetIntNumber();
      h = (TH1D*)gROOT->FindObject(fmt::format("PadSignals_{}ROC", rocType.Data()).data());
      if (h) {
        l.DrawLine(timeBin + 0.5, h->GetMinimum(), timeBin + 0.5, h->GetMaximum());
      }
    }
    if (mCheckPadTime && objectName.find("hPadTimeVals") == 0) {
      TLine l;
      l.SetLineColor(kMagenta);
      const auto timeBin = bincx;
      h = (TH1D*)gROOT->FindObject(fmt::format("PadSignals_{}ROC", rocType.Data()).data());
      if (h) {
        l.DrawLine(timeBin + 0.5, h->GetMinimum(), timeBin, h->GetMaximum());
      }
    }
    if (mCheckShowClusters->IsDown()) {
      showClusters(roc, row);
    }
    update(Form("%s;%sFFT;PadTimeVals%s;SingleTB", type.Data(), type.Data(), rocType.Data()));
  }
  //   printf("bin=%03d.%03d(%03d)[%05d], name=%s, ROC=%02d content=%.1f, ev: %d\n",row,pad,cpad,chn,h->GetName(), roc, h->GetBinContent(binx,biny), event);
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::fillHists(int type, HistogramType histogramType)
{
  //
  // type: 0 fill side and sector, 1: fill sector only
  //
  const auto& mapper = Mapper::instance();

  float kEpsilon = 0.000000000001;
  CalPad* pad = histogramType == MaxValues ? mEvDisp.getCalPadMax() : mEvDisp.getCalPadOccupancy();
  TH2F* hSide = nullptr;
  TH2F* hROC = nullptr;
  resetHists(type, histogramType);
  const int runNumber = TString(gSystem->Getenv("RUN_NUMBER")).Atoi();
  // const int eventNumber = mEvDisp.getNumberOfProcessedEvents() - 1;
  const int eventNumber = mEvDisp.getPresentEventNumber();
  const bool eventComplete = mEvDisp.isPresentEventComplete();

  for (int iROC = 0; iROC < 72; iROC++) {
    hROC = histogramType == MaxValues ? mHMaxOROC : mHOccupancyOROC;
    hSide = histogramType == MaxValues ? mHMaxC : mHOccupancyC;
    if (iROC < 36) {
      hROC = histogramType == MaxValues ? mHMaxIROC : mHOccupancyIROC;
    }
    if ((iROC % 36) < 18) {
      hSide = histogramType == MaxValues ? mHMaxA : mHOccupancyA;
    }
    if ((iROC % 36) == (mSelectedSector % 36)) {
      TString title = Form("%s Values %cROC %c%02d (%02d) TF %s%d%s", histogramType == MaxValues ? "Max" : "Occupancy", (iROC < 36) ? 'I' : 'O', (iROC % 36 < 18) ? 'A' : 'C', iROC % 18, iROC, eventComplete ? "" : "(", eventNumber, eventComplete ? "" : ")");
      // TString title = Form("Max Values Run %d Event %d", runNumber, eventNumber);
      if (hROC) {
        hROC->SetTitle(title.Data());
      }
    }
    auto& calRoc = pad->getCalArray(iROC);
    const int nRows = mapper.getNumberOfRowsROC(iROC);
    for (int irow = 0; irow < nRows; irow++) {
      const int nPads = mapper.getNumberOfPadsInRowROC(iROC, irow);
      for (int ipad = 0; ipad < nPads; ipad++) {
        float value = calRoc.getValue(irow, ipad);
        // printf("iROC: %02d, sel: %02d, row %02d, pad: %02d, value: %.5f\n", iROC, mSelectedSector, irow, ipad, value);
        if (TMath::Abs(value) > kEpsilon) {
          if (!type && hSide) {
            const GlobalPosition2D global2D = mapper.getPadCentre(PadSecPos(Sector(iROC % 36), PadPos(irow + (iROC >= 36) * mapper.getNumberOfRowsROC(0), ipad)));
            int binx = 1 + TMath::Nint((global2D.X() + 250.) * hSide->GetNbinsX() / 500.);
            int biny = 1 + TMath::Nint((global2D.Y() + 250.) * hSide->GetNbinsY() / 500.);
            hSide->SetBinContent(binx, biny, value);
          }
          const int nPads = mapper.getNumberOfPadsInRowROC(iROC, irow);
          const int cpad = ipad - nPads / 2;
          if ((iROC % 36 == mSelectedSector % 36) && hROC) {
            // printf("   ->>> Fill: iROC: %02d, sel: %02d, row %02d, pad: %02d, value: %.5f\n", iROC, mSelectedSector, irow, ipad, value);
            hROC->Fill(irow, cpad, value);
          }
        }
        if ((iROC % 36 == mSelectedSector % 36) && hROC) {
          hROC->SetUniqueID(iROC);
        }
      }
    }
  }
  if (!type) {
    update(histogramType == MaxValues ? "MaxValsA;MaxValsC" : "OccupancyValsA;OccupancyValsC");
  }

  update(histogramType == MaxValues ? "MaxValsI;MaxValsO" : "OccupancyValsI;OccupancyValsO");
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::selectSector(int sector)
{
  mSelectedSector = sector % 36;
  mEvDisp.setSelectedSector(mSelectedSector);
  fillHists(1, MaxValues);
  if (mCheckOccupancy->IsDown()) {
    fillHists(1, Occupancy);
  }
  mEvDisp.updateSectorHists();
  selectTimeBin();
}

//__________________________________________________________________________
int SimpleEventDisplayGUI::FindROCFromXY(const float x, const float y, const int side)
{
  //
  //
  //

  const auto& mapper = Mapper::instance();
  float r = TMath::Sqrt(x * x + y * y);
  static const float innerWall = mapper.getPadCentre(PadPos(0, 0)).X() - 5.;
  static const float outerWall = mapper.getPadCentre(PadPos(151, 0)).X() + 5.;
  static const float outerIROC = mapper.getPadCentre(PadPos(62, 0)).X();
  static const float innerOROC = mapper.getPadCentre(PadPos(63, 0)).X();
  static const float betweenROC = (outerIROC + innerOROC) / 2.;

  // check radial boundary
  if (r < innerWall || r > outerWall) {
    return -1;
  }

  // check for IROC or OROC
  int type = 0;

  if (r > betweenROC) {
    type = 1;
  }

  int alpha = TMath::Nint(TMath::ATan2(y, x) / TMath::Pi() * 180);
  //   printf("%6.3f %6.3f %03d\n",x, y, alpha);
  if (alpha < 0) {
    alpha += 360;
  }
  const int roc = alpha / 20 + side * 18 + type * 36;
  return roc;
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::selectSectorExec(int event, int x, int y, TObject* o)
{

  int binx, biny;
  float bincx, bincy;
  TH1* h = getBinInfoXY(binx, biny, bincx, bincy);

  if (!h) {
    return;
  }

  const int side = h->GetUniqueID();
  const int roc = FindROCFromXY(bincx, bincy, side);

  if (roc < 0) {
    return;
  }

  const int sector = roc % 36;

  std::string title("Max Values");
  std::string_view name(h->GetTitle());
  if (name.find("Occupancy") != std::string::npos) {
    title = "Occupancy Values";
  }
  h->SetTitle(fmt::format("{} {}{:02d}", title, (sector < 18) ? 'A' : 'C', sector % 18).data());

  if (sector != mOldHooverdSector) {
    auto pad = (TPad*)gTQSender;
    pad->Modified();
    pad->Update();
    mOldHooverdSector = sector;
  }

  if (event != 11) {
    return;
  }
  // printf("selectSector: %d.%02d.%d = %02d\n", side, sector, roc < 36, roc);
  selectSector(sector);
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::initGUI()
{
  const int w = 400;
  const int h = 400;
  const int hOff = 60;
  const int vOff = 2;
  TCanvas* c = nullptr;

  if (mShowSides) {
    // histograms and canvases for max values A-Side
    c = new TCanvas("MaxValsA", "MaxValsA", 0 * w - 1, 0 * h, w, h);

    c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
               "o2::tpc::SimpleEventDisplayGUI", this,
               "selectSectorExec(int,int,int,TObject*)");

    mHMaxA = new TH2F("hMaxValsA", "Max Values Side A;x (cm);y (cm)", 330, -250, 250, 330, -250, 250);
    mHMaxA->SetStats(kFALSE);
    mHMaxA->SetUniqueID(0); // A-Side
    mHMaxA->Draw("colz");
    painter::drawSectorsXY(Side::A);

    // histograms and canvases for max values C-Side
    c = new TCanvas("MaxValsC", "MaxValsC", 0 * w - 1, 1 * h + hOff, w, h);

    c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
               "o2::tpc::SimpleEventDisplayGUI", this,
               "selectSectorExec(int,int,int,TObject*)");
    mHMaxC = new TH2F("hMaxValsC", "Max Values Side C;x (cm);y (cm)", 330, -250, 250, 330, -250, 250);
    mHMaxC->SetStats(kFALSE);
    mHMaxC->SetUniqueID(1); // C-Side
    mHMaxC->Draw("colz");
    painter::drawSectorsXY(Side::C);
  }

  // histograms and canvases for max values IROC
  c = new TCanvas("MaxValsI", "MaxValsI", -1 * (w + vOff), 0 * h, w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mHMaxIROC = new TH2F("hMaxValsIROC", "Max Values IROC;row;pad", 63, 0, 63, 108, -54, 54);
  mHMaxIROC->SetDirectory(nullptr);
  mHMaxIROC->SetStats(kFALSE);
  mHMaxIROC->Draw("colz");

  // histograms and canvases for max values OROC
  c = new TCanvas("MaxValsO", "MaxValsO", -1 * (w + vOff), 1 * h + hOff, w, h);

  c->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
             "o2::tpc::SimpleEventDisplayGUI", this,
             "drawPadSignal(int,int,int,TObject*)");
  mHMaxOROC = new TH2F("hMaxValsOROC", "Max Values OROC;row;pad", 89, 0, 89, 140, -70, 70);
  mHMaxOROC->SetDirectory(nullptr);
  mHMaxOROC->SetStats(kFALSE);
  mHMaxOROC->Draw("colz");

  // canvases for pad signals
  new TCanvas("SigI", "SigI", -2 * (w + vOff), 0 * h, w, h);
  new TCanvas("SigO", "SigO", -2 * (w + vOff), 1 * h + hOff, w, h);
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::next(int eventNumber)
{
  if (mRunMode == RunMode::Online) {

    if (!mDataAvailable) {
      return;
    }

    mUpdatingDigits = true;
    usleep(1000);
    mNextEvent = true;

    while (mUpdatingDigits) {
      usleep(1000);
    }
    mProcessingEvent = true;
  }

  using Status = CalibRawBase::ProcessStatus;
  Status status = mEvDisp.processEvent(eventNumber);

  const int timeBins = mEvDisp.getNumberOfProcessedTimeBins();

  const int presentEventNumber = mEvDisp.getPresentEventNumber();
  mEventNumber->SetText(Form("%d", presentEventNumber));

  switch (status) {
    case Status::Ok: {
      std::cout << "Read in full event with " << timeBins << " time bins\n";
      break;
    }
    case Status::Truncated: {
      std::cout << "Event is truncated and contains less than " << timeBins << " time bins\n";
      break;
    }
    case Status::NoMoreData: {
      std::cout << "No more data to be read\n";
      // return;
      break;
    }
    case Status::NoReaders: {
      std::cout << "No raw readers configured\n";
      return;
      break;
    }
    default:
      // Do nothing for non-listed values of Status enum
      break;
  }
  // bool res=mEvDisp.processEvent();
  // printf("Next: %d, %d (%d - %d), %d\n",res, ((AliRawReaderGEMDate*)mRawReader)->mEventInFile,((AliRawReaderGEMDate*)mRawReader)->GetCamacData(0),mRawReader->GetEventFromTag(), mRawReader->GetDataSize());
  // printf("Next Event: %d\n",mRawReader->GetEventFromTag());

  fillHists(0, MaxValues);
  if (mCheckOccupancy->IsDown()) {
    fillHists(0, Occupancy);
  }

  if (mRunMode == RunMode::Online) {
    mProcessingEvent = false;
  }

  fillClusters(presentEventNumber);
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::callEventNumber()
{
  const int event = TString(mEventNumber->GetText()).Atoi();
  next(event);
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::runSimpleEventDisplay(std::string_view fileInfo, std::string_view pedestalFile, int firstTimeBin, int lastTimeBin, int nTimeBinsPerCall, uint32_t verbosity, uint32_t debugLevel, int selectedSector, bool showSides)
{
  fair::Logger::SetVerbosity("LOW");
  fair::Logger::SetConsoleSeverity("DEBUG");
  gStyle->SetNumberContours(255);
  if (pedestalFile.size()) {
    TFile f(pedestalFile.data());
    if (f.IsOpen()) {
      CalDet<float>* pedestal = nullptr;
      f.GetObject("Pedestals", pedestal);
      mEvDisp.setPedstals(pedestal);
    }
  }

  mSelectedSector = selectedSector;
  mShowSides = showSides;

  mEvDisp.setupContainers(fileInfo.data(), verbosity, debugLevel);
  mEvDisp.setSkipIncompleteEvents(false); // in case of the online monitor do not skip incomplete events
  mEvDisp.setSelectedSector(mSelectedSector);
  mEvDisp.setLastSelSector(mSelectedSector);
  mEvDisp.setTimeBinsPerCall(nTimeBinsPerCall);
  mEvDisp.setTimeBinRange(firstTimeBin, lastTimeBin);

  initGUI();
  monitorGui();

  next(0);

  mInputFileInfo = fileInfo;

  memset(&mClusterIndex, 0, sizeof(mClusterIndex));
}

//_____________________________________________________________________________
void SimpleEventDisplayGUI::startGUI(int maxTimeBins)
{
  mRunMode = RunMode::Online;

  TApplication evDisp("TPC raw data monitor", nullptr, nullptr);

  mEvDisp.setSkipIncompleteEvents(false); // in case of the online monitor do not skip incomplete events
  mEvDisp.setSelectedSector(mSelectedSector);
  mEvDisp.setLastSelSector(mSelectedSector);
  mEvDisp.setTimeBinsPerCall(maxTimeBins);
  mEvDisp.setTimeBinRange(0, maxTimeBins);

  initGUI();
  monitorGui();

  evDisp.Run(true);
}

//_____________________________________________________________________________
void SimpleEventDisplayGUI::applySignalThreshold()
{
  UInt_t signalThreshold = TString(mSignalThresholdValue->GetText()).Atoi();
  mEvDisp.setSignalThreshold(signalThreshold);
  callEventNumber();
}

//______________________________________________________________________________
void SimpleEventDisplayGUI::selectTimeBin()
{
  if (!mCheckSingleTB->IsDown()) {
    return;
  }
  const auto timeBin = mSelTimeBin->GetNumberEntry()->GetIntNumber();
  mEvDisp.fillSectorHistSingleTimeBin(mSectorPolyTimeBin, timeBin);
  mSectorPolyTimeBin->SetTitle(fmt::format("Sector {:02}, time bin {:6}", mSelectedSector, timeBin).data());
  showClusters(-1, -1);
  update("SingleTB");
}

//______________________________________________________________________________
void SimpleEventDisplayGUI::showClusters(int roc, int row)
{
  static int lastRow = -1;
  static int lastROC = -1;
  static int lastTimeBin = -1;
  const auto timeBin = mSelTimeBin->GetNumberEntry()->GetIntNumber();
  bool forceUpdate = false;
  // fmt::print("roc {}, row {}, lastROC {}, lastRow {}\n", roc, row, lastROC, lastRow);
  if (roc == -1) {
    roc = lastROC;
    forceUpdate = true;
  }
  if ((mTPCclusterReader.getTreeSize() == 0) || (lastRow == row) || (roc == -1)) {
    return;
  }
  if (row == -1) {
    if (lastRow == -1) {
      return;
    }
    row = lastRow;
  }
  lastRow = row;
  lastROC = roc;

  const auto& mapper = Mapper::instance();
  const int sector = roc % 36;
  TPolyMarker* marker = mClustersIROC;
  const int nPads = mapper.getNumberOfPadsInRowROC(roc, row);

  if (roc >= 36) {
    marker = mClustersOROC;
    row += mapper.getNumberOfRowsROC(0); // cluster access is using the global row in sector
  }

  marker->SetPolyMarker(0);
  if (mClustersRowPad) {
    mClustersRowPad->SetPolyMarker(0);
  }
  size_t iSelClusters = 0;
  int selFlags = 0;
  bool golden = mCheckClFlags[0]->IsDown();
  for (int iFlag = 1; iFlag < NCheckClFlags; ++iFlag) {
    selFlags += mCheckClFlags[iFlag]->IsDown() << (iFlag - 1);
  }
  const bool fillSingleTB = mCheckSingleTB->IsDown();
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;

  const int rowMin = fillSingleTB ? 0 : row;
  const int rowMax = fillSingleTB ? constants::MAXGLOBALPADROW : row + 1;

  for (int irow = rowMin; irow < rowMax; ++irow) {
    const auto nClusters = mClusterIndex.nClusters[sector][irow];
    for (size_t icl = 0; icl < nClusters; ++icl) {
      const auto& cl = mClusterIndex.clusters[sector][irow][icl];
      const auto flags = cl.getFlags();
      // fmt::print("flags: {}, selFlags: {}, selGolden: {}, ", flags, selFlags, golden);
      if (((flags == 0) && golden) || (flags & selFlags)) {
        // fmt::print("sel");
        if (row == irow) {
          marker->SetPoint(iSelClusters, cl.getTime() + 0.5, cl.getPad() + 0.5 - nPads / 2.);
          ++iSelClusters;
        }
        if (fillSingleTB && std::abs(cl.getTime() - timeBin) < 2) {
          const auto ly = gpuGeom.LinearPad2Y(sector, irow, cl.getPad() + 0.5);
          mClustersRowPad->SetNextPoint(gpuGeom.Row2X(irow), (sector >= GPUCA_NSLICES / 2) ? -ly : ly);
        }
      }
      // fmt::print("\n");
    }
  }
  // marker->SetPolyMarker(iSelClusters);

  if (forceUpdate) {
    update(Form("PadTimeVals%s;SingleTB", (roc < 36) ? "I" : "O"));
  }
}

//______________________________________________________________________________
void SimpleEventDisplayGUI::fillClusters(Long64_t entry)
{
  if (mTPCclusterReader.getTreeSize() > 0) {
    mTPCclusterReader.read(entry);
    mTPCclusterReader.fillIndex(mClusterIndex, mClusterBuffer, mClusterMCBuffer);
    LOGP(info, "Loaded cluster tree entry {} with {} clusters", entry, mClusterIndex.nClustersTotal);
  }
}
