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

#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TQObject.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TObjArray.h"
#include "TROOT.h"
#include "TMath.h"
#include "TApplication.h"

#include <fairlogger/Logger.h>

#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"

#include "TPCMonitor/SimpleEventDisplayGUI.h"

using namespace o2::tpc;

//__________________________________________________________________________
void SimpleEventDisplayGUI::monitorGui()
{
  float xsize = 145;
  float ysize = 25;
  float yoffset = 10;
  float ysize_dist = 2;
  float mainx = 170;
  float mainy = 200;
  int ycount = 0;

  TGMainFrame* mFrameMain = new TGMainFrame(gClient->GetRoot(), 200, 200, kMainFrame | kVerticalFrame);
  mFrameMain->SetLayoutBroken(kTRUE);
  mFrameMain->SetCleanup(kDeepCleanup);

  TGCompositeFrame* mContRight = new TGCompositeFrame(mFrameMain, 155, mainy, kVerticalFrame | kFixedWidth | kFitHeight);
  mFrameMain->AddFrame(mContRight, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY | kLHintsExpandX, 3, 5, 3, 3));

  //---------------------------
  TGTextButton* mFrameNextEvent = new TGTextButton(mContRight, "&Next Event");
  mContRight->AddFrame(mFrameNextEvent, new TGLayoutHints(kLHintsExpandX));

  mFrameNextEvent->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "next(=-1)");
  mFrameNextEvent->SetTextColor(200);
  mFrameNextEvent->SetToolTipText("Go to next event");
  mFrameNextEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), xsize, (unsigned int)ysize);
  ++ycount;

  //---------------------------
  TGTextButton* mFramePreviousEvent = new TGTextButton(mContRight, "&Previous Event");
  mContRight->AddFrame(mFramePreviousEvent, new TGLayoutHints(kLHintsExpandX));
  if (mRunMode == RunMode::Online) {
    mFramePreviousEvent->SetState(kButtonDisabled);
  }

  mFramePreviousEvent->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "next(=-2)");
  mFramePreviousEvent->SetTextColor(200);
  mFramePreviousEvent->SetToolTipText("Go to previous event");
  mFramePreviousEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), xsize, (unsigned int)ysize);
  ++ycount;

  //---------------------------

  TGTextButton* mGoToEvent = new TGTextButton(mContRight, "&Go to event");
  mContRight->AddFrame(mGoToEvent, new TGLayoutHints(kLHintsNormal));

  mGoToEvent->SetTextColor(200);
  mGoToEvent->SetToolTipText("Go to event");
  mGoToEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), 0.65 * xsize, (unsigned int)ysize);
  mGoToEvent->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "callEventNumber()");

  //
  auto* ftbuf = new TGTextBuffer(10);
  ftbuf->AddText(0, "0");
  mEventNumber = new TGTextEntry(mContRight, ftbuf);
  mContRight->AddFrame(mEventNumber, new TGLayoutHints(kFitHeight));
  mEventNumber->MoveResize(0.7 * xsize, yoffset + ycount * (ysize_dist + ysize), 0.3 * xsize, (unsigned int)ysize);
  mEventNumber->SetAlignment(kTextRight);
  ++ycount;

  //---------------------------
  TGTextButton* mApplySignalThreshold = new TGTextButton(mContRight, "&Apply Threshold");
  mContRight->AddFrame(mApplySignalThreshold, new TGLayoutHints(kLHintsNormal));

  mApplySignalThreshold->SetTextColor(200);
  mApplySignalThreshold->SetToolTipText("Apply Threshold");
  mApplySignalThreshold->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), 0.65 * xsize, (unsigned int)ysize);
  mApplySignalThreshold->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "applySignalThreshold()");

  auto* signalThresholdBuf = new TGTextBuffer(10);
  signalThresholdBuf->AddText(0, "0");
  mSignalThresholdValue = new TGTextEntry(mContRight, signalThresholdBuf);
  mSignalThresholdValue->MoveResize(0.7 * xsize, yoffset + ycount * (ysize_dist + ysize), 0.3 * xsize, (unsigned int)ysize);
  mSignalThresholdValue->SetAlignment(kTextRight);
  mSignalThresholdValue->Connect("ReturnPressed()", "o2::tpc::SimpleEventDisplayGUI", this, "applySignalThreshold()");
  ++ycount;

  //---------------------------
  mCheckFFT = new TGCheckButton(mContRight, "Show FFT");
  mContRight->AddFrame(mCheckFFT, new TGLayoutHints(kLHintsExpandX));

  mCheckFFT->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "toggleFFT()");
  mCheckFFT->SetTextColor(200);
  mCheckFFT->SetToolTipText("Switch on FFT calculation");
  mCheckFFT->MoveResize(10, 10 + ysize * 4, xsize, (unsigned int)ysize);
  mCheckFFT->SetDown(0);
  toggleFFT();

  //---------------------------
  mCheckOccupancy = new TGCheckButton(mContRight, "Show Occupancy");
  mContRight->AddFrame(mCheckOccupancy, new TGLayoutHints(kLHintsExpandX));

  mCheckOccupancy->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "toggleOccupancy()");
  mCheckOccupancy->SetTextColor(200);
  mCheckOccupancy->SetToolTipText("Switch on Occupancy calculation");
  mCheckOccupancy->MoveResize(10, 10 + ysize * 5, xsize, (unsigned int)ysize);
  mCheckOccupancy->SetDown(0);
  toggleOccupancy();

  //---------------------------
  TGTextButton* mFrameExit = new TGTextButton(mContRight, "Exit ROOT");
  mContRight->AddFrame(mFrameExit, new TGLayoutHints(kLHintsExpandX));

  mFrameExit->Connect("Clicked()", "o2::tpc::SimpleEventDisplayGUI", this, "exitRoot()");
  mFrameExit->SetTextColor(200);
  mFrameExit->SetToolTipText("Exit the ROOT process");
  mFrameExit->MoveResize(10, 10 + ysize * 6, xsize, (unsigned int)ysize);

  //---------------------------
  mFrameMain->MapSubwindows();
  mFrameMain->MapWindow();
  mFrameMain->SetWindowName("OM");
  mFrameMain->MoveResize(50, 50, (unsigned int)mainx, (unsigned int)mainy);
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

  binx = h->GetXaxis()->FindBin(x);
  biny = h->GetYaxis()->FindBin(y);
  bincx = h->GetXaxis()->GetBinCenter(binx);
  bincy = h->GetYaxis()->GetBinCenter(biny);

  return h;
}

//__________________________________________________________________________
void SimpleEventDisplayGUI::drawPadSignal(int event, int x, int y, TObject* o)
{
  //
  // type: name of canvas
  //

  if (!o) {
    return;
  }

  TString type;
  if (std::string_view(o->GetName()) == "hMaxValsIROC" || std::string_view(o->GetName()) == "hOccupancyValsIROC") {
    type = "SigI";
  } else if (std::string_view(o->GetName()) == "hMaxValsOROC" || std::string_view(o->GetName()) == "hOccupancyValsOROC") {
    type = "SigO";
  } else {
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

  const int row = int(TMath::Floor(bincx));
  const int cpad = int(TMath::Floor(bincy));
  // find pad and channel
  const int roc = h->GetUniqueID();
  if (roc < 0 || roc >= (int)ROC::MaxROC) {
    return;
  }

  const auto& mapper = Mapper::instance();
  if (row < 0 || row >= (int)mapper.getNumberOfRowsROC(roc)) {
    return;
  }

  const int nPads = mapper.getNumberOfPadsInRowROC(roc, row);
  const int pad = cpad + nPads / 2;
  if (pad < 0 || pad >= (int)nPads) {
    return;
  }

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
            hFFT->SetNameTitle(Form("hFFT_%sROC", (roc < 36) ? "I" : "O"), "FFT magnitude;frequency (kHz);amplitude");
          }
          hFFT->Scale(2. / (nbinsx - 1));
          cFFT->cd();
          hFFT->Draw();
        }
      }
    }
    update(Form("%s;%sFFT", type.Data(), type.Data()));
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
