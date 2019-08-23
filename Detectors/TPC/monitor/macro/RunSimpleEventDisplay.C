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
#include "TObjString.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include "TPaveText.h"
#include "TF1.h"
#include "TStopwatch.h"
#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TFile.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TGButton.h"
#include "TCanvas.h"
#include "TObjArray.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TPRegexp.h"
#include "TMath.h"
#include "TApplication.h"
#include "TPCMonitor/SimpleEventDisplay.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "FairLogger.h"
#include <iostream>
#include <fstream>
#include <memory>
#endif
using namespace std;
using namespace o2::tpc;
/*
.L RunSimpleEventDisplay.C+
RunSimpleEventDisplay("GBTx0_Run005:0:0;GBTx1_Run005:1:0")
*/

SimpleEventDisplay mEvDisp;
const Mapper& mMapper = Mapper::instance();

TObjArray mArrCanvases;
Int_t mOldHooverdSector = -1;
Int_t mSelectedSector = 0;
Int_t mMaxEvents = 100000000;
TH2F* mHMaxA = nullptr;
TH2F* mHMaxC = nullptr;
TH2F* mHMaxIROC = nullptr;
TH2F* mHMaxOROC = nullptr;
TH2F* mHPbVsCkvEle = nullptr; //Pb-glass vs. cherenkov
TH2F* mHPbVsCkvPio = nullptr; //Pb-glass vs. cherenkov
TH1F* mPulseEleMax = nullptr;
TH1F* mPulseEleTot = nullptr;
TH1F* fdEdxEleMax = nullptr;
TH1F* fdEdxEleTot = nullptr;
TH1F* mPulsePioMax = nullptr;
TH1F* mPulsePioTot = nullptr;
TH1F* fdEdxPioMax = nullptr;
TH1F* fdEdxPioTot = nullptr;
TH1F* mHDriftTime = nullptr;
TH1F* mHNcls = nullptr;
TH1* mHFFTO = nullptr;
TH1* mHFFTI = nullptr;
TGCheckButton* mCheckFFT = nullptr;
TGTextEntry* mEventNumber = nullptr;

Double_t mElePosMax = 0.;
Double_t mElePosTot = 0.;
Double_t mPioPosMax = 0.;
Double_t mPioPosTot = 0.;
Double_t mEleResMax = 0.;
Double_t mEleResTot = 0.;
Double_t mPioResMax = 0.;
Double_t mPioResTot = 0.;

Int_t mBigEventSize = 8000;

TObjArray mArrHistCamac(100); //array with histograms for camac data

void ToggleFFT();

//__________________________________________________________________________
void MonitorGui()
{
  Float_t xsize = 145;
  Float_t ysize = 25;
  Float_t yoffset = 10;
  Float_t ysize_dist = 2;
  Float_t mainx = 170;
  Float_t mainy = 170;
  int ycount = 0;

  TGMainFrame* mFrameMain = new TGMainFrame(gClient->GetRoot(), 200, 200, kMainFrame | kVerticalFrame);
  mFrameMain->SetLayoutBroken(kTRUE);

  TGCompositeFrame* mContRight = new TGCompositeFrame(mFrameMain, 155, mainy, kVerticalFrame | kFixedWidth | kFitHeight);
  mFrameMain->AddFrame(mContRight, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY | kLHintsExpandX, 3, 5, 3, 3));

  //---------------------------
  TGTextButton* mFrameNextEvent = new TGTextButton(mContRight, "&Next Event");
  mContRight->AddFrame(mFrameNextEvent, new TGLayoutHints(kLHintsExpandX));

  mFrameNextEvent->SetCommand("Next(-1)");
  mFrameNextEvent->SetTextColor(200);
  mFrameNextEvent->SetToolTipText("Go to next event");
  //mFrameNextEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), xsize, (UInt_t)ysize);
  mFrameNextEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), xsize, (UInt_t)ysize);
  ++ycount;

  //---------------------------
  TGTextButton* mFramePreviousEvent = new TGTextButton(mContRight, "&Previous Event");
  mContRight->AddFrame(mFramePreviousEvent, new TGLayoutHints(kLHintsExpandX));

  mFramePreviousEvent->SetCommand("Next(-2)");
  mFramePreviousEvent->SetTextColor(200);
  mFramePreviousEvent->SetToolTipText("Go to previous event");
  mFramePreviousEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), xsize, (UInt_t)ysize);
  ++ycount;

  //---------------------------

  //TGCompositeFrame* mContGoToEvent = new TGCompositeFrame(mContRight, 155, ysize, kHorizontalFrame | kFitWidth | kFixedHeight);
  //mContRight->AddFrame(mContGoToEvent, new TGLayoutHints(kLHintsExpandX));
  //mContGoToEvent->Move(10, yoffset + ycount * (ysize_dist + ysize));

  TGTextButton* mGoToEvent = new TGTextButton(mContRight, "&Go to event");
  //mContGoToEvent->AddFrame(mGoToEvent, new TGLayoutHints(kLHintsExpandX));
  mContRight->AddFrame(mGoToEvent, new TGLayoutHints(kLHintsNormal));

  mGoToEvent->SetTextColor(200);
  mGoToEvent->SetToolTipText("Go to event");
  mGoToEvent->MoveResize(10, yoffset + ycount * (ysize_dist + ysize), 0.65 * xsize, (UInt_t)ysize);
  mGoToEvent->SetCommand("CallEventNumber()");

  //
  auto* ftbuf = new TGTextBuffer(10);
  ftbuf->AddText(0, "0");
  mEventNumber = new TGTextEntry(mContRight, ftbuf);
  mContRight->AddFrame(mEventNumber, new TGLayoutHints(kFitHeight));
  mEventNumber->MoveResize(0.7 * xsize, yoffset + ycount * (ysize_dist + ysize), 0.3 * xsize, (UInt_t)ysize);
  mEventNumber->SetAlignment(kTextRight);
  ++ycount;

  //---------------------------
  //TGTextButton*  mFrameRewindEvent  = new TGTextButton(mContRight,  "Rewind Events"           );
  //mContRight->AddFrame(mFrameRewindEvent, new TGLayoutHints(kLHintsExpandX));

  //mFrameRewindEvent->SetCommand( "RewindEvents()");
  //mFrameRewindEvent->SetTextColor(200);
  //mFrameRewindEvent->SetToolTipText("Rewind Events to loop again");
  //mFrameRewindEvent->MoveResize(10, 10+ysize*3, xsize, (UInt_t)ysize);

  mCheckFFT = new TGCheckButton(mContRight, "Show FFT");
  mContRight->AddFrame(mCheckFFT, new TGLayoutHints(kLHintsExpandX));

  mCheckFFT->SetCommand("ToggleFFT()");
  mCheckFFT->SetTextColor(200);
  mCheckFFT->SetToolTipText("Switch on FFT calculation");
  mCheckFFT->MoveResize(10, 10 + ysize * 4, xsize, (UInt_t)ysize);
  mCheckFFT->SetDown(1);
  ToggleFFT();

  //---------------------------
  TGTextButton* mFrameExit = new TGTextButton(mContRight, "Exit ROOT");
  mContRight->AddFrame(mFrameExit, new TGLayoutHints(kLHintsExpandX));

  mFrameExit->SetCommand("ExitRoot()");
  mFrameExit->SetTextColor(200);
  mFrameExit->SetToolTipText("Exit the ROOT process");
  mFrameExit->MoveResize(10, 10 + ysize * 5, xsize, (UInt_t)ysize);

  //---------------------------
  mFrameMain->MapSubwindows();
  mFrameMain->MapWindow();
  mFrameMain->SetWindowName("OM");
  mFrameMain->MoveResize(50, 50, (UInt_t)mainx, (UInt_t)mainy);
}

//__________________________________________________________________________
void ToggleFFT()
{
  if (mCheckFFT->IsDown()) {
    if (gROOT->GetListOfCanvases()->FindObject("SigIFFT")) {
      return;
    }
    //FFT canvas
    const Int_t w = 400;
    const Int_t h = 400;
    new TCanvas("SigIFFT", "SigIFFT", 3 * w, 0 * h, w, h);
    new TCanvas("SigOFFT", "SigOFFT", 3 * w, 1 * h, w, h);
  } else {
    delete gROOT->GetListOfCanvases()->FindObject("SigIFFT");
    delete gROOT->GetListOfCanvases()->FindObject("SigOFFT");
    delete mHFFTI;
    delete mHFFTO;
    mHFFTI = mHFFTO = nullptr;
  }
}

//__________________________________________________________________________
void ExitRoot()
{
  gApplication->Terminate();
}

//__________________________________________________________________________
void RewindEvents()
{
  mEvDisp.rewindEvents();
  //if (mRawReader) mRawReader->rewindEvents();
  //printf("RewindEvents: not implemented\n");
}

//__________________________________________________________________________
void Update(TString clist)
{
  TObjArray* arr = clist.Tokenize(";");
  TIter next(arr);
  TObject* o;
  while ((o = next())) {
    TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(o->GetName());
    if (c) {
      c->Modified();
      c->Update();
    }
  }
}

//__________________________________________________________________________
void ResetHists(Int_t type)
{
  if (!type) {
    if (mHMaxA)
      mHMaxA->Reset();
    if (mHMaxC)
      mHMaxC->Reset();
  }
  if (mHMaxIROC)
    mHMaxIROC->Reset();
  if (mHMaxOROC)
    mHMaxOROC->Reset();
}

//__________________________________________________________________________
TH1* GetBinInfoXY(Int_t& binx, Int_t& biny, Float_t& bincx, Float_t& bincy)
{
  TObject* select = gPad->GetSelected();
  if (!select)
    return nullptr;
  if (!select->InheritsFrom("TH2")) {
    gPad->SetUniqueID(0);
    return nullptr;
  }
  TH1* h = (TH1*)select;
  gPad->GetCanvas()->FeedbackMode(kTRUE);

  int px = gPad->GetEventX();
  int py = gPad->GetEventY();
  Float_t xx = gPad->AbsPixeltoX(px);
  Float_t x = gPad->PadtoX(xx);
  binx = h->GetXaxis()->FindBin(x);
  Float_t yy = gPad->AbsPixeltoY(py);
  Float_t y = gPad->PadtoX(yy);
  biny = h->GetYaxis()->FindBin(y);
  bincx = h->GetXaxis()->GetBinCenter(binx);
  bincy = h->GetYaxis()->GetBinCenter(biny);
  //printf("binx, biny: %d %d\n",binx,biny);

  return h;
}

//__________________________________________________________________________
void DrawPadSignal(TString type)
{
  //
  // type: name of canvas
  //

  // check if an event was alreay loaded
  if (!mEvDisp.getNumberOfProcessedEvents())
    return;

  //return if mouse is pressed to allow looking at one pad
  Int_t event = gPad->GetEvent();
  if (event != 51)
    return;

  Int_t binx, biny;
  Float_t bincx, bincy;
  TH1* h = GetBinInfoXY(binx, biny, bincx, bincy);
  if (!h)
    return;
  Int_t row = Int_t(TMath::Floor(bincx));
  Int_t cpad = Int_t(TMath::Floor(bincy));
  //find pad and channel
  Int_t roc = h->GetUniqueID();
  if (roc < 0 || roc >= (Int_t)ROC::MaxROC)
    return;
  if (row < 0 || row >= (Int_t)mMapper.getNumberOfRowsROC(roc))
    return;
  const int nPads = mMapper.getNumberOfPadsInRowROC(roc, row);
  Int_t pad = cpad + nPads / 2;
  //printf("row %d, cpad %d, pad %d, nPads %d\n", row, cpad, pad, nPads);
  if (pad < 0 || pad >= (Int_t)nPads)
    return;
  //   Int_t chn = tpcROC->GetRowIndexes(roc)[row]+pad;
  //draw requested pad signal

  TH1*& hFFT = (roc < 36) ? mHFFTI : mHFFTO;

  TCanvas* c = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(type);
  TCanvas* cFFT = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(Form("%sFFT", type.Data()));
  if (c) {
    c->Clear();
    c->cd();
    //mRawReader->Reset();
    TH1D* h2 = mEvDisp.makePadSignals(roc, row, pad);
    if (h2) {
      h2->GetXaxis()->SetRangeUser(0, mEvDisp.getNumberOfProcessedTimeBins() + 5);
      h2->Draw();
      h2->SetStats(0);

      if (cFFT) {
        const bool init = (hFFT == nullptr);
        const double maxTime = h2->GetNbinsX() * 200.e-6;
        hFFT = h2->FFT(hFFT, "MAG M");
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
    Update(Form("%s;%sFFT", type.Data(), type.Data()));
  }
  //   printf("bin=%03d.%03d(%03d)[%05d], name=%s, ROC=%02d content=%.1f, ev: %d\n",row,pad,cpad,chn,h->GetName(), roc, h->GetBinContent(binx,biny), event);
}

//__________________________________________________________________________
void FillMaxHists(Int_t type = 0)
{
  //
  // type: 0 fill side and sector, 1: fill sector only
  //
  Float_t kEpsilon = 0.000000000001;
  CalPad* pad = mEvDisp.getCalPadMax();
  TH2F* hSide = nullptr;
  TH2F* hROC = nullptr;
  ResetHists(type);
  const int runNumber = TString(gSystem->Getenv("RUN_NUMBER")).Atoi();
  //const int eventNumber = mEvDisp.getNumberOfProcessedEvents() - 1;
  const int eventNumber = mEvDisp.getPresentEventNumber();
  const bool eventComplete = mEvDisp.isPresentEventComplete();
  for (Int_t iROC = 0; iROC < 72; iROC++) {
    // TODO: remove again at some point
    if (iROC % 36 != 0) {
      continue;
    }
    hROC = mHMaxOROC;
    hSide = mHMaxC;
    if (iROC < 36)
      hROC = mHMaxIROC;
    if (iROC % 36 < 18)
      hSide = mHMaxA;
    if (iROC % 36 == mSelectedSector % 36) {
      TString title = Form("Max Values %cROC %c%02d (%02d) Event %s%d%s", (iROC < 36) ? 'I' : 'O', (iROC % 36 < 18) ? 'A' : 'C', iROC % 18, iROC, eventComplete ? "" : "(", eventNumber, eventComplete ? "" : ")");
      //TString title = Form("Max Values Run %d Event %d", runNumber, eventNumber);
      if (hROC) {
        hROC->SetTitle(title.Data());
      }
    }
    auto& calRoc = pad->getCalArray(iROC);
    const int nRows = mMapper.getNumberOfRowsROC(iROC);
    for (Int_t irow = 0; irow < nRows; irow++) {
      const int nPads = mMapper.getNumberOfPadsInRowROC(iROC, irow);
      for (Int_t ipad = 0; ipad < nPads; ipad++) {
        Float_t value = calRoc.getValue(irow, ipad);
        //printf("iROC: %02d, sel: %02d, row %02d, pad: %02d, value: %.5f\n", iROC, mSelectedSector, irow, ipad, value);
        if (TMath::Abs(value) > kEpsilon) {
          if (!type && hSide) {
            const GlobalPosition2D global2D = mMapper.getPadCentre(PadSecPos(Sector(iROC % 36), PadPos(irow, ipad + (iROC >= 36) * mMapper.getNumberOfRowsROC(iROC))));
            Int_t binx = 1 + TMath::Nint((global2D.X() + 250.) * hSide->GetNbinsX() / 500.);
            Int_t biny = 1 + TMath::Nint((global2D.Y() + 250.) * hSide->GetNbinsY() / 500.);
            hSide->SetBinContent(binx, biny, value);
          }
          const int nPads = mMapper.getNumberOfPadsInRowROC(iROC, irow);
          const Int_t cpad = ipad - nPads / 2;
          if ((iROC % 36 == mSelectedSector % 36) && hROC) {
            //printf("   ->>> Fill: iROC: %02d, sel: %02d, row %02d, pad: %02d, value: %.5f\n", iROC, mSelectedSector, irow, ipad, value);
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
    Update("MaxValsA;MaxValsC");
  }

  Update("MaxValsI;MaxValsO");
}

//__________________________________________________________________________
void FillMaxHistsSide()
{
}

//__________________________________________________________________________
void FillMaxHistsSector()
{
}

//__________________________________________________________________________
void SelectSector(Int_t sector)
{
  mSelectedSector = sector % 36;
  mEvDisp.setSelectedSector(mSelectedSector);
  FillMaxHists(1);
}

//__________________________________________________________________________
Int_t FindROCFromXY(const Float_t x, const Float_t y, const Int_t side)
{
  //
  //
  //

  Float_t r = TMath::Sqrt(x * x + y * y);
  static const float innerWall = mMapper.getPadCentre(PadPos(0, 0)).X() - 5.;
  static const float outerWall = mMapper.getPadCentre(PadPos(151, 0)).X() + 5.;
  static const float outerIROC = mMapper.getPadCentre(PadPos(62, 0)).X();
  static const float innerOROC = mMapper.getPadCentre(PadPos(63, 0)).X();
  static const float betweenROC = (outerIROC + innerOROC) / 2.;
  //check radial boundary
  if (r < innerWall || r > outerWall)
    return -1;
  //check for IROC or OROC
  Int_t type = 0;
  if (r > betweenROC)
    type = 1;
  Int_t alpha = TMath::Nint(TMath::ATan2(y, x) / TMath::Pi() * 180);
  //   printf("%6.3f %6.3f %03d\n",x, y, alpha);
  if (alpha < 0)
    alpha += 360;
  Int_t roc = alpha / 20 + side * 18 + type * 36;
  return roc;
}

//__________________________________________________________________________
void SelectSectorExec()
{

  Int_t binx, biny;
  Float_t bincx, bincy;
  TH1* h = GetBinInfoXY(binx, biny, bincx, bincy);
  if (!h)
    return;
  Int_t side = h->GetUniqueID();
  Int_t roc = FindROCFromXY(bincx, bincy, side);
  if (roc < 0)
    return;
  Int_t sector = roc % 36;
  h->SetTitle(Form("Max Values %c%02d", (sector < 18) ? 'A' : 'C', sector % 18));
  if (sector != mOldHooverdSector) {
    gPad->Modified();
    gPad->Update();
    mOldHooverdSector = sector;
  }
  int event = gPad->GetEvent();
  if (event != 11)
    return;
  //   printf("SelectSector: %d.%02d.%d = %02d\n",side,sector,roc<36,roc);
  SelectSector(sector);
}

//__________________________________________________________________________
void InitGUI()
{
  Int_t w = 400;
  Int_t h = 400;
  TCanvas* c = nullptr;
  //histograms and canvases for max values A-Side
  //histograms for the sides are not needed for the GEM test, so they are commented...

  //   c = new TCanvas("MaxValsA","MaxValsA",0*w,0*h,w,h);
  //   c->AddExec("findSec","SelectSectorExec()");
  //   mHMaxA=new TH2F("hMaxValsA","Max Values Side A;x [cm];y [cm]",330,-250,250,330,-250,250);
  //   mHMaxA->SetStats(kFALSE);
  //   mHMaxA->SetUniqueID(0); //A-Side
  //   mHMaxA->Draw("colz");
  //histograms and canvases for max values C-Side
  //   c = new TCanvas("MaxValsC","MaxValsC",0*w,1*h,w,h);
  //   c->AddExec("findSec","SelectSectorExec()");
  //   mHMaxC=new TH2F("hMaxValsC","Max Values Side C;x [cm];y [cm]",330,-250,250,330,-250,250);
  //   mHMaxC->SetStats(kFALSE);
  //   mHMaxC->SetUniqueID(1); //C-Side
  //   mHMaxC->Draw("colz");

  //histograms and canvases for max values IROC
  // For the GEM test only the IROC is needed, so comment the OROC

  c = new TCanvas("MaxValsI", "MaxValsI", 1 * w, 0 * h, w, h);
  c->AddExec("padSig", "DrawPadSignal(\"SigI\")");
  mHMaxIROC = new TH2F("hMaxValsIROC", "Max Values IROC;row;pad", 63, 0, 63, 108, -54, 54);
  mHMaxIROC->SetDirectory(nullptr);
  //mHMaxIROC->GetYaxis()->SetRangeUser(-20,15);
  mHMaxIROC->SetStats(kFALSE);
  mHMaxIROC->Draw("colz");

  //histograms and canvases for max values OROC
  c = new TCanvas("MaxValsO", "MaxValsO", 1 * w, 1 * h, w, h);
  c->AddExec("padSig", "DrawPadSignal(\"SigO\")");
  //mHMaxOROC = new TH2F("hMaxValsOROC", "Max Values OROC;row;pad", 96, 0, 96, 140, -70, 70);
  mHMaxOROC = new TH2F("hMaxValsOROC", "Max Values OROC;row;pad", 89, 0, 89, 140, -70, 70);
  mHMaxOROC->SetDirectory(nullptr);
  mHMaxOROC->SetStats(kFALSE);
  mHMaxOROC->Draw("colz");

  //canvases for pad signals
  new TCanvas("SigI", "SigI", 2 * w, 0 * h, w, h);
  new TCanvas("SigO", "SigO", 2 * w, 1 * h, w, h);
}

//__________________________________________________________________________
void Next(int eventNumber = -1)
{
  printf("Calling event number %d\n", eventNumber);
  //Int_t ev=mRawReader->NextEvent();
  //if (!ev) return;
  using Status = CalibRawBase::ProcessStatus;
  Status status = mEvDisp.processEvent(eventNumber);
  //const Int_t timeBins = mEvDisp.getTimeBinsPerCall();
  const Int_t timeBins = mEvDisp.getNumberOfProcessedTimeBins();

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
      return;
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
  //Bool_t res=mEvDisp.processEvent();
  //printf("Next: %d, %d (%d - %d), %d\n",res, ((AliRawReaderGEMDate*)mRawReader)->mEventInFile,((AliRawReaderGEMDate*)mRawReader)->GetCamacData(0),mRawReader->GetEventFromTag(), mRawReader->GetDataSize());
  //printf("Next Event: %d\n",mRawReader->GetEventFromTag());
  printf("Next Event\n");
  FillMaxHists();
}

//__________________________________________________________________________
void CallEventNumber()
{
  const int event = TString(mEventNumber->GetText()).Atoi();
  Next(event);
}

//__________________________________________________________________________
void RunSimpleEventDisplay(TString fileInfo, TString pedestalFile = "", Int_t nTimeBinsPerCall = 500, uint32_t verbosity = 0, uint32_t debugLevel = 0)
{
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");
  if (!pedestalFile.IsNull()) {
    TFile f(pedestalFile);
    if (f.IsOpen()) {
      CalDet<float>* pedestal = nullptr;
      f.GetObject("Pedestals", pedestal);
      mEvDisp.setPedstals(pedestal);
    }
  }
  mEvDisp.setupContainers(fileInfo, verbosity, debugLevel);
  mEvDisp.setSelectedSector(mSelectedSector);
  mEvDisp.setLastSelSector(mSelectedSector);
  mEvDisp.setTimeBinsPerCall(nTimeBinsPerCall);
  mEvDisp.setTimeBinRange(0, nTimeBinsPerCall);
  InitGUI();
  //  while (mRawReader->NextEvent() && mRawReader->GetEventFromTag()==0) Next();
  MonitorGui();
  //   SelectSector(15);
  // select first event
  Next(0);
}

//__________________________________________________________________________
void GetBinMinMax(const TH1* hist, const Float_t frac, Int_t& bin1, Int_t& bin2)
{

  const Int_t binMax = hist->GetMaximumBin();
  const Double_t contMax = hist->GetBinContent(binMax);
  bin1 = binMax;
  bin2 = binMax;
  while ((bin1--) > binMax / 3.)
    if (hist->GetBinContent(bin1) < frac * contMax)
      break;
  while ((bin2++) < binMax * 3.)
    if (hist->GetBinContent(bin2) < frac * contMax)
      break;
}

//__________________________________________________________________________
void FitGaus(TH1* histEle, TH1* histPio, TPaveText* text, Double_t& posEle, Double_t& resEle, Double_t& posPio, Double_t& resPio)
{
  //
  // fit gaus to histograms and add text to the pave
  //
  TF1* mEle = new TF1("gEle", "gaus", histEle->GetXaxis()->GetXmin(), histEle->GetXaxis()->GetXmax());
  TF1* mPio = new TF1("gPio", "gaus", histPio->GetXaxis()->GetXmin(), histPio->GetXaxis()->GetXmax());
  mEle->SetLineColor(histEle->GetLineColor());
  mPio->SetLineColor(histPio->GetLineColor());
  mEle->SetLineWidth(2);
  mPio->SetLineWidth(2);
  const Float_t frac = 0.1;
  Int_t bin1 = 0, bin2 = 0;

  GetBinMinMax(histEle, frac, bin1, bin2);
  histEle->Fit(mEle, "", "Q0", histEle->GetXaxis()->GetBinLowEdge(bin1), histEle->GetXaxis()->GetBinUpEdge(bin2));

  GetBinMinMax(histPio, frac, bin1, bin2);
  histPio->Fit(mPio, "", "Q0", histPio->GetXaxis()->GetBinLowEdge(bin1), histPio->GetXaxis()->GetBinUpEdge(bin2));

  posEle = mEle->GetParameter(1);
  resEle = mEle->GetParameter(2);
  posPio = mPio->GetParameter(1);
  resPio = mPio->GetParameter(2);
  text->AddText(Form("e: %.2f #pm %.2f (%.2f%%)", posEle, resEle, resEle / posEle * 100));
  text->AddText(Form("#pi: %.2f #pm %.2f (%.2f%%)", posPio, resPio, resPio / posPio * 100));
  text->AddText(Form("Separation: %.2f#sigma", TMath::Abs(posEle - posPio) / ((resEle + resPio) / 2.)));
}
