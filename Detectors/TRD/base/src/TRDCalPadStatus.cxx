// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for the single pad status                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TH1F.h>
#include <TH2F.h>
#include <TStyle.h>
#include <TCanvas.h>

#include "TRDBase/TRDGeometryBase.h"
#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "DetectorsCommonDataFormats/DetID.h"

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"
#include "TRDBase/TRDCalSingleChamberStatus.h" // test
#include "TRDBase/TRDCalPadStatus.h"

using namespace o2::trd;

//_____________________________________________________________________________
TRDCalPadStatus::TRDCalPadStatus()
{
  //
  // TRDCalPadStatus default constructor
  //

  for (Int_t idet = 0; idet < kNdet; idet++) {
    mROC[idet] = 0;
  }
}

//_____________________________________________________________________________
TRDCalPadStatus::TRDCalPadStatus(const Text_t* name, const Text_t* title)
{
  //
  // TRDCalPadStatus constructor
  //
  //TRDGeometry fgeom;
  for (Int_t isec = 0; isec < kNsect; isec++) {
    for (Int_t ipla = 0; ipla < kNplan; ipla++) {
      for (Int_t icha = 0; icha < kNcham; icha++) {
        Int_t idet = o2::trd::TRDGeometry::getDetector(ipla, icha, isec);
        //    Int_t idet = fgeom.getDetector(ipla,icha,isec);//TRDGeometryBase::getDetector(ipla,icha,isec);
        mROC[idet] = new TRDCalSingleChamberStatus(ipla, icha, 144);
      }
    }
  }
  mName = name;
  mTitle = title;
}

//_____________________________________________________________________________
TRDCalPadStatus::TRDCalPadStatus(const TRDCalPadStatus& c)
{
  //
  // TRDCalPadStatus copy constructor
  //

  ((TRDCalPadStatus&)c).Copy(*this);
}

//_____________________________________________________________________________
TRDCalPadStatus::~TRDCalPadStatus()
{
  //
  // TRDCalPadStatus destructor
  //

  for (Int_t idet = 0; idet < kNdet; idet++) {
    if (mROC[idet]) {
      delete mROC[idet];
      mROC[idet] = 0;
    }
  }
}

//_____________________________________________________________________________
TRDCalPadStatus& TRDCalPadStatus::operator=(const TRDCalPadStatus& c)
{
  //
  // Assignment operator
  //

  if (this != &c)
    ((TRDCalPadStatus&)c).Copy(*this);
  return *this;
}

//_____________________________________________________________________________
void TRDCalPadStatus::Copy(TRDCalPadStatus& c) const
{
  //
  // Copy function
  //

  for (Int_t idet = 0; idet < kNdet; idet++) {
    if (mROC[idet]) {
      mROC[idet]->Copy(*((TRDCalPadStatus&)c).mROC[idet]);
    }
  }
}

//_____________________________________________________________________________
Bool_t TRDCalPadStatus::CheckStatus(Int_t d, Int_t col, Int_t row, Int_t bitMask) const
{
  //
  // Checks the pad status
  //

  TRDCalSingleChamberStatus* roc = getCalROC(d);
  if (!roc) {
    return kFALSE;
  } else {
    return (roc->getStatus(col, row) & bitMask) ? kTRUE : kFALSE;
  }
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus* TRDCalPadStatus::getCalROC(Int_t p, Int_t c, Int_t s) const
{
  //
  // Returns the readout chamber of this pad
  //
  //TRDGeometry fgeom;
  //return mROC[fgeom.getDetector(p,c,s)];
  return mROC[o2::trd::TRDGeometry::getDetector(p, c, s)];
}

//_____________________________________________________________________________
TH1F* TRDCalPadStatus::MakeHisto1D()
{
  //
  // Make 1D histo
  //

  char name[1000];
  snprintf(name, 1000, "%s Pad 1D", getTitle());
  TH1F* his = new TH1F(name, name, 6, -0.5, 5.5);
  his->GetXaxis()->SetBinLabel(1, "Good");
  his->GetXaxis()->SetBinLabel(2, "Masked");
  his->GetXaxis()->SetBinLabel(3, "PadBridgedLeft");
  his->GetXaxis()->SetBinLabel(4, "PadBridgedRight");
  his->GetXaxis()->SetBinLabel(5, "ReadSecond");
  his->GetXaxis()->SetBinLabel(6, "NotConnected");

  for (Int_t idet = 0; idet < kNdet; idet++) {
    if (mROC[idet]) {
      for (Int_t ichannel = 0; ichannel < mROC[idet]->getNchannels(); ichannel++) {
        Int_t status = (Int_t)mROC[idet]->getStatus(ichannel);
        if (status == 2)
          status = 1;
        if (status == 4)
          status = 2;
        if (status == 8)
          status = 3;
        if (status == 16)
          status = 4;
        if (status == 32)
          status = 5;
        his->Fill(status);
      }
    }
  }

  return his;
}

//_____________________________________________________________________________
TH2F* TRDCalPadStatus::MakeHisto2DSmPl(Int_t sm, Int_t pl)
{
  //
  // Make 2D graph
  //

  gStyle->SetPalette(1);
  TRDGeometry* trdGeo = new TRDGeometry();
  const TRDPadPlane* padPlane0 = trdGeo->getPadPlane(pl, 0);
  Double_t row0 = padPlane0->getRow0();
  Double_t col0 = padPlane0->getCol0();

  char name[1000];
  snprintf(name, 1000, "%s Pad 2D sm %d pl %d", getTitle(), sm, pl);
  TH2F* his = new TH2F(name, name, 88, -TMath::Abs(row0), TMath::Abs(row0), 148, -TMath::Abs(col0), TMath::Abs(col0));

  // Where we begin
  Int_t offsetsmpl = 30 * sm + pl;

  for (Int_t k = 0; k < kNcham; k++) {
    Int_t det = offsetsmpl + k * 6;
    if (mROC[det]) {
      TRDCalSingleChamberStatus* calRoc = mROC[det];
      for (Int_t icol = 0; icol < calRoc->getNcols(); icol++) {
        for (Int_t irow = 0; irow < calRoc->getNrows(); irow++) {
          Int_t binz = 0;
          Int_t kb = kNcham - 1 - k;
          Int_t krow = calRoc->getNrows() - 1 - irow;
          Int_t kcol = calRoc->getNcols() - 1 - icol;
          if (kb > 2)
            binz = 16 * (kb - 1) + 12 + krow + 1 + 2 * (kb + 1);
          else
            binz = 16 * kb + krow + 1 + 2 * (kb + 1);
          Int_t biny = kcol + 1 + 2;
          Float_t value = calRoc->getStatus(icol, irow);
          his->SetBinContent(binz, biny, value);
        }
      }
      for (Int_t icol = 1; icol < 147; icol++) {
        for (Int_t l = 0; l < 2; l++) {
          Int_t binz = 0;
          Int_t kb = kNcham - 1 - k;
          if (kb > 2)
            binz = 16 * (kb - 1) + 12 + 1 + 2 * (kb + 1) - (l + 1);
          else
            binz = 16 * kb + 1 + 2 * (kb + 1) - (l + 1);
          his->SetBinContent(binz, icol, 50.0);
        }
      }
    }
  }
  for (Int_t icol = 1; icol < 147; icol++) {
    his->SetBinContent(88, icol, 50.0);
    his->SetBinContent(87, icol, 50.0);
  }
  for (Int_t irow = 1; irow < 89; irow++) {
    his->SetBinContent(irow, 1, 50.0);
    his->SetBinContent(irow, 2, 50.0);
    his->SetBinContent(irow, 147, 50.0);
    his->SetBinContent(irow, 148, 50.0);
  }

  his->SetXTitle("z (cm)");
  his->SetYTitle("y (cm)");
  his->SetMaximum(50);
  his->SetMinimum(0.0);
  his->SetStats(0);

  return his;
}

//_____________________________________________________________________________
void TRDCalPadStatus::PlotHistos2DSm(Int_t sm, const char* name)
{
  //
  // Make 2D graph
  //

  gStyle->SetPalette(1);
  TCanvas* c1 = new TCanvas(name, name, 50, 50, 600, 800);
  c1->Divide(3, 2);
  c1->cd(1);
  MakeHisto2DSmPl(sm, 0)->Draw("colz");
  c1->cd(2);
  MakeHisto2DSmPl(sm, 1)->Draw("colz");
  c1->cd(3);
  MakeHisto2DSmPl(sm, 2)->Draw("colz");
  c1->cd(4);
  MakeHisto2DSmPl(sm, 3)->Draw("colz");
  c1->cd(5);
  MakeHisto2DSmPl(sm, 4)->Draw("colz");
  c1->cd(6);
  MakeHisto2DSmPl(sm, 5)->Draw("colz");
}
