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
//  Describes a pad plane of a TRD ROC                                       //
//                                                                           //
//  Contains the information on pad postions, pad dimensions,                //
//  tilting angle, etc.                                                      //
//  It also provides methods to identify the current pad number from         //
//  global coordinates.                                                      //
//  The numbering and coordinates should follow the official convention      //
//  (see David Emschermanns note on TRD convention)                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDPadPlane.h"
#include <TMath.h>

using namespace o2::trd;

//_____________________________________________________________________________
TRDPadPlane::TRDPadPlane()
  : mLayer(0),
    mStack(0),
    mLength(0),
    mWidth(0),
    mLengthRim(0),
    mWidthRim(0),
    mLengthOPad(0),
    mWidthOPad(0),
    mLengthIPad(0),
    mWidthIPad(0),
    mRowSpacing(0),
    mColSpacing(0),
    mNrows(0),
    mNcols(0),
    mTiltingAngle(0),
    mTiltingTan(0),
    mPadRow(nullptr),
    mPadCol(nullptr),
    mPadRowSMOffset(0),
    mAnodeWireOffset(0)
{
  //
  // Default constructor
  //
}

//_____________________________________________________________________________
TRDPadPlane::TRDPadPlane(int layer, int stack)
  : mLayer(layer),
    mStack(stack),
    mLength(0),
    mWidth(0),
    mLengthRim(0),
    mWidthRim(0),
    mLengthOPad(0),
    mWidthOPad(0),
    mLengthIPad(0),
    mWidthIPad(0),
    mRowSpacing(0),
    mColSpacing(0),
    mNrows(0),
    mNcols(0),
    mTiltingAngle(0),
    mTiltingTan(0),
    mPadRow(nullptr),
    mPadCol(nullptr),
    mPadRowSMOffset(0),
    mAnodeWireOffset(0)
{
  //
  // Constructor
  //
}

//_____________________________________________________________________________
TRDPadPlane::~TRDPadPlane()
{
  //
  // TRDPadPlane destructor
  //

  if (mPadRow) {
    delete[] mPadRow;
    mPadRow = nullptr;
  }

  if (mPadCol) {
    delete[] mPadCol;
    mPadCol = nullptr;
  }
}

/*
//_____________________________________________________________________________
void TRDPadPlane::Copy(TObject &p) const
{
  //
  // Copy function
  //

  int iBin = 0;

  ((TRDPadPlane &) p).fLayer           = fLayer;
  ((TRDPadPlane &) p).fStack           = fStack;

  ((TRDPadPlane &) p).fLength          = fLength;
  ((TRDPadPlane &) p).fWidth           = fWidth;
  ((TRDPadPlane &) p).fLengthRim       = fLengthRim;
  ((TRDPadPlane &) p).fWidthRim        = fWidthRim;
  ((TRDPadPlane &) p).fLengthOPad      = fLengthOPad;
  ((TRDPadPlane &) p).fWidthOPad       = fWidthOPad;
  ((TRDPadPlane &) p).fLengthIPad      = fLengthIPad;
  ((TRDPadPlane &) p).fWidthIPad       = fWidthIPad;

  ((TRDPadPlane &) p).fRowSpacing      = fRowSpacing;
  ((TRDPadPlane &) p).fColSpacing      = fColSpacing;

  ((TRDPadPlane &) p).fNrows           = fNrows;
  ((TRDPadPlane &) p).fNcols           = fNcols;

  ((TRDPadPlane &) p).fTiltingAngle    = fTiltingAngle;
  ((TRDPadPlane &) p).fTiltingTan      = fTiltingTan;

  ((TRDPadPlane &) p).fPadRowSMOffset  = fPadRowSMOffset;
  ((TRDPadPlane &) p).fAnodeWireOffset = fAnodeWireOffset;

  if (((TRDPadPlane &) p).fPadRow) {
    delete [] ((TRDPadPlane &) p).fPadRow;
  }
  ((TRDPadPlane &) p).fPadRow = new double[fNrows];
  for (iBin = 0; iBin < fNrows; iBin++) {
    ((TRDPadPlane &) p).fPadRow[iBin] = fPadRow[iBin];
  }

  if (((TRDPadPlane &) p).fPadCol) {
    delete [] ((TRDPadPlane &) p).fPadCol;
  }
  ((TRDPadPlane &) p).fPadCol = new double[fNrows];
  for (iBin = 0; iBin < fNrows; iBin++) {
    ((TRDPadPlane &) p).fPadCol[iBin] = fPadCol[iBin];
  }

  TObject::Copy(p);

}
*/

//_____________________________________________________________________________
void TRDPadPlane::SetTiltingAngle(double t)
{
  //
  // Set the tilting angle of the pads
  //

  mTiltingAngle = t;
  mTiltingTan = TMath::Tan(TMath::Pi() / 180.0 * mTiltingAngle);
}

//_____________________________________________________________________________
int TRDPadPlane::GetPadRowNumber(double z) const
{
  //
  // Finds the pad row number for a given z-position in local supermodule system
  //

  int row = 0;
  int nabove = 0;
  int nbelow = 0;
  int middle = 0;

  if ((z > GetRow0()) || (z < GetRowEnd())) {
    row = -1;

  } else {
    nabove = mNrows + 1;
    nbelow = 0;
    while (nabove - nbelow > 1) {
      middle = (nabove + nbelow) / 2;
      if (z == (mPadRow[middle - 1] + mPadRowSMOffset)) {
        row = middle;
      }
      if (z > (mPadRow[middle - 1] + mPadRowSMOffset)) {
        nabove = middle;
      } else {
        nbelow = middle;
      }
    }
    row = nbelow - 1;
  }

  return row;
}

//_____________________________________________________________________________
int TRDPadPlane::GetPadRowNumberROC(double z) const
{
  //
  // Finds the pad row number for a given z-position in local ROC system
  //

  int row = 0;
  int nabove = 0;
  int nbelow = 0;
  int middle = 0;

  if ((z > GetRow0ROC()) || (z < GetRowEndROC())) {
    row = -1;

  } else {
    nabove = mNrows + 1;
    nbelow = 0;
    while (nabove - nbelow > 1) {
      middle = (nabove + nbelow) / 2;
      if (z == mPadRow[middle - 1]) {
        row = middle;
      }
      if (z > mPadRow[middle - 1]) {
        nabove = middle;
      } else {
        nbelow = middle;
      }
    }
    row = nbelow - 1;
  }

  return row;
}

//_____________________________________________________________________________
int TRDPadPlane::GetPadColNumber(double rphi) const
{
  //
  // Finds the pad column number for a given rphi-position
  //

  int col = 0;
  int nabove = 0;
  int nbelow = 0;
  int middle = 0;

  if ((rphi < GetCol0()) || (rphi > GetColEnd())) {
    col = -1;

  } else {
    nabove = mNcols;
    nbelow = 0;
    while (nabove - nbelow > 1) {
      middle = (nabove + nbelow) / 2;
      if (rphi == mPadCol[middle]) {
        col = middle;
      }
      if (rphi > mPadCol[middle]) {
        nbelow = middle;
      } else {
        nabove = middle;
      }
    }
    col = nbelow;
  }

  return col;
}

ClassImp(TRDPadPlane)
