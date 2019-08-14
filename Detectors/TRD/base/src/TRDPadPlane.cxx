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
#include <FairLogger.h>

using namespace o2::trd;

//_____________________________________________________________________________
void TRDPadPlane::setTiltingAngle(double t)
{
  //
  // Set the tilting angle of the pads
  //

  mTiltingAngle = t;
  mTiltingTan = TMath::Tan(TMath::Pi() / 180.0 * mTiltingAngle);
}

//_____________________________________________________________________________
int TRDPadPlane::getPadRowNumber(double z) const
{
  //
  // Finds the pad row number for a given z-position in local supermodule system
  //

  int row = 0;
  int nabove = 0;
  int nbelow = 0;
  int middle = 0;

  if ((z > getRow0()) || (z < getRowEnd())) {
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
int TRDPadPlane::getPadRowNumberROC(double z) const
{
  //
  // Finds the pad row number for a given z-position in local ROC system
  //

  int row = 0;
  int nabove = 0;
  int nbelow = 0;
  int middle = 0;

  if ((z > getRow0ROC()) || (z < getRowEndROC())) {
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
int TRDPadPlane::getPadColNumber(double rphi) const
{
  //
  // Finds the pad column number for a given rphi-position
  //

  int col = 0;
  int nabove = 0;
  int nbelow = 0;
  int middle = 0;

  if ((rphi < getCol0()) || (rphi > getColEnd())) {
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

void TRDPadPlane::setNcols(int n)
{
  if (n > MAXCOLS)
    LOG(FATAL) << "MAXCOLS exceeded " << n << " > " << MAXCOLS;
  mNcols = n;
};

void TRDPadPlane::setNrows(int n)
{
  if (n > MAXROWS)
    LOG(FATAL) << "MAXROWS exceeded " << n << " > " << MAXROWS;
  mNrows = n;
};
