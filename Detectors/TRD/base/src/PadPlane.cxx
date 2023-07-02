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

#include "TRDBase/PadPlane.h"
#include <TMath.h>
#include <fairlogger/Logger.h>
#include "DataFormatsTRD/Constants.h"

using namespace o2::trd;
using namespace o2::trd::constants;

//_____________________________________________________________________________
void PadPlane::setTiltingAngle(double t)
{
  //
  // Set the tilting angle of the pads
  //

  mTiltingAngle = t;
  mTiltingTan = TMath::Tan(TMath::DegToRad() * mTiltingAngle);
}

//_____________________________________________________________________________
int PadPlane::getPadRowNumberROC(double z) const
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
int PadPlane::getPadColNumber(double rphi) const
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

void PadPlane::setNcols(int n)
{
  if (n > MAXCOLS) {
    LOG(fatal) << "MAXCOLS exceeded " << n << " > " << MAXCOLS;
  }
  mNcols = n;
};

void PadPlane::setNrows(int n)
{
  if (n > MAXROWS) {
    LOG(fatal) << "MAXROWS exceeded " << n << " > " << MAXROWS;
  }
  mNrows = n;
};

double PadPlane::getPadRow(double z) const
{
  double lengthCorr = mLengthIPad * mInverseLengthOPad;

  // calculate position based on inner pad length
  double padrow = -z * mInverseLengthIPad + mNrows * 0.5;

  // correct row for outer pad rows
  if (padrow <= 1.0) {
    padrow = 1.0 - (1.0 - padrow) * lengthCorr;
  }

  if (padrow >= double(mNrows - 1)) {
    padrow = double(mNrows - 1) + (padrow - double(mNrows - 1)) * lengthCorr;
  }

  // sanity check: is the padrow coordinate reasonable?
  // assert(!(padrow < 0.0 || padrow > double(mNrows)));
  if (padrow < 0.0) {
    padrow = 0;
  } else {
    if (padrow > double(mNrows)) {
      padrow = mNrows;
    }
  }

  return padrow;
}

double PadPlane::getPad(double y, double z) const
{
  int padrow = getPadRow(z);
  double padrowOffset = getPadRowOffsetROC(padrow, z);
  double tiltOffsetY = getTiltOffset(padrow, padrowOffset);

  double pad = y * mInverseWidthIPad + mNcols * 0.5;

  double lengthCorr = mWidthIPad * mInverseWidthOPad;
  // correct row for outer pad rows
  if (pad <= 1.0) {
    pad = 1.0 - (1.0 - pad) * lengthCorr;
  }

  if (pad >= double(mNcols - 1)) {
    pad = double(mNcols - 1) + (pad - double(mNcols - 1)) * lengthCorr;
  }

  double tiltOffsetPad;
  if (pad <= 1.0 || pad >= double(mNcols - 1)) {
    tiltOffsetPad = tiltOffsetY * mInverseWidthOPad;
    pad += tiltOffsetPad;
  } else {
    tiltOffsetPad = tiltOffsetY * mInverseWidthIPad;
    pad += tiltOffsetPad;
  }

  // TODO come back and find why this assert fails on mac arm.
  // assert(!(pad < 0.0 || pad > double(mNcols)));
  if (pad < 0.0) {
    pad = 0;
  } else {
    if (pad > double(mNcols)) {
      pad = mNcols;
    }
  }

  return pad;
}
