
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
//  Calibration base class for a single ROC                                  //
//  Contains one char value per pad                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDCalSingleChamberStatus.h"
#include <TMath.h>
#include <Rtypes.h>
#include "TRDBase/TRDCommonParam.h"
#include <FairLogger.h>

using namespace o2::trd;

//_____________________________________________________________________________
TRDCalSingleChamberStatus::TRDCalSingleChamberStatus()
  : mPla(0), mCha(0), mNrows(0), mNcols(0), mNchannels(0), mData(0)
{
  //
  // Default constructor
  //
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus::TRDCalSingleChamberStatus(Int_t p, Int_t c, Int_t cols)
  : mPla(p), mCha(c), mNrows(0), mNcols(cols), mNchannels(0), mData(0)
{
  //
  // Constructor that initializes a given pad plane type
  //

  //
  // The pad plane parameter
  //
  switch (p) {
    case 0:
      if (c == 2) {
        // L0C0 type
        mNrows = 12;
      } else {
        // L0C1 type
        mNrows = 16;
      }
      break;
    case 1:
      if (c == 2) {
        // L1C0 type
        mNrows = 12;
      } else {
        // L1C1 type
        mNrows = 16;
      }
      break;
    case 2:
      if (c == 2) {
        // L2C0 type
        mNrows = 12;
      } else {
        // L2C1 type
        mNrows = 16;
      }
      break;
    case 3:
      if (c == 2) {
        // L3C0 type
        mNrows = 12;
      } else {
        // L3C1 type
        mNrows = 16;
      }
      break;
    case 4:
      if (c == 2) {
        // L4C0 type
        mNrows = 12;
      } else {
        // L4C1 type
        mNrows = 16;
      }
      break;
    case 5:
      if (c == 2) {
        // L5C0 type
        mNrows = 12;
      } else {
        // L5C1 type
        mNrows = 16;
      }
      break;
  };

  mNchannels = mNrows * mNcols;
  if (mNchannels != 0) {
    mData = new Char_t[mNchannels];
  }
  for (Int_t i = 0; i < mNchannels; ++i) {
    mData[i] = 0;
  }
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus::TRDCalSingleChamberStatus(const TRDCalSingleChamberStatus& c)
  : mPla(c.mPla), mCha(c.mCha), mNrows(c.mNrows), mNcols(c.mNcols), mNchannels(c.mNchannels), mData(0)
{
  //
  // TRDCalSingleChamberStatus copy constructor
  //

  mData = new Char_t[mNchannels];
  for (Int_t iBin = 0; iBin < mNchannels; iBin++) {
    mData[iBin] = ((TRDCalSingleChamberStatus&)c).mData[iBin];
  }
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus::~TRDCalSingleChamberStatus()
{
  //
  // TRDCalSingleChamberStatus destructor
  //

  if (mData) {
    delete[] mData;
    mData = 0;
  }
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus& TRDCalSingleChamberStatus::operator=(const TRDCalSingleChamberStatus& c)
{
  //
  // Assignment operator
  //

  if (this == &c) {
    return *this;
  }

  mPla = c.mPla;
  mCha = c.mCha;
  mNrows = c.mNrows;
  mNcols = c.mNcols;
  mNchannels = c.mNchannels;

  if (mData) {
    delete[] mData;
  }
  mData = new Char_t[mNchannels];
  for (Int_t iBin = 0; iBin < mNchannels; iBin++) {
    mData[iBin] = c.mData[iBin];
  }

  return *this;
}

//_____________________________________________________________________________
void TRDCalSingleChamberStatus::Copy(TRDCalSingleChamberStatus& c) const
{
  //
  // Copy function
  //

  Int_t iBin = 0;

  c.mPla = mPla;
  c.mCha = mCha;

  c.mNrows = mNrows;
  c.mNcols = mNcols;

  c.mNchannels = mNchannels;

  if (c.mData) {
    delete[] c.mData;
  }
  c.mData = new Char_t[mNchannels];
  for (iBin = 0; iBin < mNchannels; iBin++) {
    c.mData[iBin] = mData[iBin];
  }
}
