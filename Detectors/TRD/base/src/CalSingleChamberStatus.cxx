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
//  Calibration base class for a single ROC                                  //
//  Contains one char value per pad                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/CalSingleChamberStatus.h"
#include <TMath.h>
#include <Rtypes.h>
#include "TRDBase/CommonParam.h"

using namespace o2::trd;

//_____________________________________________________________________________
CalSingleChamberStatus::CalSingleChamberStatus() = default;

//_____________________________________________________________________________
CalSingleChamberStatus::CalSingleChamberStatus(Int_t p, Int_t c, Int_t cols)
  : mPla(p), mCha(c), mNcols(cols)
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
    mData.resize(mNchannels);
  }
  memset(&mData[0], 0, sizeof(mData[0]) * mData.size());
}

//_____________________________________________________________________________
CalSingleChamberStatus::CalSingleChamberStatus(const CalSingleChamberStatus& c)
  : mPla(c.mPla), mCha(c.mCha), mNrows(c.mNrows), mNcols(c.mNcols), mNchannels(c.mNchannels)
{
  //
  // CalSingleChamberStatus copy constructor
  //

  mData = c.mData;
}

//_____________________________________________________________________________
CalSingleChamberStatus::~CalSingleChamberStatus() = default;

//_____________________________________________________________________________
CalSingleChamberStatus& CalSingleChamberStatus::operator=(const CalSingleChamberStatus& c)
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
  mData = c.mData;

  return *this;
}

//_____________________________________________________________________________
void CalSingleChamberStatus::Copy(CalSingleChamberStatus& c) const
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

  c.mData = mData;
}
