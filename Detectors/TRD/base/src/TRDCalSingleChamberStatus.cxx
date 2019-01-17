
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
  : fPla(0), fCha(0), fNrows(0), fNcols(0), fNchannels(0), fData(0)
{
  //
  // Default constructor
  //
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus::TRDCalSingleChamberStatus(Int_t p, Int_t c, Int_t cols)
  : fPla(p), fCha(c), fNrows(0), fNcols(cols), fNchannels(0), fData(0)
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
        fNrows = 12;
      } else {
        // L0C1 type
        fNrows = 16;
      }
      break;
    case 1:
      if (c == 2) {
        // L1C0 type
        fNrows = 12;
      } else {
        // L1C1 type
        fNrows = 16;
      }
      break;
    case 2:
      if (c == 2) {
        // L2C0 type
        fNrows = 12;
      } else {
        // L2C1 type
        fNrows = 16;
      }
      break;
    case 3:
      if (c == 2) {
        // L3C0 type
        fNrows = 12;
      } else {
        // L3C1 type
        fNrows = 16;
      }
      break;
    case 4:
      if (c == 2) {
        // L4C0 type
        fNrows = 12;
      } else {
        // L4C1 type
        fNrows = 16;
      }
      break;
    case 5:
      if (c == 2) {
        // L5C0 type
        fNrows = 12;
      } else {
        // L5C1 type
        fNrows = 16;
      }
      break;
  };

  fNchannels = fNrows * fNcols;
  if (fNchannels != 0) {
    fData = new Char_t[fNchannels];
  }
  for (Int_t i = 0; i < fNchannels; ++i) {
    fData[i] = 0;
  }
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus::TRDCalSingleChamberStatus(const TRDCalSingleChamberStatus& c)
  : fPla(c.fPla), fCha(c.fCha), fNrows(c.fNrows), fNcols(c.fNcols), fNchannels(c.fNchannels), fData(0)
{
  //
  // TRDCalSingleChamberStatus copy constructor
  //

  fData = new Char_t[fNchannels];
  for (Int_t iBin = 0; iBin < fNchannels; iBin++) {
    fData[iBin] = ((TRDCalSingleChamberStatus&)c).fData[iBin];
  }
}

//_____________________________________________________________________________
TRDCalSingleChamberStatus::~TRDCalSingleChamberStatus()
{
  //
  // TRDCalSingleChamberStatus destructor
  //

  if (fData) {
    delete[] fData;
    fData = 0;
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

  fPla = c.fPla;
  fCha = c.fCha;
  fNrows = c.fNrows;
  fNcols = c.fNcols;
  fNchannels = c.fNchannels;

  if (fData) {
    delete[] fData;
  }
  fData = new Char_t[fNchannels];
  for (Int_t iBin = 0; iBin < fNchannels; iBin++) {
    fData[iBin] = c.fData[iBin];
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

  c.fPla = fPla;
  c.fCha = fCha;

  c.fNrows = fNrows;
  c.fNcols = fNcols;

  c.fNchannels = fNchannels;

  if (c.fData) {
    delete[] c.fData;
  }
  c.fData = new Char_t[fNchannels];
  for (iBin = 0; iBin < fNchannels; iBin++) {
    c.fData[iBin] = fData[iBin];
  }
}
