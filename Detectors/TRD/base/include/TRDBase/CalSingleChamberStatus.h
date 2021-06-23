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

#ifndef O2_TRD_CALSINGLECHAMBERSTATUS_H
#define O2_TRD_CALSINGLECHAMBERSTATUS_H

#include <Rtypes.h>
namespace o2
{
namespace trd
{

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRD calibration base class containing status values for one ROC       //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
class CalSingleChamberStatus
{

 public:
  enum { kMasked = 2,
         kPadBridgedLeft = 4,
         kPadBridgedRight = 8,
         kReadSecond = 16,
         kNotConnected = 32 };

  CalSingleChamberStatus();
  CalSingleChamberStatus(Int_t p, Int_t c, Int_t cols);
  CalSingleChamberStatus(const CalSingleChamberStatus& c);
  ~CalSingleChamberStatus();
  CalSingleChamberStatus& operator=(const CalSingleChamberStatus& c);
  void Copy(CalSingleChamberStatus& c) const;

  Bool_t isMasked(Int_t col, Int_t row) const { return ((getStatus(col, row) & kMasked)
                                                          ? kTRUE
                                                          : kFALSE); };
  Bool_t isBridgedLeft(Int_t col, Int_t row) const { return ((getStatus(col, row) & kPadBridgedLeft) ? kTRUE : kFALSE); };
  Bool_t isBridgedRight(Int_t col, Int_t row) const { return ((getStatus(col, row) & kPadBridgedRight) ? kTRUE : kFALSE); };
  Bool_t isNotConnected(Int_t col, Int_t row) const { return ((getStatus(col, row) & kNotConnected) ? kTRUE : kFALSE); };
  Int_t getNrows() const { return mNrows; };
  Int_t getNcols() const { return mNcols; };

  Int_t getChannel(Int_t col, Int_t row) const { return row + col * mNrows; };
  Int_t getNchannels() const { return mNchannels; };
  Char_t getStatus(Int_t ich) const { return mData[ich]; };
  Char_t getStatus(Int_t col, Int_t row) const { return mData[getChannel(col, row)]; };

  void setStatus(Int_t ich, Char_t vd) { mData[ich] = vd; };
  void setStatus(Int_t col, Int_t row, Char_t vd) { mData[getChannel(col, row)] = vd; };

 protected:
  Int_t mPla{0}; //  Plane number
  Int_t mCha{0}; //  Chamber number

  Int_t mNrows{0}; //  Number of rows
  Int_t mNcols{0}; //  Number of columns

  Int_t mNchannels{0};     //  Number of channels
  std::vector<char> mData; //[fNchannels] Data

  ClassDefNV(CalSingleChamberStatus, 1); //  TRD ROC calibration class
};

} // namespace trd
} // namespace o2
#endif
