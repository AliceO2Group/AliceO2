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
//  General container for data from TRD detector segments                    //
//  Adapted from AliDigits, origin M.Ivanov                                  //
//                                                                           //
//  Author:                                                                  //
//    Mateusz Ploskon (ploskon@ikf.uni-frankfurt.de)                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TMath.h>
#include <TVirtualMC.h>

#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"
#include "TRDBase/TRDSignalIndex.h"
#include <algorithm>

using namespace o2::trd;

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex()
{
  //
  // Default contructor
  //

  resetCounters();
}

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex(Int_t nrow, Int_t ncol, Int_t ntime)
{
  //
  // Not the default contructor... hmmm...
  //

  allocate(nrow, ncol, ntime);
}

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex(const TRDSignalIndex& a)
  : mDet(a.mDet), mLayer(a.mLayer), mStack(a.mStack), mSM(a.mSM), mBoolIndex(nullptr), mSortedIndex(nullptr), mMaxLimit(a.mMaxLimit), mPositionRC(a.mPositionRC), mCountRC(a.mCountRC), mSortedWasInit(a.mSortedWasInit), mCurrRow(a.mCurrRow), mCurrCol(a.mCurrCol), mCurrTbin(a.mCurrTbin), mNrows(a.mNrows), mNcols(a.mNcols), mNtbins(a.mNtbins)
{
  //
  // Copy constructor
  //

  mBoolIndex = new bool[mMaxLimit];
  memcpy(mBoolIndex, a.mBoolIndex, mMaxLimit * sizeof(bool));

  mSortedIndex = new RowCol[mMaxLimit + 1];
  memcpy(mSortedIndex, a.mSortedIndex, (mMaxLimit + 1) * sizeof(RowCol));
}

//_____________________________________________________________________________
TRDSignalIndex::~TRDSignalIndex()
{
  //
  // Destructor
  //

  if (mBoolIndex) {
    delete[] mBoolIndex;
    mBoolIndex = nullptr;
  }

  if (mSortedIndex) {
    delete[] mSortedIndex;
    mSortedIndex = nullptr;
  }
}

//_____________________________________________________________________________
void TRDSignalIndex::Copy(TRDSignalIndex& a) const
{
  //
  // Copy function
  //

  a.mDet = mDet;
  a.mLayer = mLayer;
  a.mStack = mStack;
  a.mSM = mSM;
  a.mMaxLimit = mMaxLimit;
  a.mPositionRC = mPositionRC;
  a.mCountRC = mCountRC;
  a.mSortedWasInit = mSortedWasInit;
  a.mCurrRow = mCurrRow;
  a.mCurrCol = mCurrCol;
  a.mCurrTbin = mCurrTbin;
  a.mNrows = mNrows;
  a.mNcols = mNcols;
  a.mNtbins = mNtbins;

  if (a.mBoolIndex) {
    delete[] a.mBoolIndex;
  }
  a.mBoolIndex = new bool[mMaxLimit];
  memcpy(a.mBoolIndex, mBoolIndex, mMaxLimit * sizeof(bool));

  if (a.mSortedIndex) {
    delete[] a.mSortedIndex;
  }
  a.mSortedIndex = new RowCol[mMaxLimit + 1];
  memcpy(a.mSortedIndex, mSortedIndex, (mMaxLimit + 1) * sizeof(RowCol));
}

//_____________________________________________________________________________
TRDSignalIndex& TRDSignalIndex::operator=(const TRDSignalIndex& a)
{
  //
  // Assignment operator
  //

  if (this == &a) {
    return *this;
  }

  mDet = a.mDet;
  mLayer = a.mLayer;
  mStack = a.mStack;
  mSM = a.mSM;
  mMaxLimit = a.mMaxLimit;
  mPositionRC = a.mPositionRC;
  mCountRC = a.mCountRC;
  mSortedWasInit = a.mSortedWasInit;
  mCurrRow = a.mCurrRow;
  mCurrCol = a.mCurrCol;
  mCurrTbin = a.mCurrTbin;
  mNrows = a.mNrows;
  mNcols = a.mNcols;
  mNtbins = a.mNtbins;

  if (mBoolIndex) {
    delete[] mBoolIndex;
  }
  mBoolIndex = new bool[mMaxLimit];
  memcpy(mBoolIndex, mBoolIndex, mMaxLimit * sizeof(bool));

  if (mSortedIndex) {
    delete[] mSortedIndex;
  }
  mSortedIndex = new RowCol[mMaxLimit + 1];
  memcpy(mSortedIndex, mSortedIndex, (mMaxLimit + 1) * sizeof(RowCol));

  resetCounters();

  return *this;
}

//_____________________________________________________________________________
void TRDSignalIndex::allocate(const Int_t nrow, const Int_t ncol, const Int_t ntime)
{
  //
  // Create the arrays
  //

  mNrows = nrow;
  mNcols = ncol;
  mNtbins = ntime;

  mMaxLimit = nrow * ncol + 1;

  if (mBoolIndex) {
    delete[] mBoolIndex;
    mBoolIndex = nullptr;
  }
  if (mSortedIndex) {
    delete[] mSortedIndex;
    mSortedIndex = nullptr;
  }

  mBoolIndex = new bool[mMaxLimit];
  mSortedIndex = new RowCol[mMaxLimit + 1];

  mCountRC = mMaxLimit + 1;

  resetArrays();
  resetCounters();

  mCountRC = 1;
}

//_____________________________________________________________________________
void TRDSignalIndex::resetArrays()
{
  if (!isAllocated())
    return;
  memset(mBoolIndex, 0x00, sizeof(bool) * mMaxLimit);
  memset(mSortedIndex, 0xFF, sizeof(RowCol) * mCountRC);
  mSortedWasInit = kFALSE;
}

//_____________________________________________________________________________
void TRDSignalIndex::reset()
{
  //
  // Reset the array but keep the size - realloc
  //

  mDet = -1;
  mLayer = -1;
  mStack = -1;
  mSM = -1;

  // All will be lost
  allocate(mNrows, mNcols, mNtbins);
}

//_____________________________________________________________________________
void TRDSignalIndex::resetContent()
{
  //
  // Reset the array but keep the size - no realloc
  //

  mDet = -1;
  mLayer = -1;
  mStack = -1;
  mSM = -1;

  resetArrays();
  resetCounters();

  mCountRC = 1;
}

//_____________________________________________________________________________
void TRDSignalIndex::resetContentConditional(const Int_t nrow, const Int_t ncol, const Int_t ntime)
{
  //
  // Reset the array but keep the size if no need to enlarge - no realloc
  //

  mDet = -1;
  mLayer = -1;
  mStack = -1;
  mSM = -1;

  if ((nrow > mNrows) ||
      (ncol > mNcols) ||
      (ntime > mNtbins)) {
    allocate(nrow, ncol, ntime);
  } else {
    resetArrays();
    resetCounters();
    mCountRC = 1;
  }
}

//_____________________________________________________________________________
void TRDSignalIndex::clearAll()
{
  //
  // Reset the values - clear all!
  //

  mDet = -1;
  mLayer = -1;
  mStack = -1;
  mSM = -1;

  mNrows = -1;
  mNcols = -1;
  mNtbins = -1;

  if (mBoolIndex) {
    delete[] mBoolIndex;
    mBoolIndex = nullptr;
  }

  if (mSortedIndex) {
    delete[] mSortedIndex;
    mSortedIndex = nullptr;
  }

  resetCounters();

  mCountRC = 1;
  mSortedWasInit = kFALSE;
  mMaxLimit = 0;
}

//_____________________________________________________________________________
bool TRDSignalIndex::checkSorting(Int_t& row, Int_t& col)
{
  //
  // Check whether array was read to end or it was not sorted until now
  //

  if (mSortedWasInit || mCountRC == 1) { //we already reached the end of the array
    resetCounters();
    row = mCurrRow;
    col = mCurrCol;
    return kFALSE;
  } else { //we have not sorted the array up to now, let's do so
    initSortedIndex();
    return nextRCIndex(row, col);
  }
}

//_____________________________________________________________________________
bool TRDSignalIndex::nextRCTbinIndex(Int_t& row, Int_t& col, Int_t& tbin)
{
  //
  // Returns the next tbin, or if there is no next time bin, it returns the
  // next used RC combination.
  //

  if (nextTbinIndex(tbin)) {
    row = mCurrRow;
    col = mCurrCol;
    return kTRUE;
  } else {
    if (nextRCIndex(row, col)) {
      return nextRCTbinIndex(row, col, tbin);
    }
  }

  return kFALSE;
}

//_____________________________________________________________________________
bool TRDSignalIndex::nextTbinIndex(Int_t& tbin)
{
  //
  // Returns the next tbin of the current RC combination
  //

  if (mCurrTbin < mNtbins) {
    tbin = mCurrTbin++;
    return kTRUE;
  }

  return kFALSE;
}

//_____________________________________________________________________________
void TRDSignalIndex::initSortedIndex()
{
  //
  // Creates the SortedIndex
  //

  mSortedWasInit = kTRUE;
  std::sort((unsigned short*)mSortedIndex, ((unsigned short*)mSortedIndex) + mCountRC);
}

//_____________________________________________________________________________
void TRDSignalIndex::resetCounters()
{
  //
  // Reset the counters/iterators
  //

  mCurrRow = -1;
  mCurrCol = -1;
  mCurrTbin = -1;
  mPositionRC = 0;
}
