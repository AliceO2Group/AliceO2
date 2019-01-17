
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
#include <FairLogger.h>

#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDPadPlane.h"
#include "TRDBase/TRDSignalIndex.h"
#include <algorithm>

using namespace o2::trd;

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex()
  : mDet(-1), mLayer(-1), mStack(-1), mSM(-1), mBoolIndex(NULL), mSortedIndex(NULL), mMaxLimit(0), mPositionRC(0), mCountRC(1), mSortedWasInit(kmALSE), mCurrRow(0), mCurrCol(0), mCurrTbin(0), mNrows(0), mNcols(0), mNtbins(0)
{
  //
  // Default contructor
  //

  ResetCounters();
}

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex(Int_t nrow, Int_t ncol, Int_t ntime)
  : mDet(-1), mLayer(-1), mStack(-1), mSM(-1), mBoolIndex(NULL), mSortedIndex(NULL), mMaxLimit(0), mPositionRC(0), mCountRC(1), mSortedWasInit(kmALSE), mCurrRow(0), mCurrCol(0), mCurrTbin(0), mNrows(0), mNcols(0), mNtbins(0)
{
  //
  // Not the default contructor... hmmm...
  //

  Allocate(nrow, ncol, ntime);
}

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex(const TRDSignalIndex& a)
  : mDet(a.mDet), mLayer(a.mLayer), mStack(a.mStack), mSM(a.mSM), mBoolIndex(NULL), mSortedIndex(NULL), mMaxLimit(a.mMaxLimit), mPositionRC(a.mPositionRC), mCountRC(a.mCountRC), mSortedWasInit(a.mSortedWasInit), mCurrRow(a.mCurrRow), mCurrCol(a.mCurrCol), mCurrTbin(a.mCurrTbin), mNrows(a.mNrows), mNcols(a.mNcols), mNtbins(a.mNtbins)
{
  //
  // Copy constructor
  //

  mBoolIndex = new Bool_t[mMaxLimit];
  memcpy(mBoolIndex, a.mBoolIndex, mMaxLimit * sizeof(Bool_t));

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
    mBoolIndex = NULL;
  }

  if (mSortedIndex) {
    delete[] mSortedIndex;
    mSortedIndex = NULL;
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
  a.mBoolIndex = new Bool_t[mMaxLimit];
  memcpy(a.mBoolIndex, mBoolIndex, mMaxLimit * sizeof(Bool_t));

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
  mBoolIndex = new Bool_t[mMaxLimit];
  memcpy(mBoolIndex, mBoolIndex, mMaxLimit * sizeof(Bool_t));

  if (mSortedIndex) {
    delete[] mSortedIndex;
  }
  mSortedIndex = new RowCol[mMaxLimit + 1];
  memcpy(mSortedIndex, mSortedIndex, (mMaxLimit + 1) * sizeof(RowCol));

  ResetCounters();

  return *this;
}

//_____________________________________________________________________________
void TRDSignalIndex::Allocate(const Int_t nrow, const Int_t ncol, const Int_t ntime)
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
    mBoolIndex = NULL;
  }
  if (mSortedIndex) {
    delete[] mSortedIndex;
    mSortedIndex = NULL;
  }

  mBoolIndex = new Bool_t[mMaxLimit];
  mSortedIndex = new RowCol[mMaxLimit + 1];

  mCountRC = mMaxLimit + 1;

  ResetArrays();
  ResetCounters();

  mCountRC = 1;
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetArrays()
{
  if (!IsAllocated())
    return;
  memset(mBoolIndex, 0x00, sizeof(Bool_t) * mMaxLimit);
  memset(mSortedIndex, 0xFF, sizeof(RowCol) * mCountRC);
  mSortedWasInit = kFALSE;
}

//_____________________________________________________________________________
void TRDSignalIndex::Reset()
{
  //
  // Reset the array but keep the size - realloc
  //

  mDet = -1;
  mLayer = -1;
  mStack = -1;
  mSM = -1;

  // All will be lost
  Allocate(mNrows, mNcols, mNtbins);
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetContent()
{
  //
  // Reset the array but keep the size - no realloc
  //

  mDet = -1;
  mLayer = -1;
  mStack = -1;
  mSM = -1;

  ResetArrays();
  ResetCounters();

  mCountRC = 1;
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetContentConditional(const Int_t nrow, const Int_t ncol, const Int_t ntime)
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
    Allocate(nrow, ncol, ntime);
  } else {
    ResetArrays();
    ResetCounters();
    mCountRC = 1;
  }
}

//_____________________________________________________________________________
void TRDSignalIndex::ClearAll()
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
    mBoolIndex = NULL;
  }

  if (mSortedIndex) {
    delete[] mSortedIndex;
    mSortedIndex = NULL;
  }

  ResetCounters();

  mCountRC = 1;
  mSortedWasInit = kFALSE;
  mMaxLimit = 0;
}

//_____________________________________________________________________________
Bool_t TRDSignalIndex::CheckSorting(Int_t& row, Int_t& col)
{
  //
  // Check whether array was read to end or it was not sorted until now
  //

  if (mSortedWasInit || mCountRC == 1) { //we already reached the end of the array
    ResetCounters();
    row = mCurrRow;
    col = mCurrCol;
    return kFALSE;
  } else { //we have not sorted the array up to now, let's do so
    InitSortedIndex();
    return NextRCIndex(row, col);
  }
}

//_____________________________________________________________________________
Bool_t TRDSignalIndex::NextRCTbinIndex(Int_t& row, Int_t& col, Int_t& tbin)
{
  //
  // Returns the next tbin, or if there is no next time bin, it returns the
  // next used RC combination.
  //

  if (NextTbinIndex(tbin)) {
    row = mCurrRow;
    col = mCurrCol;
    return kTRUE;
  } else {
    if (NextRCIndex(row, col)) {
      return NextRCTbinIndex(row, col, tbin);
    }
  }

  return kFALSE;
}

//_____________________________________________________________________________
Bool_t TRDSignalIndex::NextTbinIndex(Int_t& tbin)
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
void TRDSignalIndex::InitSortedIndex()
{
  //
  // Creates the SortedIndex
  //

  mSortedWasInit = kTRUE;
  std::sort((UShort_t*)mSortedIndex, ((UShort_t*)mSortedIndex) + mCountRC);
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetCounters()
{
  //
  // Reset the counters/iterators
  //

  mCurrRow = -1;
  mCurrCol = -1;
  mCurrTbin = -1;
  mPositionRC = 0;
}
