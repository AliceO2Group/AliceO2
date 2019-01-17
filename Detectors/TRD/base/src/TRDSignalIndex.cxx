
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
  : fDet(-1), fLayer(-1), fStack(-1), fSM(-1), fBoolIndex(NULL), fSortedIndex(NULL), fMaxLimit(0), fPositionRC(0), fCountRC(1), fSortedWasInit(kFALSE), fCurrRow(0), fCurrCol(0), fCurrTbin(0), fNrows(0), fNcols(0), fNtbins(0)
{
  //
  // Default contructor
  //

  ResetCounters();
}

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex(Int_t nrow, Int_t ncol, Int_t ntime)
  : fDet(-1), fLayer(-1), fStack(-1), fSM(-1), fBoolIndex(NULL), fSortedIndex(NULL), fMaxLimit(0), fPositionRC(0), fCountRC(1), fSortedWasInit(kFALSE), fCurrRow(0), fCurrCol(0), fCurrTbin(0), fNrows(0), fNcols(0), fNtbins(0)
{
  //
  // Not the default contructor... hmmm...
  //

  Allocate(nrow, ncol, ntime);
}

//_____________________________________________________________________________
TRDSignalIndex::TRDSignalIndex(const TRDSignalIndex& a)
  : fDet(a.fDet), fLayer(a.fLayer), fStack(a.fStack), fSM(a.fSM), fBoolIndex(NULL), fSortedIndex(NULL), fMaxLimit(a.fMaxLimit), fPositionRC(a.fPositionRC), fCountRC(a.fCountRC), fSortedWasInit(a.fSortedWasInit), fCurrRow(a.fCurrRow), fCurrCol(a.fCurrCol), fCurrTbin(a.fCurrTbin), fNrows(a.fNrows), fNcols(a.fNcols), fNtbins(a.fNtbins)
{
  //
  // Copy constructor
  //

  fBoolIndex = new Bool_t[fMaxLimit];
  memcpy(fBoolIndex, a.fBoolIndex, fMaxLimit * sizeof(Bool_t));

  fSortedIndex = new RowCol[fMaxLimit + 1];
  memcpy(fSortedIndex, a.fSortedIndex, (fMaxLimit + 1) * sizeof(RowCol));
}

//_____________________________________________________________________________
TRDSignalIndex::~TRDSignalIndex()
{
  //
  // Destructor
  //

  if (fBoolIndex) {
    delete[] fBoolIndex;
    fBoolIndex = NULL;
  }

  if (fSortedIndex) {
    delete[] fSortedIndex;
    fSortedIndex = NULL;
  }
}

//_____________________________________________________________________________
void TRDSignalIndex::Copy(TRDSignalIndex& a) const
{
  //
  // Copy function
  //

  a.fDet = fDet;
  a.fLayer = fLayer;
  a.fStack = fStack;
  a.fSM = fSM;
  a.fMaxLimit = fMaxLimit;
  a.fPositionRC = fPositionRC;
  a.fCountRC = fCountRC;
  a.fSortedWasInit = fSortedWasInit;
  a.fCurrRow = fCurrRow;
  a.fCurrCol = fCurrCol;
  a.fCurrTbin = fCurrTbin;
  a.fNrows = fNrows;
  a.fNcols = fNcols;
  a.fNtbins = fNtbins;

  if (a.fBoolIndex) {
    delete[] a.fBoolIndex;
  }
  a.fBoolIndex = new Bool_t[fMaxLimit];
  memcpy(a.fBoolIndex, fBoolIndex, fMaxLimit * sizeof(Bool_t));

  if (a.fSortedIndex) {
    delete[] a.fSortedIndex;
  }
  a.fSortedIndex = new RowCol[fMaxLimit + 1];
  memcpy(a.fSortedIndex, fSortedIndex, (fMaxLimit + 1) * sizeof(RowCol));
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

  fDet = a.fDet;
  fLayer = a.fLayer;
  fStack = a.fStack;
  fSM = a.fSM;
  fMaxLimit = a.fMaxLimit;
  fPositionRC = a.fPositionRC;
  fCountRC = a.fCountRC;
  fSortedWasInit = a.fSortedWasInit;
  fCurrRow = a.fCurrRow;
  fCurrCol = a.fCurrCol;
  fCurrTbin = a.fCurrTbin;
  fNrows = a.fNrows;
  fNcols = a.fNcols;
  fNtbins = a.fNtbins;

  if (fBoolIndex) {
    delete[] fBoolIndex;
  }
  fBoolIndex = new Bool_t[fMaxLimit];
  memcpy(fBoolIndex, fBoolIndex, fMaxLimit * sizeof(Bool_t));

  if (fSortedIndex) {
    delete[] fSortedIndex;
  }
  fSortedIndex = new RowCol[fMaxLimit + 1];
  memcpy(fSortedIndex, fSortedIndex, (fMaxLimit + 1) * sizeof(RowCol));

  ResetCounters();

  return *this;
}

//_____________________________________________________________________________
void TRDSignalIndex::Allocate(const Int_t nrow, const Int_t ncol, const Int_t ntime)
{
  //
  // Create the arrays
  //

  fNrows = nrow;
  fNcols = ncol;
  fNtbins = ntime;

  fMaxLimit = nrow * ncol + 1;

  if (fBoolIndex) {
    delete[] fBoolIndex;
    fBoolIndex = NULL;
  }
  if (fSortedIndex) {
    delete[] fSortedIndex;
    fSortedIndex = NULL;
  }

  fBoolIndex = new Bool_t[fMaxLimit];
  fSortedIndex = new RowCol[fMaxLimit + 1];

  fCountRC = fMaxLimit + 1;

  ResetArrays();
  ResetCounters();

  fCountRC = 1;
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetArrays()
{
  if (!IsAllocated())
    return;
  memset(fBoolIndex, 0x00, sizeof(Bool_t) * fMaxLimit);
  memset(fSortedIndex, 0xFF, sizeof(RowCol) * fCountRC);
  fSortedWasInit = kFALSE;
}

//_____________________________________________________________________________
void TRDSignalIndex::Reset()
{
  //
  // Reset the array but keep the size - realloc
  //

  fDet = -1;
  fLayer = -1;
  fStack = -1;
  fSM = -1;

  // All will be lost
  Allocate(fNrows, fNcols, fNtbins);
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetContent()
{
  //
  // Reset the array but keep the size - no realloc
  //

  fDet = -1;
  fLayer = -1;
  fStack = -1;
  fSM = -1;

  ResetArrays();
  ResetCounters();

  fCountRC = 1;
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetContentConditional(const Int_t nrow, const Int_t ncol, const Int_t ntime)
{
  //
  // Reset the array but keep the size if no need to enlarge - no realloc
  //

  fDet = -1;
  fLayer = -1;
  fStack = -1;
  fSM = -1;

  if ((nrow > fNrows) ||
      (ncol > fNcols) ||
      (ntime > fNtbins)) {
    Allocate(nrow, ncol, ntime);
  } else {
    ResetArrays();
    ResetCounters();
    fCountRC = 1;
  }
}

//_____________________________________________________________________________
void TRDSignalIndex::ClearAll()
{
  //
  // Reset the values - clear all!
  //

  fDet = -1;
  fLayer = -1;
  fStack = -1;
  fSM = -1;

  fNrows = -1;
  fNcols = -1;
  fNtbins = -1;

  if (fBoolIndex) {
    delete[] fBoolIndex;
    fBoolIndex = NULL;
  }

  if (fSortedIndex) {
    delete[] fSortedIndex;
    fSortedIndex = NULL;
  }

  ResetCounters();

  fCountRC = 1;
  fSortedWasInit = kFALSE;
  fMaxLimit = 0;
}

//_____________________________________________________________________________
Bool_t TRDSignalIndex::CheckSorting(Int_t& row, Int_t& col)
{
  //
  // Check whether array was read to end or it was not sorted until now
  //

  if (fSortedWasInit || fCountRC == 1) { //we already reached the end of the array
    ResetCounters();
    row = fCurrRow;
    col = fCurrCol;
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
    row = fCurrRow;
    col = fCurrCol;
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

  if (fCurrTbin < fNtbins) {
    tbin = fCurrTbin++;
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

  fSortedWasInit = kTRUE;
  std::sort((UShort_t*)fSortedIndex, ((UShort_t*)fSortedIndex) + fCountRC);
}

//_____________________________________________________________________________
void TRDSignalIndex::ResetCounters()
{
  //
  // Reset the counters/iterators
  //

  fCurrRow = -1;
  fCurrCol = -1;
  fCurrTbin = -1;
  fPositionRC = 0;
}
