
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDSIGNALINDEX_H
#define O2_TRDSIGNALINDEX_H

//Forwards to standard header with protection for GPU compilation
#include "AliTPCCommonRtypes.h" // for ClassDef

namespace o2
{
namespace trd
{

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  General container for data from TRD detector segments                 //
//  Adapted from AliDigits, origin M.Ivanov                               //
//                                                                        //
//  Author:                                                               //
//    Mateusz Ploskon (ploskon@ikf.uni-frankfurt.de)                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TRDSignalIndex
{
 protected:
  union RowCol {
    Short_t rc;
    struct {
      UChar_t col;
      Char_t row;
    } s;
  };

 public:
  TRDSignalIndex();
  TRDSignalIndex(Int_t nrow, Int_t ncol, Int_t ntime);
  TRDSignalIndex(const TRDSignalIndex& d);
  virtual ~TRDSignalIndex();
  TRDSignalIndex& operator=(const TRDSignalIndex& d);

  void Copy(TRDSignalIndex& d) const;
  void Allocate(const Int_t nrow, const Int_t ncol, const Int_t ntime);

  void Reset();
  void ResetContentConditional(const Int_t nrow, const Int_t ncol, const Int_t ntime);
  void ResetContent();
  void ResetCounters();
  void ResetTbinCounter() const {};

  void ResetArrays();

  // Store the index row-column as an interesting one
  inline void AddIndexRC(const Int_t row, const Int_t col);
  // get the next pad (row and column) and return kTRUE on success
  inline Bool_t NextRCIndex(Int_t& row, Int_t& col);
  // get the next timebin of a pad (row and column) and return kTRUE on success
  Bool_t NextRCTbinIndex(Int_t& row, Int_t& col, Int_t& tbin);
  // get the next active timebin and return kTRUE on success
  Bool_t NextTbinIndex(Int_t& tbin);

  Bool_t CheckSorting(Int_t& row, Int_t& col);

  Int_t getCurrentRow() const { return mCurrRow; }
  Int_t getCurrentCol() const { return mCurrCol; }
  Int_t getCurrentTbin() const { return mCurrTbin; }

  Bool_t IsBoolIndex(Int_t row, Int_t col) const { return mBoolIndex[row * mNcols + col]; };
  void InitSortedIndex();

  // Clear the array, actually destroy and recreate w/o allocating
  void ClearAll();
  // Return kTRUE if array allocated and there is no need to call allocate
  Bool_t IsAllocated() const
  {
    if (!mBoolIndex)
      return kFALSE;
    if (mMaxLimit <= 0)
      return kFALSE;
    else
      return kTRUE;
  }

  void setSM(const Int_t ix) { mSM = ix; }
  void setStack(const Int_t ix) { mStack = ix; }
  void setLayer(const Int_t ix) { mLayer = ix; }
  void setDetNumber(const Int_t ix) { mDet = ix; }

  Int_t getDetNumber() const { return mDet; }                  // get Det number
  Int_t getLayer() const { return mLayer; }                    // Layer position of the chamber in TRD
  Int_t getStack() const { return mStack; }                    // Stack position of the chamber in TRD
  Int_t getSM() const { return mSM; }                          // Super module of the TRD
  Short_t* getArray() const { return (Short_t*)mSortedIndex; } // get the array pointer for god knows what reason
  Int_t getNoOfIndexes() const { return mCountRC - 1; }

  Bool_t HasEntry() const { return mCountRC > 1 ? kTRUE : kFALSE; } // Return status if has an entry

  Int_t getNrow() const { return mNrows; }   // get Nrows
  Int_t getNcol() const { return mNcols; }   // get Ncols
  Int_t getNtime() const { return mNtbins; } // get Ntbins

 private:
  Int_t mDet;   //  Detector number
  Int_t mLayer; //  Layer position in the full TRD
  Int_t mStack; //  Stack position in the full TRD
  Int_t mSM;    //  Super module - position in the full TRD

  Bool_t* mBoolIndex;    //  Indices
  RowCol* mSortedIndex;  //  Sorted indices
  Int_t mMaxLimit;       //  Max number of things in the array
  Int_t mPositionRC;     //  Position in the SortedIndex
  Int_t mCountRC;        //  the number of added rc combinations
  Bool_t mSortedWasInit; //  Was SortedIndex initialized?

  Int_t mCurrRow;  //  Last Row read out of SortedIndex
  Int_t mCurrCol;  //  Last Col read out of SortedIndex
  Int_t mCurrTbin; //  Last outgiven Tbin

  Int_t mNrows;  //  Number of rows in the chamber
  Int_t mNcols;  //  Number of cols in the chamber
  Int_t mNtbins; //  Number of tbins in the chamber

  ClassDefNV(TRDSignalIndex, 1) //  Data container for one TRD detector segment
};

void TRDSignalIndex::AddIndexRC(const Int_t row, const Int_t col)
{
  //
  // Adds RC combination to array
  //

  const Int_t num = row * mNcols + col;
  if (mBoolIndex[num])
    return;
  mBoolIndex[num] = kTRUE;
  mSortedIndex[mCountRC].s.col = col;
  mSortedIndex[mCountRC].s.row = row;
  mCountRC++;
}

Bool_t TRDSignalIndex::NextRCIndex(Int_t& row, Int_t& col)
{
  //
  // Returns next used RC combination
  //

  if (!IsAllocated())
    return kFALSE;

  if (mSortedIndex[mPositionRC].rc > -1) {
    row = mCurrRow = mSortedIndex[mPositionRC].s.row;
    col = mCurrCol = mSortedIndex[mPositionRC].s.col;
    mPositionRC++;
    return kTRUE;
  } else
    return CheckSorting(row, col);
}

} //namespace trd
} //namespace o2

#endif
