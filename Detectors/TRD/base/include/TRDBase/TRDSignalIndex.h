
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

  Int_t getCurrentRow() const { return fCurrRow; }
  Int_t getCurrentCol() const { return fCurrCol; }
  Int_t getCurrentTbin() const { return fCurrTbin; }

  Bool_t IsBoolIndex(Int_t row, Int_t col) const { return fBoolIndex[row * fNcols + col]; };
  void InitSortedIndex();

  // Clear the array, actually destroy and recreate w/o allocating
  void ClearAll();
  // Return kTRUE if array allocated and there is no need to call allocate
  Bool_t IsAllocated() const
  {
    if (!fBoolIndex)
      return kFALSE;
    if (fMaxLimit <= 0)
      return kFALSE;
    else
      return kTRUE;
  }

  void setSM(const Int_t ix) { fSM = ix; }
  void setStack(const Int_t ix) { fStack = ix; }
  void setLayer(const Int_t ix) { fLayer = ix; }
  void setDetNumber(const Int_t ix) { fDet = ix; }

  Int_t getDetNumber() const { return fDet; }                  // get Det number
  Int_t getLayer() const { return fLayer; }                    // Layer position of the chamber in TRD
  Int_t getStack() const { return fStack; }                    // Stack position of the chamber in TRD
  Int_t getSM() const { return fSM; }                          // Super module of the TRD
  Short_t* getArray() const { return (Short_t*)fSortedIndex; } // get the array pointer for god knows what reason
  Int_t getNoOfIndexes() const { return fCountRC - 1; }

  Bool_t HasEntry() const { return fCountRC > 1 ? kTRUE : kFALSE; } // Return status if has an entry

  Int_t getNrow() const { return fNrows; }   // get Nrows
  Int_t getNcol() const { return fNcols; }   // get Ncols
  Int_t getNtime() const { return fNtbins; } // get Ntbins

 private:
  Int_t fDet;   //  Detector number
  Int_t fLayer; //  Layer position in the full TRD
  Int_t fStack; //  Stack position in the full TRD
  Int_t fSM;    //  Super module - position in the full TRD

  Bool_t* fBoolIndex;    //  Indices
  RowCol* fSortedIndex;  //  Sorted indices
  Int_t fMaxLimit;       //  Max number of things in the array
  Int_t fPositionRC;     //  Position in the SortedIndex
  Int_t fCountRC;        //  the number of added rc combinations
  Bool_t fSortedWasInit; //  Was SortedIndex initialized?

  Int_t fCurrRow;  //  Last Row read out of SortedIndex
  Int_t fCurrCol;  //  Last Col read out of SortedIndex
  Int_t fCurrTbin; //  Last outgiven Tbin

  Int_t fNrows;  //  Number of rows in the chamber
  Int_t fNcols;  //  Number of cols in the chamber
  Int_t fNtbins; //  Number of tbins in the chamber

  ClassDefNV(TRDSignalIndex, 1) //  Data container for one TRD detector segment
};

void TRDSignalIndex::AddIndexRC(const Int_t row, const Int_t col)
{
  //
  // Adds RC combination to array
  //

  const Int_t num = row * fNcols + col;
  if (fBoolIndex[num])
    return;
  fBoolIndex[num] = kTRUE;
  fSortedIndex[fCountRC].s.col = col;
  fSortedIndex[fCountRC].s.row = row;
  fCountRC++;
}

Bool_t TRDSignalIndex::NextRCIndex(Int_t& row, Int_t& col)
{
  //
  // Returns next used RC combination
  //

  if (!IsAllocated())
    return kFALSE;

  if (fSortedIndex[fPositionRC].rc > -1) {
    row = fCurrRow = fSortedIndex[fPositionRC].s.row;
    col = fCurrCol = fSortedIndex[fPositionRC].s.col;
    fPositionRC++;
    return kTRUE;
  } else
    return CheckSorting(row, col);
}

} //namespace trd
} //namespace o2

#endif
