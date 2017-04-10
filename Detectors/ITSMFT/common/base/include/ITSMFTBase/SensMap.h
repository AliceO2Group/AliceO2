/// \file SensMap.h
/// \brief Definition of the ITSMFT sensor map

//***********************************************************************
//
// It consist of a TClonesArray of objects
// and B-tree for fast sorted access
// This array can be accessed via 2 indexes
// it is used at digitization level
//
// The items should be added to the map like this:
// map->RegisterItem( new(map->GetFree()) ItemConstructor(...) );
//
// ***********************************************************************

#ifndef ALICEO2_ITSMFT_SENSMAP_H
#define ALICEO2_ITSMFT_SENSMAP_H

#include <TBtree.h>
#include <TClonesArray.h>
#define _ROWWISE_SORT_

namespace o2
{
namespace ITSMFT
{
class SensMap : public TObject
{
 public:
  enum { kDisableBit = BIT(14) };
  //
  SensMap();
  SensMap(const char* className, UInt_t dimCol, UInt_t dimRow, UInt_t dimCycle = 1);
  SensMap(const SensMap& source);
  SensMap& operator=(const SensMap& source);

  ~SensMap() override;

  void clear(Option_t* option = "");
  void deleteItem(UInt_t col, UInt_t row, Int_t cycle);
  void deleteItem(TObject* obj);
  //
  void setDimensions(UInt_t dimCol, UInt_t dimRow, UInt_t dimCycle = 1);
  void getMaxIndex(UInt_t& col, UInt_t& row, UInt_t& cycle) const
  {
    col = mDimCol;
    row = mDimRow;
    cycle = mDimCycle;
  }
  Int_t getMaxIndex() const { return mDimCol * mDimRow * (mDimCycle * 2 + 1); }
  Int_t getEntries() const { return mBTree->GetEntries(); }
  Int_t getEntriesUnsorted() const { return mItems->GetEntriesFast(); }
  void getMapIndex(UInt_t index, UInt_t& col, UInt_t& row, Int_t& cycle) const
  {
    return getCell(index, mDimCol, mDimRow, mDimCycle, col, row, cycle);
  }
  void getCell(UInt_t index, UInt_t& col, UInt_t& row, Int_t& cycle) const
  {
    return getCell(index, mDimCol, mDimRow, mDimCycle, col, row, cycle);
  }
  TObject* getItem(UInt_t col, UInt_t row, Int_t cycle)
  {
    SetUniqueID(getIndex(col, row, cycle));
    return mBTree->FindObject(this);
  }
  TObject* getItem(UInt_t index)
  {
    SetUniqueID(index);
    return mBTree->FindObject(this);
  }
  TObject* getItem(const TObject* obj) { return mBTree->FindObject(obj); }
  TObject* At(Int_t i) const { return mBTree->At(i); }         //!!! Access in sorted order !!!
  TObject* AtUnsorted(Int_t i) const { return mItems->At(i); } //!!! Access in unsorted order !!!
  TObject* registerItem(TObject* obj)
  {
    mBTree->Add(obj);
    return obj;
  }
  TObject* getFree() { return (*mItems)[mItems->GetEntriesFast()]; }
  //
  UInt_t getIndex(UInt_t col, UInt_t row, Int_t cycle = 0) const;
  //
  TClonesArray* getItems() const { return mItems; }
  TBtree* getItemsBTree() const { return mBTree; }
  //
  Bool_t isSortable() const { return kTRUE; }
  Bool_t isEqual(const TObject* obj) const { return GetUniqueID() == obj->GetUniqueID(); }
  Int_t Compare(const TObject* obj) const override
  {
    return (GetUniqueID() < obj->GetUniqueID()) ? -1 : ((GetUniqueID() > obj->GetUniqueID()) ? 1 : 0);
  }
  //
  static Bool_t isDisabled(TObject* obj) { return obj ? obj->TestBit(kDisableBit) : kFALSE; }
  static void disable(TObject* obj)
  {
    if (obj)
      obj->SetBit(kDisableBit);
  }
  static void enable(TObject* obj)
  {
    if (obj)
      obj->ResetBit(kDisableBit);
  }
  static void getCell(UInt_t index, UInt_t dcol, UInt_t drow, UInt_t dcycle, UInt_t& col, UInt_t& row, Int_t& cycle);
  //
 protected:
  //
  UInt_t mDimCol;       ///< 1st dimension of the matrix, col index may span from 0 to mDimCol
  UInt_t mDimRow;       ///< 2nd dimention of the matrix, row index may span from 0 to mDimRow
  UInt_t mDimCycle;     ///< readout cycle range, may span from -mDimCycle to mDimCycle
  TClonesArray* mItems; ///< pListItems array
  TBtree* mBTree;       ///< tree for ordered access
  //
  ClassDefOverride(SensMap, 1) ///< list of sensor signals (should be sortable objects)
};

//______________________________________________________________________
inline UInt_t SensMap::getIndex(UInt_t col, UInt_t row, Int_t cycle) const
{
  // linearized ID of digit
  UInt_t cyclePos = cycle + mDimCycle; // cycle may span from -mDimCycle to mDimCycle
#ifdef _ROWWISE_SORT_
  return mDimCol * (cyclePos * mDimRow + row) + col; // sorted in row, then in column
#else
  return mDimRow * (cyclePos * mDimCol + col) + row; // sorted in column, then in row
#endif
}

//______________________________________________________________________
inline void SensMap::getCell(UInt_t index, UInt_t dcol, UInt_t drow, UInt_t dcycle, UInt_t& col, UInt_t& row,
                             Int_t& cycle)
{
  // returns the i,j index numbers from the linearized index computed with GetIndex
  UInt_t dcr = dcol * drow;
  cycle = int(index / dcr) - dcycle;
  index %= dcr;
#ifdef _ROWWISE_SORT_
  col = index % dcol; // sorted in row, then in column
  row = index / dcol;
#else
  col = index / drow;                                // sorted in column, then in row
  row = index % drow;
#endif
}
}
}
#endif
