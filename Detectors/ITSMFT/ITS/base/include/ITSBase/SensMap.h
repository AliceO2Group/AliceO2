/// \file SensMap.h
/// \brief Sensor map structure for upgrade ITS

//***********************************************************************
//
// It consist of a TClonesArray of objects
// and B-tree for fast sorted access
// This array can be accessed via 2 indexed
// it is used at digitization level by 
// all the ITS subdetectors
//
// The items should be added to the map like this:
// map->RegisterItem( new(map->GetFree()) ItemConstructor(...) );
//
// ***********************************************************************


#ifndef ALICEO2_ITS_SENSMAP_H
#define ALICEO2_ITS_SENSMAP_H

#include <TClonesArray.h>
#include <TBtree.h>
#define _ROWWISE_SORT_

namespace AliceO2 {
namespace ITS {

class SensMap: public TObject 
{

 public:
  enum {kDisableBit=BIT(14)};
  //
  SensMap();
  SensMap(const char* className, UInt_t dimCol,UInt_t dimRow,UInt_t dimCycle=1);
  virtual ~SensMap();
  SensMap(const SensMap &source);
  SensMap& operator=(const SensMap &source);
  void Clear(Option_t* option = "");
  void DeleteItem(UInt_t col,UInt_t row, Int_t cycle);
  void DeleteItem(TObject* obj);
  //
  void  SetDimensions(UInt_t dimCol,UInt_t dimRow,UInt_t dimCycle=1);
  void  GetMaxIndex(UInt_t &col,UInt_t &row,UInt_t &cycle) const {col=fDimCol; row=fDimRow; cycle=fDimCycle;}
  Int_t GetMaxIndex()                      const {return fDimCol*fDimRow*(fDimCycle*2+1);}
  Int_t GetEntries()                       const {return fBTree->GetEntries();}
  Int_t GetEntriesUnsorted()               const {return fItems->GetEntriesFast();}
  void  GetMapIndex(UInt_t index,UInt_t &col,UInt_t &row,Int_t &cycle) const {return GetCell(index,fDimCol,fDimRow,fDimCycle,col,row,cycle);}
  void  GetCell(UInt_t index,UInt_t &col,UInt_t &row,Int_t &cycle)     const {return GetCell(index,fDimCol,fDimRow,fDimCycle,col,row,cycle);}
  TObject* GetItem(UInt_t col,UInt_t row,Int_t cycle)  {SetUniqueID(GetIndex(col,row,cycle)); return fBTree->FindObject(this);}
  TObject* GetItem(UInt_t index)                 {SetUniqueID(index);         return fBTree->FindObject(this);}
  TObject* GetItem(const TObject* obj)           {return fBTree->FindObject(obj);}
  TObject* At(Int_t i)                     const {return fBTree->At(i);}             //!!! Access in sorted order !!!
  TObject* AtUnsorted(Int_t i)             const {return fItems->At(i);}             //!!! Access in unsorted order !!!
  TObject* RegisterItem(TObject* obj)            {fBTree->Add(obj); return obj;}
  TObject* GetFree()                             {return (*fItems)[fItems->GetEntriesFast()];}
  //
  UInt_t   GetIndex(UInt_t col,UInt_t row,Int_t cycle=0)  const;
  //
  TClonesArray* GetItems()                 const {return fItems;}
  TBtree*       GetItemsBTree()            const {return fBTree;}
  //
  Bool_t        IsSortable()                const {return kTRUE;}
  Bool_t        IsEqual(const TObject* obj) const {return GetUniqueID()==obj->GetUniqueID();}
  Int_t         Compare(const TObject* obj) const {return (GetUniqueID()<obj->GetUniqueID()) ? -1 : ((GetUniqueID()>obj->GetUniqueID()) ? 1 : 0 );}
  //
  static Bool_t IsDisabled(TObject* obj)         {return obj ? obj->TestBit(kDisableBit) : kFALSE;}
  static void   Disable(TObject* obj)            {if (obj) obj->SetBit(kDisableBit);}
  static void   Enable(TObject* obj)             {if (obj) obj->ResetBit(kDisableBit);}
  static void   GetCell(UInt_t index,UInt_t dcol,UInt_t drow,UInt_t dcycle,UInt_t &col,UInt_t &row,Int_t &cycle);
  //
 protected:
  //
  UInt_t fDimCol;              // 1st dimension of the matrix, col index may span from 0 to fDimCol
  UInt_t fDimRow;              // 2nd dimention of the matrix, row index may span from 0 to fDimRow
  UInt_t fDimCycle;            // readout cycle range, may span from -fDimCycle to fDimCycle
  TClonesArray*    fItems;   // pListItems array
  TBtree*          fBTree;   // tree for ordered access
  //
  ClassDef(SensMap,1) // list of sensor signals (should be sortable objects)
};	

//______________________________________________________________________
inline UInt_t SensMap::GetIndex(UInt_t col,UInt_t row, Int_t cycle) const  
{
  // linearized ID of digit
  UInt_t cyclePos = cycle+fDimCycle; // cycle may span from -fDimCycle to fDimCycle
#ifdef _ROWWISE_SORT_
  return fDimCol*(cyclePos*fDimRow+row)+col; // sorted in row, then in column
#else
  return fDimRow*(cyclePos*fDimCol+col)+row; // sorted in column, then in row
#endif
}

//______________________________________________________________________
inline void SensMap::GetCell(UInt_t index,UInt_t dcol,UInt_t drow,UInt_t dcycle,UInt_t &col,UInt_t &row,Int_t &cycle) 
{
  // returns the i,j index numbers from the linearized index computed with GetIndex
  UInt_t dcr = dcol*drow;
  cycle = int(index/dcr) - dcycle;
  index %= dcr;
#ifdef _ROWWISE_SORT_
  col = index%dcol;   // sorted in row, then in column
  row = index/dcol;
#else
  col = index/drow;   // sorted in column, then in row
  row = index%drow;
#endif  
}
}
}
#endif
