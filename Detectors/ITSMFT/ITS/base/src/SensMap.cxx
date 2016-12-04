/// \file SensMap.cxx
/// \brief Sensor map structure for ITS digits

//***********************************************************************
//
// It consist of a TClonesArray of 
// SensMapItem objects
// This array can be accessed via 2 indexed
// it is used at digitization level by 
// all the 3 ITS subdetectors
//
//
// The items should be added to the map like this:
// map->RegisterItem( new(map->GetFree()) ItemConstructor(...) );
//
// The items must be sortable with the same sorting algorithm like 
// for SensMap::IsSortable,IsEqual,Compare
//
// ***********************************************************************

#include "ITSBase/SensMap.h"
#include "FairLogger.h"

ClassImp(AliceO2::ITS::SensMap)

using namespace AliceO2::ITS;

//______________________________________________________________________
SensMap::SensMap() 
:  fDimCol(0)
  ,fDimRow(0)
  ,fDimCycle(0)
  ,fItems(0)
  ,fBTree(0)
{
  // Default constructor
}

//______________________________________________________________________
SensMap::SensMap(const char* className, UInt_t dimCol,UInt_t dimRow,UInt_t dimCycle)
  :fDimCol(dimCol)
  ,fDimRow(dimRow)
  ,fDimCycle(dimCycle)
  ,fItems(new TClonesArray(className,100))
  ,fBTree(new TBtree())
{
  // Standard constructor
}

//______________________________________________________________________
SensMap::~SensMap() 
{
  // Default destructor
  delete fItems;
  delete fBTree;
}


//______________________________________________________________________
SensMap::SensMap(const SensMap &source)
  :TObject(source)
  ,fDimCol(source.fDimCol)
  ,fDimRow(source.fDimRow)
  ,fDimCycle(source.fDimCycle)
  ,fItems( source.fItems ? new TClonesArray(*source.fItems) : 0)
  ,fBTree( 0 )
{
  if (source.fBTree) {
    fBTree = new TBtree();
    if (fItems) {
      for (int i=fItems->GetEntriesFast();i--;) {
	TObject* obj = fItems->At(i);
	if (obj && ! IsDisabled(obj)) continue;
	RegisterItem(obj);
      }
    }
  }
}

//______________________________________________________________________
SensMap& SensMap::operator=(const SensMap &source)
{
  // = operator
  if (this!=&source) {
    this->~SensMap();
    new(this) SensMap(source);
  }
  return *this;
}

//______________________________________________________________________
void SensMap::Clear(Option_t*) 
{
  // clean everything
  if (fItems) fItems->Clear();
  if (fBTree) fBTree->Clear();
}

//______________________________________________________________________
void SensMap::DeleteItem(UInt_t col,UInt_t row,Int_t cycle)
{
  // Delete a particular SensMapItems.
  SetUniqueID( GetIndex(col,row,cycle) );
  TObject* fnd = fBTree->FindObject(this);
  if (!fnd) return;
  Disable(fnd);
  fBTree->Remove(fnd);
}

//______________________________________________________________________
void SensMap::DeleteItem(TObject* obj)
{
  // Delete a particular SensMapItems.
  TObject* fnd = fBTree->FindObject(obj);
  if (!fnd) return;
  Disable(fnd);
  fBTree->Remove(fnd);
}

//______________________________________________________________________
void SensMap::SetDimensions(UInt_t dimCol,UInt_t dimRow,UInt_t dimCycle) 
{
  // set dimensions for current sensor
  const UInt_t kMaxPackDim = 0xffffffff;
  fDimCol = dimCol; 
  fDimRow = dimRow; 
  fDimCycle=dimCycle;
  if ((fDimCol*fDimRow*fDimCycle)>kMaxPackDim/2)
    LOG(FATAL)<<"Dimension "<<fDimCol<<'x'<<fDimRow<<'x'<<fDimCycle<<"*2 cannot be packed to UInt_t"<<FairLogger::endl;
}

