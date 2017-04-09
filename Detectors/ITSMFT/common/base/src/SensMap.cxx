/// \file SensMap.cxx
/// \brief Implementation of hte ITSMFT sensor map

//***********************************************************************
//
// It consist of a TClonesArray of
// SensMapItem objects
// This array can be accessed via 2 indexes
// it is used at digitization level
//
//
// The items should be added to the map like this:
// map->RegisterItem( new(map->GetFree()) ItemConstructor(...) );
//
// The items must be sortable with the same sorting algorithm like
// for SensMap::IsSortable,IsEqual,Compare
//
// ***********************************************************************

#include "ITSMFTBase/SensMap.h"
#include "FairLogger.h"

ClassImp(o2::ITSMFT::SensMap)

  using namespace o2::ITSMFT;

//______________________________________________________________________
SensMap::SensMap() : mDimCol(0), mDimRow(0), mDimCycle(0), mItems(nullptr), mBTree(nullptr)
{
  // Default constructor
}

//______________________________________________________________________
SensMap::SensMap(const char* className, UInt_t dimCol, UInt_t dimRow, UInt_t dimCycle)
  : mDimCol(dimCol),
    mDimRow(dimRow),
    mDimCycle(dimCycle),
    mItems(new TClonesArray(className, 100)),
    mBTree(new TBtree())
{
  // Standard constructor
}

//______________________________________________________________________
SensMap::~SensMap()
{
  // Default destructor
  delete mItems;
  delete mBTree;
}

//______________________________________________________________________
SensMap::SensMap(const SensMap& source)
  : TObject(source),
    mDimCol(source.mDimCol),
    mDimRow(source.mDimRow),
    mDimCycle(source.mDimCycle),
    mItems(source.mItems ? new TClonesArray(*source.mItems) : nullptr),
    mBTree(nullptr)
{
  if (source.mBTree) {
    mBTree = new TBtree();
    if (mItems) {
      for (int i = mItems->GetEntriesFast(); i--;) {
        TObject* obj = mItems->At(i);
        if (obj && !isDisabled(obj))
          continue;
        registerItem(obj);
      }
    }
  }
}

//______________________________________________________________________
SensMap& SensMap::operator=(const SensMap& source)
{
  // = operator
  if (this != &source) {
    this->~SensMap();
    new (this) SensMap(source);
  }
  return *this;
}

//______________________________________________________________________
void SensMap::clear(Option_t*)
{
  // clean everything
  if (mItems)
    mItems->Clear();
  if (mBTree)
    mBTree->Clear();
}

//______________________________________________________________________
void SensMap::deleteItem(UInt_t col, UInt_t row, Int_t cycle)
{
  // Delete a particular SensMapItems.
  SetUniqueID(getIndex(col, row, cycle));
  TObject* fnd = mBTree->FindObject(this);
  if (!fnd)
    return;
  disable(fnd);
  mBTree->Remove(fnd);
}

//______________________________________________________________________
void SensMap::deleteItem(TObject* obj)
{
  // Delete a particular SensMapItems.
  TObject* fnd = mBTree->FindObject(obj);
  if (!fnd)
    return;
  disable(fnd);
  mBTree->Remove(fnd);
}

//______________________________________________________________________
void SensMap::setDimensions(UInt_t dimCol, UInt_t dimRow, UInt_t dimCycle)
{
  // set dimensions for current sensor
  const UInt_t kMaxPackDim = 0xffffffff;
  mDimCol = dimCol;
  mDimRow = dimRow;
  mDimCycle = dimCycle;
  if ((mDimCol * mDimRow * mDimCycle) > kMaxPackDim / 2)
    LOG(FATAL) << "Dimension " << mDimCol << 'x' << mDimRow << 'x' << mDimCycle << "*2 cannot be packed to UInt_t"
               << FairLogger::endl;
}
