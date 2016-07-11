#include "ClusterCRU.h"
#include "ClusterRow.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

#include <iostream>

ClusterCRU::ClusterCRU(Int_t cruID, Int_t nrows):
  mCRUID(cruID),
  mNRows(nrows),
  mRows(nrows)
{}

ClusterCRU::~ClusterCRU()
{
  for (Int_t irow = 0; irow < mNRows; irow++) {
    delete mRows[irow];
  }
}

void ClusterCRU::SetCluster(Int_t row, Int_t pad, Int_t time, Float_t charge)
{
  // Check input
  if(row < 0 || row >= mNrows) {
    // error
    return;
  }
  
  // if row container does not exist, create it
  if(mRows[row] == nullptr){
    mRows[row] = new ClusterRow(row);
  }

  mRows[row]->SetCluster(pad, time, charge);
}

void ClusterCRU::Reset()
{
  for(std::vector<ClusterRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); iterRow++) {
    if((*iterRow) == nullptr) continue;
    (*iterRow)->Reset();
  }
}

void ClusterCRU::FillOutputContainer(TClonesArray *output, Int_t cruID)
{
  for(std::vector<ClusterRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); iterRow++) {
    if((*iterRow) == nullptr) continue;
    (*iterRow)->FillOutputContainer(output, cruID, (*iterRow)->GetRow());
  }
}
