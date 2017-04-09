//
//  DigitWriteoutBuffer.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 21.07.15.
//
//
#include "ITSSimulation/DigitWriteoutBuffer.h"
#include "FairRootManager.h"  // for FairRootManager
#include "TClonesArray.h"     // for TClonesArray
#include "TString.h"          // for TString

class FairTimeStamp;

ClassImp(o2::ITS::DigitWriteoutBuffer)

using o2::ITSMFT::Digit;
using namespace o2::ITS;

DigitWriteoutBuffer::DigitWriteoutBuffer() :
  FairWriteoutBuffer(),
  mData_map()
{

}

DigitWriteoutBuffer::DigitWriteoutBuffer(TString branchname, TString foldername, Bool_t persistance) :
  FairWriteoutBuffer(branchname, "o2::ITSMFT::Digit", foldername, persistance),
  mData_map()
{

}

DigitWriteoutBuffer::~DigitWriteoutBuffer()
= default;

void DigitWriteoutBuffer::AddNewDataToTClonesArray(FairTimeStamp *timestamp)
{
  FairRootManager *iohandler = FairRootManager::Instance();
  TClonesArray *outputarray = iohandler->GetTClonesArray(fBranchName);

  new((*outputarray)[outputarray->GetEntries()])Digit(*(static_cast<Digit *>(timestamp)));
}

double DigitWriteoutBuffer::FindTimeForData(FairTimeStamp *timestamp)
{
  Digit itsdigit = *(static_cast<Digit *>(timestamp));
  auto result = mData_map.find(itsdigit);
  if (result != mData_map.end()) {
    return result->second;
  }
  return -1;
}

void DigitWriteoutBuffer::FillDataMap(FairTimeStamp *data, double activeTime)
{
  Digit itsdigit = *(static_cast<Digit *>(data));
  mData_map[itsdigit] = activeTime;
}

void DigitWriteoutBuffer::EraseDataFromDataMap(FairTimeStamp *data)
{
  Digit itsdigit = *(static_cast<Digit *>(data));
  if (mData_map.find(itsdigit) != mData_map.end()) {
    mData_map.erase(itsdigit);
  }
}
