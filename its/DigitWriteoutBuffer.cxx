//
//  DigitWriteoutBuffer.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 21.07.15.
//
//
#include "its/DigitWriteoutBuffer.h"
#include <TClonesArray.h>     // for TClonesArray
#include "FairRootManager.h"  // for FairRootManager
#include "TString.h"          // for TString
class FairTimeStamp;


ClassImp(AliceO2::ITS::DigitWriteoutBuffer)

using namespace AliceO2::ITS;

DigitWriteoutBuffer::DigitWriteoutBuffer():
FairWriteoutBuffer(),
fData_map()
{
    
}

DigitWriteoutBuffer::DigitWriteoutBuffer(TString branchname, TString foldername, Bool_t persistance):
FairWriteoutBuffer(branchname, "AliceO2::ITS::Digit", foldername, persistance),
fData_map()
{
    
}

DigitWriteoutBuffer::~DigitWriteoutBuffer(){
    
}

void DigitWriteoutBuffer::AddNewDataToTClonesArray(FairTimeStamp *timestamp){
    FairRootManager *iohandler = FairRootManager::Instance();
    TClonesArray *outputarray = iohandler->GetTClonesArray(fBranchName);
    
    new ((*outputarray)[outputarray->GetEntries()])Digit(*(static_cast<Digit *>(timestamp)));
}

double DigitWriteoutBuffer::FindTimeForData(FairTimeStamp *timestamp){
    Digit itsdigit = *(static_cast<Digit *>(timestamp));
    std::map<Digit, double>::iterator result = fData_map.find(itsdigit);
    if (result != fData_map.end()) {
        return result->second;
    }
    return -1;
}

void DigitWriteoutBuffer::FillDataMap(FairTimeStamp *data, double activeTime){
    Digit itsdigit = *(static_cast<Digit *>(data));
    fData_map[itsdigit] = activeTime;
}

void DigitWriteoutBuffer::EraseDataFromDataMap(FairTimeStamp *data){
    Digit itsdigit = *(static_cast<Digit *>(data));
    if (fData_map.find(itsdigit) != fData_map.end()) {
        fData_map.erase(itsdigit);
    }
}