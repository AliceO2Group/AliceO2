//
//  DigitWriteoutBuffer.h
//  ALICEO2
//
//  Created by Markus Fasel on 21.07.15.
//
//

#ifndef ALICEO2_ITS_DIGITWRITEOUTBUFFER_H_
#define ALICEO2_ITS_DIGITWRITEOUTBUFFER_H_

#include <map>
#include <TString.h>             // for TString
#include "FairWriteoutBuffer.h"  // for FairWriteoutBuffer
#include "Rtypes.h"              // for DigitWriteoutBuffer::Class, Bool_t, etc
#include "ITSSimulation/Digit.h"            // for Digit

namespace AliceO2 {
namespace ITS {
class DigitWriteoutBuffer : public FairWriteoutBuffer
{
  public:
    DigitWriteoutBuffer();

    DigitWriteoutBuffer(TString branchname, TString foldername, Bool_t persistance);

    virtual ~DigitWriteoutBuffer();

    // Implementation of virtual function required by the interface
    void AddNewDataToTClonesArray(FairTimeStamp *);

    virtual double FindTimeForData(FairTimeStamp *);

    virtual void FillDataMap(FairTimeStamp *data, double activeTime);

    virtual void EraseDataFromDataMap(FairTimeStamp *data);

  protected:
    std::map<Digit, double> fData_map;

  ClassDef(DigitWriteoutBuffer, 1);
};
}
}

#endif /* defined(ALICEO2_ITS_DIGITWRITEOUTBUFFER_H_) */
