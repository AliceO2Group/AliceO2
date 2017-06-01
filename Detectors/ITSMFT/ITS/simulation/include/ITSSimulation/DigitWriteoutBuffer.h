// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
#include "ITSMFTBase/Digit.h"            // for Digit

namespace o2 {
namespace ITS {
class DigitWriteoutBuffer : public FairWriteoutBuffer
{
  public:
    DigitWriteoutBuffer();

    DigitWriteoutBuffer(TString branchname, TString foldername, Bool_t persistance);

    ~DigitWriteoutBuffer() override;

    // Implementation of virtual function required by the interface
    void AddNewDataToTClonesArray(FairTimeStamp *) override;

    double FindTimeForData(FairTimeStamp *) override;

    void FillDataMap(FairTimeStamp *data, double activeTime) override;

    void EraseDataFromDataMap(FairTimeStamp *data) override;

  protected:
    std::map<o2::ITSMFT::Digit, double> mData_map;

  ClassDefOverride(DigitWriteoutBuffer, 1);
};
}
}

#endif /* defined(ALICEO2_ITS_DIGITWRITEOUTBUFFER_H_) */
