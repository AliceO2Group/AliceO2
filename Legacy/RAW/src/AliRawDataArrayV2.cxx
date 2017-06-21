// Author: Cvetan Cheshkov 27/03/2007

/**************************************************************************
 * Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliRawDataArrayV2                                                    //
// A container object which is used in order to write the sub-detector  //
// raw-data payloads into a separate branches                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "AliRawDataArrayV2.h"
#include "AliRawData.h"


ClassImp(AliRawDataArrayV2)

AliRawDataArrayV2::AliRawDataArrayV2():
fRawDataArray("AliRawData",100),
fNAlloc(0)
{
  // Default constructor
}

AliRawDataArrayV2::AliRawDataArrayV2(Int_t n):
fRawDataArray("AliRawData",n),
fNAlloc(0)
{
  // Default constructor
}

AliRawDataArrayV2::~AliRawDataArrayV2()
{
  fRawDataArray.Delete();
}

void AliRawDataArrayV2::ClearData()
{
  fRawDataArray.Clear();
}

AliRawData *AliRawDataArrayV2::Add()
{
  Int_t nEntries = fRawDataArray.GetEntriesFast();
  if (nEntries < fNAlloc) {
    return (AliRawData *)fRawDataArray[nEntries];
  }
  else {
    fNAlloc = nEntries + 1;
    return new (fRawDataArray[nEntries]) AliRawData();
  }
}
