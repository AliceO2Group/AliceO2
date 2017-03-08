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
// AliRawDataArray                                                      //
// A container object which is used in order to write the sub-detector  //
// raw-data payloads into a separate branches                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <TObjArray.h>

#include "AliRawDataArray.h"


ClassImp(AliRawDataArray)

AliRawDataArray::AliRawDataArray():
fRawDataArray(NULL)
{
  // Default constructor
}

AliRawDataArray::AliRawDataArray(Int_t n):
fRawDataArray(new TObjArray(n))
{
  // Default constructor
}

AliRawDataArray::~AliRawDataArray()
{
  if (fRawDataArray) delete fRawDataArray;
}

void AliRawDataArray::ClearData()
{
  if (fRawDataArray) fRawDataArray->Clear();
}

void AliRawDataArray::Add(AliRawData *data)
{
  if (fRawDataArray)
    fRawDataArray->Add((TObject *)data);
  else
    Error("Add", "TObjArray is not initialized! Cannot add raw-data payload!");
}
