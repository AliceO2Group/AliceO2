// @(#) $Id: AliRawEquipment.cxx 23318 2008-01-14 12:43:28Z hristov $
// Author: Fons Rademakers  26/11/99


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
// AliRawEquipmentV2                                                    //
//                                                                      //
// Set of classes defining the ALICE RAW event format. The AliRawEventV2//
// class defines a RAW event. It consists of an AliEventHeader object   //
// an AliEquipmentHeader object, an AliRawData object and an array of   //
// sub-events, themselves also being AliRawEventV2s. The number of      //
// sub-events depends on the number of DATE LDC's.                      //
// The AliRawEventV2 objects are written to a ROOT file using different //
// technologies, i.e. to local disk via AliRawDB or via rfiod using     //
// AliRawRFIODB or via rootd using AliRawRootdDB or to CASTOR via       //
// rootd using AliRawCastorDB (and for performance testing there is     //
// also AliRawNullDB).                                                  //
// The AliStats class provides statics information that is added as     //
// a single keyed object to each raw file.                              //
// The AliTagDB provides an interface to a TAG database.                //
// The AliMDC class is usid by the "alimdc" stand-alone program         //
// that reads data directly from DATE.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "AliRawEquipmentV2.h"
#include "AliRawDataArrayV2.h"
#include "AliRawData.h"

ClassImp(AliRawEquipmentV2)

//______________________________________________________________________________
AliRawEquipmentV2::AliRawEquipmentV2():
AliRawVEquipment(),
fEqpHdr(),
fRawData(NULL),
fRawDataRef(NULL)
{
   // Create ALICE equipment object.

}

//______________________________________________________________________________
AliRawEquipmentHeader *AliRawEquipmentV2::GetEquipmentHeader()
{
   // Get equipment header part of AliRawEquipmentV2.

   return &fEqpHdr;
}

//______________________________________________________________________________
AliRawData *AliRawEquipmentV2::GetRawData()
{
   // Get raw data part of AliRawEquipmentV2.

  if (!fRawData) {
    if (fRawDataRef.IsValid()) {
      fRawData = (AliRawData*)fRawDataRef.GetObject();
    }
  }
  return fRawData;
}

//______________________________________________________________________________
void AliRawEquipmentV2::Reset()
{
   // Reset the equipment in case it needs to be re-used (avoiding costly
   // new/delete cycle). We reset the size marker for the AliRawData
   // object.

   fEqpHdr.Reset();
   fRawDataRef = NULL;
}

//______________________________________________________________________________
void AliRawEquipmentV2::Clear(Option_t*)
{
   // Clear the equipment in case it needs to be re-used (avoiding costly
   // new/delete cycle). Called by AliRawEventV2 Clear method inside the event loop.

   fEqpHdr.Reset();
   fRawDataRef = NULL;
   fRawData = NULL;
}

//______________________________________________________________________________
AliRawEquipmentV2::~AliRawEquipmentV2()
{
   // Clean up event object. Delete also, possible, private raw data.

   if (!fRawDataRef.IsValid()) delete fRawData;
}

//______________________________________________________________________________
AliRawData *AliRawEquipmentV2::NextRawData(AliRawDataArrayV2 *array)
{
  // Get a pointer to the raw-data object
  // stored within an array in a separate
  // branch of the tree.
  // Set the reference to the raw-data object

  AliRawData *raw = NULL;
  if (array) {
    raw = array->Add();
    fRawDataRef = raw;
  }
  else {
    Error("NextRawData", "Raw-data array does not exist! Can not set a reference to a raw-data object!");    
    fRawDataRef = NULL;
  }

  return raw;
}

//______________________________________________________________________________
void AliRawEquipmentV2::CloneRawData(const AliRawData *rawData)
{
  // Clone the input raw data and
  // flush the TRef

  fRawDataRef = NULL;
  if (rawData) fRawData = (AliRawData*)rawData->Clone();
}
