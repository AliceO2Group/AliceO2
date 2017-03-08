// Author: Cvetan Cheshkov  11/05/2009

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
// AliRawEventV2                                                          //
//                                                                      //
// Set of classes defining the ALICE RAW event format. The AliRawEventV2  //
// class defines a RAW event. It consists of an AliEventHeader object   //
// an AliEquipmentHeader object, an AliRawData object and an array of   //
// sub-events, themselves also being AliRawEventV2s. The number of        //
// sub-events depends on the number of DATE LDC's.                      //
// The AliRawEventV2 objects are written to a ROOT file using different   //
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

#include <TObjArray.h>
#include <TClass.h>

#include "AliLog.h"

#include "AliRawEventHeaderBase.h"
#include "AliRawEquipmentV2.h"

#include "AliRawEventV2.h"


ClassImp(AliRawEventV2)


//______________________________________________________________________________
AliRawEventV2::AliRawEventV2():
AliRawVEvent(),
fEquipments("AliRawEquipmentV2",1000),
fEvtHdrs(NULL),
fIndex(0),
fNAllocHdrs(0),
fNAllocEqs(0)
{
   // Create ALICE event object. If ownData is kFALSE we will use a static
   // raw data object, otherwise a private copy will be made.

}

//______________________________________________________________________________
AliRawEventHeaderBase *AliRawEventV2::GetHeader(char*& data)
{
  // Get event header part of AliRawEventV2.
  // First the DATE version is identified and then the
  // corresponding event header version object is created

  AliRawEventHeaderBase *hdr = NULL;

  if (!fEvtHdrs) {
    hdr = AliRawEventHeaderBase::Create(data);
    hdr->IsA()->IgnoreTObjectStreamer();
    fEvtHdrs = new TClonesArray(hdr->IsA()->GetName(),100);
    delete hdr;
  }

  if (fIndex < fNAllocHdrs) {
    TClonesArray &arr = *fEvtHdrs;
    return (AliRawEventHeaderBase *)arr[fIndex];
  }
  else {
    fNAllocHdrs = fIndex + 1;
    return (AliRawEventHeaderBase *)fEvtHdrs->New(fIndex);
  }
}

//______________________________________________________________________________
AliRawEventHeaderBase *AliRawEventV2::GetHeader()
{
  AliRawEventHeaderBase *hdr = NULL;
  if (!fEvtHdrs || !(hdr = (AliRawEventHeaderBase *)fEvtHdrs->UncheckedAt(fIndex))) {
    AliFatal("Event header does not exist!");
    return NULL;
  }

  return hdr;
}

//______________________________________________________________________________
AliRawEquipmentV2 *AliRawEventV2::NextEquipment()
{
   // Returns next equipment object.

  AliRawEventHeaderBase *hdr = (AliRawEventHeaderBase *)fEvtHdrs->UncheckedAt(fIndex);
  Int_t nEquipments = fEquipments.GetEntriesFast();
  hdr->AddEqIndex(nEquipments);

  if (nEquipments < fNAllocEqs) {
    return (AliRawEquipmentV2 *)fEquipments[nEquipments];
  }
  else {
    fNAllocEqs = nEquipments + 1;
    return new (fEquipments[nEquipments]) AliRawEquipmentV2();
  }
}

//______________________________________________________________________________
AliRawVEquipment *AliRawEventV2::GetEquipment(Int_t index) const
{
   // Get specified equipment. Returns 0 if equipment does not exist.

  //   if (!fEquipments)
  //      return NULL;

   AliRawEventHeaderBase *hdr = NULL;
   if (!fEvtHdrs || !(hdr = (AliRawEventHeaderBase *)fEvtHdrs->UncheckedAt(fIndex))) {
     AliFatal("Header is not yet initialized!");
     return NULL;
   }

   if ((index + hdr->GetFirstEqIndex()) > hdr->GetLastEqIndex()) {
     AliFatal("Equipment index out of scope!");
     return NULL;
   }     

   return (AliRawVEquipment *) fEquipments.UncheckedAt(index+hdr->GetFirstEqIndex());
}


//______________________________________________________________________________
Int_t AliRawEventV2::GetNEquipments() const
{
  //   if (!fEquipments)
  //      return 0;
  
   AliRawEventHeaderBase *hdr = NULL;
   if (!fEvtHdrs || !(hdr = (AliRawEventHeaderBase *)fEvtHdrs->UncheckedAt(fIndex))) {
     AliFatal("Header is not yet initialized!");
     return 0;
   }

   return (hdr->GetFirstEqIndex() < 0) ? 0 : (hdr->GetLastEqIndex() - hdr->GetFirstEqIndex() + 1);
}

//______________________________________________________________________________
AliRawEventV2 *AliRawEventV2::NextSubEvent()
{
   // Returns next sub-event object.

  fIndex++;

  return this;
}

//______________________________________________________________________________
AliRawVEvent *AliRawEventV2::GetSubEvent(Int_t index)
{
   // Get specified sub event. Returns 0 if sub event does not exist.

  if (!fEvtHdrs) {
    AliFatal("Headers are not yet initialized!");
    return NULL;
  }

  fIndex = index + 1;

  return this;
}

//______________________________________________________________________________
void AliRawEventV2::Reset()
{
   // Reset the event in case it needs to be re-used (avoiding costly
   // new/delete cycle). We reset the size marker for the AliRawData
   // objects and the sub event counter.

  fEquipments.Clear();

  if (fEvtHdrs) {
    for (int i = 0; i < fEvtHdrs->GetEntriesFast(); i++) {
      AliRawEventHeaderBase *hdr = (AliRawEventHeaderBase *)fEvtHdrs->UncheckedAt(i);
      hdr->Reset();
    }
    fEvtHdrs->Clear();
  }
  fIndex = 0;
}

//______________________________________________________________________________
AliRawEventV2::~AliRawEventV2()
{
   // Clean up event object. Delete also, possible, private raw data.

  //   if (fEquipments)
  fEquipments.Delete();
  //   delete fEquipments;
   if (fEvtHdrs)
      fEvtHdrs->Delete();
   delete fEvtHdrs;
}

//______________________________________________________________________________
void AliRawEventV2::Clear(Option_t*)
{
   // Clear the event in case it needs to be re-used (avoiding costly
   // new/delete cycle). Can be used inside the event loop.

  fEquipments.Clear("C");

  if (fEvtHdrs) {
    for (int i = 0; i < fEvtHdrs->GetEntriesFast(); i++) {
      AliRawEventHeaderBase *hdr = (AliRawEventHeaderBase *)fEvtHdrs->UncheckedAt(i);
      hdr->Reset();
    }
    fEvtHdrs->Clear();
  }
  fIndex = 0;
}

