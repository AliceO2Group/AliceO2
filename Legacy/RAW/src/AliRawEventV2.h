#ifndef ALIRAWEVENTV2_H
#define ALIRAWEVENTV2_H
// Author: Cvetan Cheshkov  11/05/2009

/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

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
// The AliRunDB class provides the interface to the run and file        //
// catalogues (AliEn or plain MySQL).                                   //
// The AliStats class provides statics information that is added as     //
// a single keyed object to each raw file.                              //
// The AliTagDB provides an interface to a TAG database.                //
// The AliMDC class is usid by the "alimdc" stand-alone program         //
// that reads data directly from DATE.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include <TObject.h>
#endif

#ifndef ROOT_TClonesArray
#include <TClonesArray.h>
#endif


// Forward class declarations
class AliRawEventHeaderBase;
class AliRawVEquipment;
class AliRawEquipmentV2;

#include "AliRawVEvent.h"

class AliRawEventV2 : public AliRawVEvent {

public:
   AliRawEventV2();
   virtual ~AliRawEventV2();

   AliRawEventHeaderBase *GetHeader(char*& data);
   virtual AliRawEventHeaderBase *GetHeader();
   virtual Int_t                  GetNEquipments() const;
   AliRawEquipmentV2             *NextEquipment();
   virtual AliRawVEquipment      *GetEquipment(Int_t index) const;
   virtual Int_t                  GetNSubEvents() const { return (fEvtHdrs) ? (fEvtHdrs->GetEntriesFast()-1) : 0; }
   AliRawEventV2                 *NextSubEvent();
   virtual AliRawVEvent          *GetSubEvent(Int_t index);
   void                           Reset();
   virtual void	                  Clear(Option_t* = "");

private:
   TClonesArray           fEquipments;  // AliRawEquipmentV2's
   TClonesArray          *fEvtHdrs;     //-> event and subevent headers

   Int_t                  fIndex;       //!
   Int_t                  fNAllocHdrs;  //!
   Int_t                  fNAllocEqs;   //!

   AliRawEventV2(const AliRawEventV2& rawEvent);
   AliRawEventV2& operator = (const AliRawEventV2& rawEvent);

   ClassDef(AliRawEventV2,1)  // ALICE raw event object
};

#endif
