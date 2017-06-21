#ifndef ALIRAWVEVENT_H
#define ALIRAWVEVENT_H
// Author: Cvetan Cheshkov  11/05/2009

/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliRawVEvent                                                         //
//                                                                      //
// Set of classes defining the ALICE RAW event format. The AliRawVEvent //
// class defines a RAW event. It consists of an AliEventHeader object   //
// an AliEquipmentHeader object, an AliRawData object and an array of   //
// sub-events, themselves also being AliRawVEvents. The number of        //
// sub-events depends on the number of DATE LDC's.                      //
// The AliRawVEvent objects are written to a ROOT file using different   //
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


class AliRawVEvent : public TObject {

public:
   AliRawVEvent() { }
   virtual ~AliRawVEvent() { }

   virtual AliRawEventHeaderBase *GetHeader() = 0;
   virtual Int_t                  GetNEquipments() const = 0;
   virtual AliRawVEquipment      *GetEquipment(Int_t index) const = 0;
   virtual Int_t                  GetNSubEvents() const = 0;
   virtual AliRawVEvent          *GetSubEvent(Int_t index) = 0;

private:

   AliRawVEvent(const AliRawVEvent& rawEvent);
   AliRawVEvent& operator = (const AliRawVEvent& rawEvent);

   ClassDef(AliRawVEvent,1)  // ALICE raw event object
};

#endif
