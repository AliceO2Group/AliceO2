#ifndef ALIRAWEQUIPMENTV2_H
#define ALIRAWEQUIPMENTV2_H
// Author: Cvetan Cheshkov 11/05/2009

/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

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
#include <TRef.h>

// Forward class declarations
class AliRawDataArrayV2;

#include "AliRawVEquipment.h"
#include "AliRawEquipmentHeader.h"

class AliRawEquipmentV2 : public AliRawVEquipment {

public:
   AliRawEquipmentV2();
   virtual ~AliRawEquipmentV2();

   virtual AliRawEquipmentHeader *GetEquipmentHeader();
   virtual AliRawData            *GetRawData();
   void                           Reset();
   virtual void	                  Clear(Option_t* = "");
   AliRawData                    *NextRawData(AliRawDataArrayV2 *array);

   virtual void                   CloneRawData(const AliRawData *rawData);

private:
   AliRawEquipmentHeader  fEqpHdr;      // equipment header
   AliRawData            *fRawData;     //! raw data container
   TRef                   fRawDataRef;  // reference to raw data container

   AliRawEquipmentV2(const AliRawEquipmentV2& eq);
   AliRawEquipmentV2& operator = (const AliRawEquipmentV2& eq);

   ClassDef(AliRawEquipmentV2,1)  // ALICE raw equipment object
};

#endif
