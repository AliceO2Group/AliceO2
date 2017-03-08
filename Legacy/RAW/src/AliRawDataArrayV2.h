#ifndef ALIRAWDATAARRAYV2_H
#define ALIRAWDATAARRAYV2_H

// Author: Cvetan Cheshkov  27/03/2007

/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliRawDataArrayV2                                                    //
// A container object which is used in order to write the sub-detector  //
// raw-data payloads into a separate branches                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include <TObject.h>
#endif
#include <TClonesArray.h>

class AliRawData;

class AliRawDataArrayV2 : public TObject {

public:
   AliRawDataArrayV2();
   AliRawDataArrayV2(Int_t n);
   virtual ~AliRawDataArrayV2();

   void ClearData();
   AliRawData *Add();

private:
   TClonesArray  fRawDataArray; // Array containing raw-data payloads
   Int_t         fNAlloc;       //!

   AliRawDataArrayV2(const AliRawDataArrayV2 &);      // not implemented, usage causes
   AliRawDataArrayV2 &operator=(const AliRawDataArrayV2 &);  // link time error

   ClassDef(AliRawDataArrayV2,1)  // Alice raw event buffer
};

#endif
