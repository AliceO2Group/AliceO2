#ifndef ALIRAWDATAARRAY_H
#define ALIRAWDATAARRAY_H

// Author: Cvetan Cheshkov  27/03/2007

/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliRawDataArray                                                      //
// A container object which is used in order to write the sub-detector  //
// raw-data payloads into a separate branches                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include <TObject.h>
#endif

class TObjArray;
class AliRawData;

class AliRawDataArray : public TObject {

public:
   AliRawDataArray();
   AliRawDataArray(Int_t n);
   virtual ~AliRawDataArray();

   void ClearData();
   void Add(AliRawData *data);

private:
   TObjArray *fRawDataArray; // Array containing raw-data payloads

   AliRawDataArray(const AliRawDataArray &);      // not implemented, usage causes
   AliRawDataArray &operator=(const AliRawDataArray &);  // link time error

   ClassDef(AliRawDataArray,1)  // Alice raw event buffer
};

#endif
