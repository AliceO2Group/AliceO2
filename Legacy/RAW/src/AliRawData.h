#ifndef ALIRAWDATA_H
#define ALIRAWDATA_H
// @(#) $Id$
// Author: Fons Rademakers  26/11/99

/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliRawData                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include <TObject.h>
#endif


class AliRawData : public TObject {

public:
   AliRawData();
   virtual ~AliRawData() { if (fOwner) delete [] fRawData; }

   inline void SetBuffer(void *buf, Int_t size);
   Int_t       GetSize() const { return fSize; }
   void       *GetBuffer() { return fRawData; }

private:
   Int_t   fSize;         // number of raw data bytes
   char   *fRawData;      //[fSize] raw event data
   Bool_t  fOwner;        //!if true object owns fRawData buffer

   AliRawData(const AliRawData &);
   AliRawData &operator=(const AliRawData &);

   ClassDef(AliRawData,2)  // Alice raw event buffer
};

void AliRawData::SetBuffer(void *buf, Int_t size)
{
   if (fOwner) delete [] fRawData;
   fRawData = (char *) buf;
   fSize    = size;
   fOwner   = kFALSE;
}

#endif
