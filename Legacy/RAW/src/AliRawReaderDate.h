#ifndef ALIRAWREADERDATE_H
#define ALIRAWREADERDATE_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a class for reading raw data from a date file or event.
///
///////////////////////////////////////////////////////////////////////////////

#include "AliRawReader.h"

struct eventHeaderStruct;
struct equipmentHeaderStruct;


class AliRawReaderDate: public AliRawReader {
  public :
    AliRawReaderDate(void* event, Bool_t owner = kFALSE);
    AliRawReaderDate(const char* fileName, Int_t eventNumber = -1);
    virtual ~AliRawReaderDate();

    virtual UInt_t   GetType() const;
    virtual UInt_t   GetRunNumber() const;
    virtual const UInt_t* GetEventId() const;
    virtual const UInt_t* GetTriggerPattern() const;
    virtual const UInt_t* GetDetectorPattern() const;
    virtual const UInt_t* GetAttributes() const;
    virtual const UInt_t* GetSubEventAttributes() const;
    virtual UInt_t   GetLDCId() const;
    virtual UInt_t   GetGDCId() const;
    virtual UInt_t   GetTimestamp() const;

    virtual Int_t    GetEquipmentSize() const;
    virtual Int_t    GetEquipmentType() const;
    virtual Int_t    GetEquipmentId() const;
    virtual const UInt_t* GetEquipmentAttributes() const;
    virtual Int_t    GetEquipmentElementSize() const;
    virtual Int_t    GetEquipmentHeaderSize() const;

    virtual Bool_t   ReadHeader();
    virtual Bool_t   ReadNextData(UChar_t*& data);
    virtual Bool_t   ReadNext(UChar_t* data, Int_t size);

    virtual Bool_t   Reset();

    virtual Bool_t   NextEvent();
    virtual Bool_t   RewindEvents();

    virtual Int_t    CheckData() const;

    virtual AliRawReader* CloneSingleEvent() const;

  protected :
    FILE*            fFile;         // DATE file
    eventHeaderStruct* fEvent;      // raw data super event
    eventHeaderStruct* fSubEvent;   // raw data sub event
    equipmentHeaderStruct* fEquipment; // raw data equipment header

    UChar_t*         fPosition;     // current position in the raw data
    UChar_t*         fEnd;          // end position of the current data block

    Bool_t           fOwner;        // is raw-reader owner of the raw event or not

  private:
    AliRawReaderDate(const AliRawReaderDate& rawReader); // Not implemented
    AliRawReaderDate& operator = (const AliRawReaderDate& rawReader); // Not implemented

    ClassDef(AliRawReaderDate, 0) // class for reading raw digits from a root file
};

#endif
