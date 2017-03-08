#ifndef ALIRAWREADERROOT_H
#define ALIRAWREADERROOT_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a class for reading raw data from a root file.
///
///////////////////////////////////////////////////////////////////////////////

#include "AliRawReader.h"

class AliRawVEvent;
class AliRawVEquipment;
class AliRawData;
class TFile;
class TBranch;


class AliRawReaderRoot: public AliRawReader {
  public :
    AliRawReaderRoot();
    AliRawReaderRoot(const char* fileName, Int_t eventNumber = -1);
    AliRawReaderRoot(AliRawVEvent* event, Int_t evID=-1);
    AliRawReaderRoot(const AliRawReaderRoot& rawReader);
    AliRawReaderRoot& operator = (const AliRawReaderRoot& rawReader);
    virtual ~AliRawReaderRoot();

    virtual const AliRawEventHeaderBase* GetEventHeader() const;

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
    virtual Bool_t   GotoEvent(Int_t event);
    virtual Int_t    GetEventIndex() const { return fEventIndex; }
    virtual Int_t    GetNumberOfEvents() const;

    virtual Int_t    CheckData() const;

    virtual const AliRawVEvent* GetEvent() const {return fEvent;}

    virtual AliRawReader* CloneSingleEvent() const;
    static Bool_t           GetUseOrder() {return fgUseOrder;}
    static void             UseOrder() {fgUseOrder = kTRUE;}

  protected :
    TFile*           fFile;         // raw data root file
    TBranch*         fBranch;       // branch of raw events
    Int_t            fEventIndex;   // index of the event in the tree
    AliRawVEvent*    fEvent;        // (super) event
    AliRawEventHeaderBase* fEventHeader; // (super) event header
    Int_t            fSubEventIndex; // index of current sub event
    AliRawVEvent*    fSubEvent;     // current sub event
    Int_t            fEquipmentIndex; // index of current equipment
    AliRawVEquipment*fEquipment;    // current equipment
    AliRawData*      fRawData;      // current raw data
    UChar_t*         fPosition;     // current position in the raw data
    UChar_t*         fEnd;          // end position of the current subevent
    Long64_t*        fIndex;       // Index of the tree
    static Bool_t    fgUseOrder;       // Flag to use or not sorting in decreased size order

    void SwapData(const void* inbuf, const void* outbuf, UInt_t size);
    void MakeIndex();


    ClassDef(AliRawReaderRoot, 0) // class for reading raw digits from a root file
};

#endif
