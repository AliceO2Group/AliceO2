#ifndef ALIRAWREADER_H
#define ALIRAWREADER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
///
/// This is the base class for reading raw data.
///
///////////////////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <TArrayI.h>
#include <TClonesArray.h>

#include "AliRawDataErrorLog.h"
#include "AliRawDataHeader.h"
#include "AliRawDataHeaderV3.h"

class THashList;
class TChain;
class AliRawEventHeaderBase;
class AliRawVEvent;

class AliRawReader: public TObject {
  friend class AliEveEventManager;
  public :
    AliRawReader();
    AliRawReader(const AliRawReader& rawReader);
    AliRawReader& operator = (const AliRawReader& rawReader);
    virtual ~AliRawReader();

    static  AliRawReader* Create(const char *uri);

    virtual void     Select(Int_t detectorID, 
			    Int_t minDDLID = -1, Int_t maxDDLID = -1);
    virtual void     Select(const char *detectorName, 
			    Int_t minDDLID = -1, Int_t maxDDLID = -1);
    virtual void     SelectEquipment(Int_t equipmentType, 
				     Int_t minEquipmentId = -1, 
				     Int_t maxEquipmentId = -1);
    virtual void     SkipInvalid(Bool_t skip = kTRUE)
      {fSkipInvalid = skip;};
    virtual void     RequireHeader(Bool_t required)
      {fRequireHeader = required;};

    virtual const AliRawEventHeaderBase* GetEventHeader() const {return NULL;};
    virtual const AliRawVEvent* GetEvent() const {return NULL;}

    virtual UInt_t   GetType() const = 0;
    virtual UInt_t   GetRunNumber() const = 0;
    virtual const UInt_t* GetEventId() const = 0;
    UInt_t                GetPeriod() const {
      const UInt_t *id = GetEventId();
      return id ? (((id)[0]>>4)&0x0fffffff): 0;
    }
    UInt_t                GetOrbitID() const {
      const UInt_t *id = GetEventId();
      return id ? ((((id)[0]<<20)&0xf00000)|(((id)[1]>>12)&0xfffff)) : 0;
    }
    UShort_t              GetBCID() const {
      const UInt_t *id = GetEventId();
      return id ? ((id)[1]&0x00000fff) : 0;
    }
    ULong64_t             GetEventIdAsLong() const {
      return (((ULong64_t)GetPeriod() << 36) |
	      ((ULong64_t)GetOrbitID() << 12) |
	      (ULong64_t)GetBCID()); 
    }
    virtual const UInt_t* GetTriggerPattern() const = 0;
    ULong64_t             GetClassMask() const {
      const UInt_t *pattern = GetTriggerPattern();
      return pattern ? (((ULong64_t)pattern[1] & 0x3ffff) << 32)|(pattern[0]) : 0;
    }
    ULong64_t GetClassMaskNext50() const  { 
      const UInt_t *pattern = GetTriggerPattern();
      return pattern ? ((ULong64_t)(pattern[1]>>18) & 0x3fff)|((ULong64_t)(pattern[2]&0xffffffff)<<14)|((ULong64_t)(pattern[3]&0xf)<<46): 0; 
    } 
    virtual const UInt_t* GetDetectorPattern() const = 0;
    virtual const UInt_t* GetAttributes() const = 0;
    virtual const UInt_t* GetSubEventAttributes() const = 0;
    virtual UInt_t   GetLDCId() const = 0;
    virtual UInt_t   GetGDCId() const = 0;
    virtual UInt_t   GetTimestamp() const = 0;

    virtual Int_t    GetEquipmentSize() const = 0;
    virtual Int_t    GetEquipmentType() const = 0;
    virtual Int_t    GetEquipmentId() const = 0;
    Int_t            GetMappedEquipmentId() const;
    Bool_t           LoadEquipmentIdsMap(const char *fileName);
    virtual const UInt_t* GetEquipmentAttributes() const = 0;
    virtual Int_t    GetEquipmentElementSize() const = 0;
    virtual Int_t    GetEquipmentHeaderSize() const = 0;

    Int_t            GetDetectorID() const;
    Int_t            GetDDLID() const;

    Int_t            GetDataSize() const 
      {if (fHeader) {
	  if (fHeader->fSize != 0xFFFFFFFF) return fHeader->fSize - sizeof(AliRawDataHeader); 
	  else return GetEquipmentSize() - GetEquipmentHeaderSize() - sizeof(AliRawDataHeader);
	} else if(fHeaderV3){
	  if (fHeaderV3->fSize != 0xFFFFFFFF) return fHeaderV3->fSize - sizeof(AliRawDataHeaderV3); 
	  else return GetEquipmentSize() - GetEquipmentHeaderSize() - sizeof(AliRawDataHeaderV3);
	}
	else return GetEquipmentSize() - GetEquipmentHeaderSize();};

    Int_t            GetVersion() const 
      {if (fHeader) return fHeader->GetVersion();
	else if (fHeaderV3) return fHeaderV3->GetVersion();
	else return -1;};
    Bool_t           IsValid() const 
      {if (fHeader) return fHeader->TestAttribute(0);
	else if (fHeaderV3) return fHeaderV3->TestAttribute(0);
	else return kFALSE;};
    Bool_t           IsCompressed() const 
      {if (fHeader) return fHeader->TestAttribute(1); 
	else if (fHeaderV3) return fHeaderV3->TestAttribute(1);
      else return kFALSE;};
    Bool_t           TestBlockAttribute(Int_t index) const
      {if (fHeader) return fHeader->TestAttribute(index);
	else if (fHeaderV3) return fHeaderV3->TestAttribute(index);
      else return kFALSE;};
    UChar_t          GetBlockAttributes() const 
      {if (fHeader) return fHeader->GetAttributes(); 
	else if (fHeaderV3) return fHeaderV3->GetAttributes();
      else return 0;};
    UInt_t           GetStatusBits() const
      {if (fHeader) return fHeader->GetStatus(); 
	else if (fHeaderV3) return fHeaderV3->GetStatus();
      else return 0;};
    const AliRawDataHeader* GetDataHeader() const
      {return fHeader;}
    const AliRawDataHeaderV3* GetDataHeaderV3() const
      {return fHeaderV3;}

    virtual Bool_t   ReadHeader() = 0;
    virtual Bool_t   ReadNextData(UChar_t*& data) = 0;
    virtual Bool_t   ReadNextInt(UInt_t& data);
    virtual Bool_t   ReadNextShort(UShort_t& data);
    virtual Bool_t   ReadNextChar(UChar_t& data);
    virtual Bool_t   ReadNext(UChar_t* data, Int_t size) = 0;

    virtual Bool_t   Reset() = 0;

    virtual Bool_t   NextEvent() = 0;
    virtual Bool_t   RewindEvents() = 0;
    virtual Bool_t   GotoEvent(Int_t event);
    virtual Bool_t   GotoEventWithID(Int_t event,
				     UInt_t period,
				     UInt_t orbitID,
				     UShort_t bcID);
    virtual Int_t    GetEventIndex() const { return -1; }
    virtual Int_t    GetNumberOfEvents() const { return -1; }

    enum {kErrMagic=1, kErrNoDataHeader=2, 
	  kErrSize=4, kErrOutOfBounds=8};
    virtual Int_t    CheckData() const;
    Int_t            GetErrorCode() const {return fErrorCode;};

    void             DumpData(Int_t limit = -1);

    void             AddErrorLog(AliRawDataErrorLog::ERawDataErrorLevel level,
				 Int_t code,
				 const char *message = NULL);
    void             AddMinorErrorLog(Int_t code,
				      const char *message = NULL) {
      return AddErrorLog(AliRawDataErrorLog::kMinor,code,message);
    }
    void             AddMajorErrorLog(Int_t code,
				      const char *message = NULL) {
      return AddErrorLog(AliRawDataErrorLog::kMajor,code,message);
    }
    void             AddFatalErrorLog(Int_t code,
				      const char *message = NULL) {
      return AddErrorLog(AliRawDataErrorLog::kFatal,code,message);
    }
    Int_t            GetNumberOfErrorLogs() const { return fErrorLogs.GetEntriesFast(); }
    const TClonesArray &GetAllErrorLogs() const { return fErrorLogs; }
    AliRawDataErrorLog *GetErrorLog(Int_t i) const {
      return (AliRawDataErrorLog *)fErrorLogs.UncheckedAt(i);
    }

    const TString&   GetParsedTriggerExpression() const {return fSelectTriggerExpr;}
    // Method which can be used in order to force the auto-save on
    // ESD tree inside AliReconstruction. For the moment it will be
    // activated only for AliRawReaderDateOnline.
    virtual Bool_t   UseAutoSaveESD() const { return kFALSE; }
    virtual TChain*  GetChain() const { return NULL; }

    Bool_t           IsRawReaderValid() const { return fIsValid; }

    void             LoadTriggerClass(const char* name, Int_t index);
    void             LoadTriggerAlias(const THashList *lst);

    virtual AliRawReader* CloneSingleEvent() const { return NULL; }

  protected :
    virtual void     SelectEvents(Int_t type,ULong64_t triggerMask=0,const char *triggerExpr=NULL,ULong64_t triggerMask50=0,const char *contExpr=NULL, const char *exclExpr=NULL);
    Bool_t           IsSelected() const;
    Bool_t           IsEventSelected() const;

    TArrayI         *fEquipmentIdsIn;       // array of equipment Ids to be mapped
    TArrayI         *fEquipmentIdsOut;      // array of mapped equipment Ids

    Bool_t           fRequireHeader;        // if false, data without header is accepted

    AliRawDataHeader* fHeader;              // current data header
    AliRawDataHeaderV3* fHeaderV3;              // current data header
    Int_t            fCount;                // counter of bytes to be read for current DDL

    Int_t            fSelectEquipmentType;  // type of selected equipment (<0 = no selection)
    Int_t            fSelectMinEquipmentId; // minimal index of selected equipment (<0 = no selection)
    Int_t            fSelectMaxEquipmentId; // maximal index of selected equipment (<0 = no selection)
    Bool_t           fSkipInvalid;          // skip invalid data
    Int_t            fSelectEventType;      // type of selected events (<0 = no selection)
    ULong64_t        fSelectTriggerMask;    // trigger mask for selecting events (0 = no selection)
    ULong64_t        fSelectTriggerMask50;  // trigger maskNext50 for selecting events (0 = no selection)
    TString          fSelectTriggerExpr;    // trigger expression for selecting events (empty = no selection)
    TString          fContainTriggerExpr;   // string required in trigger name for selecting events (empty = no selection)
    TString          fExcludeTriggerExpr;   // string required in trigger name for exluding events (empty = no selection)
    Bool_t           fVeto[100];            // veto for the 50+50 trigger masks

    Int_t            fErrorCode;            // code of last error

    Int_t            fEventNumber;          // current event number
    TClonesArray     fErrorLogs;            // raw data decoding errors

    AliRawDataHeader* fHeaderSwapped;       // temporary buffer for swapping header on PowerPC
    AliRawDataHeaderV3* fHeaderSwappedV3;       // temporary buffer for swapping header on PowerPC

    UInt_t SwapWord(UInt_t x) const;
    UShort_t SwapShort(UShort_t x) const;

    Bool_t           fIsValid;              // is raw-reader created successfully
    Bool_t           fIsTriggerClassLoaded; // flags the call to LoadTriggerClass

    ClassDef(AliRawReader, 0) // base class for reading raw digits
};

#endif
