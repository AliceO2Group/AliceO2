#ifndef AliVHeader_H
#define AliVHeader_H
/* Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//     Event header base class
//     Author: Markus Oldenburg, CERN
//-------------------------------------------------------------------------

#include <TNamed.h>
#include <TBits.h>
class TList;
class AliVHeader : public TNamed {

 public :
  AliVHeader(); 
  virtual ~AliVHeader() { };
  AliVHeader(const AliVHeader& evt); 
  AliVHeader& operator=(const AliVHeader& evt);
  
  virtual UShort_t  GetBunchCrossNumber()   const = 0;
  virtual UInt_t    GetOrbitNumber()        const = 0;
  virtual UInt_t    GetPeriodNumber()       const = 0;
  virtual UInt_t    GetTimeStamp()          const = 0;
  virtual ULong64_t GetTriggerMask()        const = 0;
  virtual ULong64_t GetTriggerMaskNext50()  const = 0;
  virtual UChar_t   GetTriggerCluster()     const = 0;
  virtual UInt_t    GetEventType()          const = 0;
  virtual TList*    GetCocktailHeaders()    {return 0;}
  ULong64_t GetEventIdAsLong() const;
  virtual Bool_t    IsTriggerInputFired(const char* /*name*/) const {return kFALSE;}
      
  /*
  virtual void SetBunchCrossNumber(UShort_t nBx)   = 0;
  virtual void SetOrbitNumber(UInt_t nOr)          = 0;
  virtual void SetPeriodNumber(UInt_t nPer)        = 0;
  virtual void SetTriggerMask(ULong64_t trigMsk)   = 0;
  virtual void SetTriggerCluster(UChar_t trigClus) = 0;
  virtual void SetEventType(UInt_t evttype)        = 0;
  */
  virtual UInt_t   GetL0TriggerInputs() const {return 0;}
  virtual UInt_t   GetL1TriggerInputs() const {return 0;}
  virtual UShort_t GetL2TriggerInputs() const {return 0;}
  virtual TBits  GetIRInt2InteractionMap() const { static TBits dummy; return dummy; }
  virtual TBits  GetIRInt1InteractionMap() const { static TBits dummy; return dummy; }

  virtual Int_t  GetIRInt2ClosestInteractionMap()                    const {return 0;}
  virtual Int_t  GetIRInt1ClosestInteractionMap(Int_t =3/*gap = 3*/) const {return 0;}


  virtual void Print(Option_t* option = "") const  = 0;
    
  ClassDef(AliVHeader,1);
};

#endif
