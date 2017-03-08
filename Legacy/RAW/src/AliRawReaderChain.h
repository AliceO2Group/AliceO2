#ifndef ALIRAWREADERCHAIN_H
#define ALIRAWREADERCHAIN_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
///
/// This is a class for reading raw data from a root chain.
///
///////////////////////////////////////////////////////////////////////////////

#include "AliRawReaderRoot.h"
#include <TString.h>

class TChain;
class TFileCollection;
class TEntryList;

class AliRawReaderChain: public AliRawReaderRoot {
  public :
    AliRawReaderChain();
    AliRawReaderChain(const char* fileName);
    AliRawReaderChain(TFileCollection *collection);
    AliRawReaderChain(TChain *chain);
    AliRawReaderChain(TEntryList *elist);
    AliRawReaderChain(Int_t runNumber);
    AliRawReaderChain(const AliRawReaderChain& rawReader);
    AliRawReaderChain& operator = (const AliRawReaderChain& rawReader);
    virtual ~AliRawReaderChain();

    virtual Bool_t   NextEvent();
    virtual Bool_t   RewindEvents();
    virtual Bool_t   GotoEvent(Int_t event);
    virtual Int_t    GetNumberOfEvents() const;

    virtual TChain*  GetChain() const { return fChain; }
    //
    static const char* GetSearchPath()                               {return fgSearchPath;}
    static       void  SetSearchPath(const char* path="/alice/data");
  protected :
    TChain*          fChain;        // root chain with raw events
    static TString   fgSearchPath;   // search path for "find"
    ClassDef(AliRawReaderChain, 0) // class for reading raw digits from a root file
};

#endif
