// -*- mode: C++ -*- 
#ifndef ALIMCEVENT_H
#define ALIMCEVENT_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


/* $Id$ */

//-------------------------------------------------------------------------
//                          Class AliMCEvent
//      
// Origin: Andreas.Morsch, CERN, andreas.morsch@cern.ch 
//-------------------------------------------------------------------------


#include <TTree.h>
#include <TRefArray.h>
#include <TClonesArray.h>

#include "AliVEvent.h"
#include "AliVHeader.h"
#include "AliVParticle.h"
#include "AliVVertex.h"
#include "AliMCParticle.h"

class AliCentrality;
class AliEventplane;
class AliStack;
class AliHeader;
class AliGenEventHeader;

class TClonesArray;
class TList;

class AliMCEvent : public AliVEvent {

public:

    AliMCEvent();
    virtual ~AliMCEvent();
    AliMCEvent(const AliMCEvent& mcEvnt); 
    AliMCEvent& operator=(const AliMCEvent& mcEvnt);
    //
    // Methods implementing the interface
    //
    // Services
    virtual void AddObject(TObject* /*obj*/)               {;}
    virtual TObject* FindListObject(const char */*name*/)  const {return 0;}
    virtual TList* GetList() const                         {return 0;}
    virtual void CreateStdContent()                        {;} 
    virtual void GetStdContent()                           {;}
    virtual void ReadFromTree(TTree * /*tree*/, Option_t* /*opt*/) {;}
    virtual void WriteToTree(TTree* /*tree*/)  const       {;}

    virtual void SetStdNames()                             {;}
    virtual void Print(Option_t */*option=""*/)  const     {;}
    virtual void PreReadAll();
    virtual void Reset()                                   {;}

    // Header
    virtual AliVHeader* GetHeader()          const         {return 0;}

    // Delegated methods for fESDRun or AODHeader
  
    virtual void     SetRunNumber(Int_t /*n*/)             {;}
    virtual void     SetPeriodNumber(UInt_t /*n*/)         {;}
    virtual void     SetMagneticField(Double_t /*mf*/)     {;}
    
  
    virtual Int_t    GetRunNumber()          const         {return 0;}
    virtual UInt_t   GetPeriodNumber()       const         {return 0;}
    virtual Double_t GetMagneticField()      const         {return 0.;}

    // Setters not needed
    virtual void      SetOrbitNumber(UInt_t /*n*/)         {;}
    virtual void      SetBunchCrossNumber(UShort_t /*n*/)  {;}
    virtual void      SetEventType(UInt_t /*eventType*/)   {;}
    virtual void      SetTriggerMask(ULong64_t /*n*/)      {;}
    virtual void      SetTriggerCluster(UChar_t /*n*/)     {;} 

    virtual UInt_t    GetOrbitNumber()        const {return 0;}
    virtual UShort_t  GetBunchCrossNumber()   const {return 0;}
    
    virtual UInt_t    GetEventType()          const {return 0;}

    virtual ULong64_t GetTriggerMask()        const {return 0;}
    virtual UChar_t   GetTriggerCluster()     const {return 0;}
    virtual TString   GetFiredTriggerClasses()const {return ("");}
    virtual Double_t  GetZDCN1Energy()        const {return 0.;}
    virtual Double_t  GetZDCP1Energy()        const {return 0.;}
    virtual Double_t  GetZDCN2Energy()        const {return 0.;}
    virtual Double_t  GetZDCP2Energy()        const {return 0.;}
    virtual Double_t  GetZDCEMEnergy(Int_t /*i*/) 
                                              const {return 0.;}
    // Tracks
    virtual AliVParticle *GetTrack(Int_t i) const;
    virtual Int_t     GetNumberOfTracks()    const {return fNparticles;}
    virtual Int_t     GetNumberOfV0s()       const {return -1;}
    virtual Int_t     GetNumberOfCascades()  const {return -1;}

    Int_t Raw2MergedLabel(int lbRaw) const;
  
    TParticle* ParticleFromStack(Int_t i) const;
    TParticle* Particle(int i) const;
    // Vertex
    using AliVEvent::GetPrimaryVertex;
    virtual const AliVVertex *GetPrimaryVertex() const;
    
    //
    // MC Specific methods
    //
    // Getters
    AliStack*    Stack()   {return fStack;}
    AliHeader*   Header()  {return fHeader;}
    AliGenEventHeader* GenEventHeader() const;
    // Services
    virtual void      ConnectTreeE (TTree* tree);
    virtual void      ConnectTreeK (TTree* tree);
    virtual void      ConnectHeaderAndStack(AliHeader* header);
    virtual void      ConnectTreeTR(TTree* tree);
    virtual void      Clean();
    virtual void      InitEvent();
    virtual void      FinishEvent();
    virtual Int_t     GetParticleAndTR(Int_t i, TParticle*& particle, TClonesArray*& trefs);
    virtual void      DrawCheck(Int_t i, Int_t search);
    virtual void      AddSubsidiaryEvent(AliMCEvent* event);
    virtual Int_t     GetNumberOfPrimaries() const {return fNprimaries;}
    virtual Int_t     GetPrimaryOffset()    const {return fPrimaryOffset;}
    virtual Int_t     GetSecondaryOffset()  const {return fSecondaryOffset;}    
    virtual void      SetPrimaryOffset(Int_t ioff)    {fPrimaryOffset = ioff;}
    virtual void      SetSecondaryOffset(Int_t ioff)  {fSecondaryOffset = ioff;}    
    virtual Bool_t    IsPhysicalPrimary(Int_t i) const;

     Int_t            GetPrimary(Int_t id);
  
    virtual Int_t     BgLabelToIndex(Int_t label);
    static  Int_t     BgLabelOffset() {return fgkBgLabelOffset;}
    virtual Bool_t    IsFromBGEvent(Int_t index);
    TList*  GetCocktailList();
    TString  GetGenerator(Int_t index); 
    Bool_t GetCocktailGenerator(Int_t index,TString &nameGen);
    virtual Bool_t    IsSecondaryFromWeakDecay(Int_t index);
    virtual Bool_t    IsSecondaryFromMaterial(Int_t index);
    // External particle array
  virtual void      SetParticleArray(TClonesArray* mcParticles); 
    //External Header 
     virtual void SetExternalHeader(AliVHeader* aodmcHeader)
       {fAODMCHeader=aodmcHeader;}  
  virtual AliGenEventHeader *FindHeader(Int_t ipart);
    //Following needed only for mixed event
  virtual Int_t        EventIndex(Int_t)       const {return 0;}
  virtual Int_t        EventIndexForCaloCluster(Int_t) const {return 0;}
  virtual Int_t        EventIndexForPHOSCell(Int_t)    const {return 0;}
  virtual Int_t        EventIndexForEMCALCell(Int_t)   const {return 0;} 
  AliCentrality*       GetCentrality()  {return 0;} 
  AliEventplane*       GetEventplane()  {return 0;} 

  virtual AliVVZERO    *GetVZEROData() const {return 0;}
  virtual AliVZDC      *GetZDCData()   const {return 0;}

  virtual AliVEvent::EDataLayoutType GetDataLayoutType() const;

  virtual Int_t     FindIndexAndEvent(Int_t oldidx, AliMCEvent*& event) const; //RS

  Bool_t HasSubsidiaries() const {return fSubsidiaryEvents!=0;}
  Bool_t IsFromSubsidiaryEvent(int id) const;
  void   SetTopEvent(const AliMCEvent* ev) {fTopEvent = ev;}
  const AliMCEvent* GetTopEvent() const {return fTopEvent;}

  static AliMCParticle *GetDummyTrack();
  
private:
    virtual void      ReorderAndExpandTreeTR();

    void 	      UpdateEventInformation();
    virtual void      AssignGeneratorIndex();    
    virtual void      AssignGeneratorIndex(Int_t index, Int_t dmin, Int_t dmax);    

  using AliVEvent::GetVZEROData;
    
private: 
    // Stanndard implementation for ESD production
    AliStack         *fStack;            // Current pointer to stack
    TClonesArray     *fMCParticles;      // Pointer to list of particles
    TObjArray        *fMCParticleMap;    // Map of MC Particles
    AliHeader        *fHeader;           // Current pointer to header
    AliVHeader       *fAODMCHeader;      //Current pointer to AODMC header
    TClonesArray     *fTRBuffer;         // Track reference buffer    
    TClonesArray     *fTrackReferences;  // Array of track references
    TTree            *fTreeTR;           // Pointer to Track Reference Tree
    TTree            *fTmpTreeTR;        // Temporary tree TR to read old format
    TFile            *fTmpFileTR;        // Temporary file with TreeTR to read old format
    Int_t             fNprimaries;       // Number of primaries
    Int_t             fNparticles;       // Number of particles
    TList            *fSubsidiaryEvents; // List of possible subsidiary events (for example merged underlying event)
    Int_t             fPrimaryOffset;    // Offset for primaries
    Int_t             fSecondaryOffset;  // Offset for secondaries
    Bool_t            fExternal;         // True if external particle array
    const AliMCEvent* fTopEvent;         //! Top MCEvent (if not embedded, then itself)
    static   Int_t        fgkBgLabelOffset;  // Standard branch name    
    mutable  AliVVertex*  fVertex;           // MC Vertex
    Int_t             fNBG;              //! Background particles in current event
    
    ClassDef(AliMCEvent, 2)              // AliVEvent realisation for MC data
};


#endif 

