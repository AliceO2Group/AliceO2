#ifndef ALIHEADER_H
#define ALIHEADER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-----------------------------------------------------------------------
//     Simulation event header class
//     Collaborates with AliRun, AliStack, and AliGenReaderTreeK classes
//     Many other classes depend on it
//-----------------------------------------------------------------------

#include <TObject.h>
#include <time.h>

class AliStack;
class AliGenEventHeader;
class AliDetectorEventHeader;
class TObjArray;

class AliHeader : public TObject {
public:
  AliHeader();
  AliHeader(const AliHeader& head);
  AliHeader(Int_t run, Int_t event);
  AliHeader(Int_t run, Int_t eventSerialNr, Int_t evNrInRun);
  virtual ~AliHeader();
  

  virtual void Reset(Int_t run, Int_t event);
  virtual void Reset(Int_t run, Int_t eventSerialNr, Int_t evNrInRun);

  virtual  void  SetRun(Int_t run) {fRun = run;}
  virtual  Int_t GetRun() const {return fRun;}
  
  virtual  void  SetNprimary(Int_t nprimary) {fNprimary = nprimary;}
  virtual  Int_t GetNprimary()   const {return fNprimary;}
  virtual  Int_t GetNsecondary() const {return fNtrack-fNprimary;}
  
  virtual  void  SetNvertex(Int_t vertex) {fNvertex = vertex;}
  virtual  Int_t GetNvertex() const {return fNvertex;}
  
  virtual  void  SetNtrack(Int_t ntrack) {fNtrack = ntrack;}
  virtual  Int_t GetNtrack() const {return fNtrack;}
  
  virtual  void  SetEvent(Int_t event) {fEvent = event;}
  virtual  Int_t GetEvent() const {return fEvent;}

  virtual  void  SetEventNrInRun(Int_t event) {fEventNrInRun = event;}
  virtual  Int_t GetEventNrInRun() const {return fEventNrInRun;}

  virtual  AliStack* Stack() const;
  virtual  void SetStack(AliStack* stack);

  virtual  void SetGenEventHeader(AliGenEventHeader* header);
  virtual  AliGenEventHeader*  GenEventHeader() const;

  virtual void AddDetectorEventHeader(AliDetectorEventHeader* header);
  virtual AliDetectorEventHeader* GetDetectorEventHeader(const char *name) const;
  
  virtual  void SetTimeStamp(time_t timeStamp) {fTimeStamp = timeStamp;}
  virtual  time_t GetTimeStamp() const {return fTimeStamp;}
  
  virtual void Print(const char *opt=0) const;

  Int_t    GetSgPerBgEmbedded()      const {return fSgPerBgEmbedded;}
  void     SetSgPerBgEmbedded(int i)       {fSgPerBgEmbedded = i;}
  Int_t    GetBgReuseID()            const {return fSgPerBgEmbedded ? (fEventNrInRun%fSgPerBgEmbedded) : 0;}
  
  AliHeader& operator=(const AliHeader& head) 
    {head.Copy(*this); return *this;}
  
protected:

  void Copy(TObject& head) const;

  Int_t         fRun;               //Run number
  Int_t         fNvertex;           //Number of vertices
  Int_t         fNprimary;          //Number of primary tracks
  Int_t         fNtrack;            //Number of tracks
  Int_t         fEvent;             //Event number (serial in the file)
  Int_t         fEventNrInRun;      //Unique Event number within the run
  Int_t         fSgPerBgEmbedded;   //  in case this is embedded signal: bg.event repetition factor
  time_t        fTimeStamp;         //Event time-stamp
  AliStack     *fStack;             //Pointer to stack
  AliGenEventHeader* fGenHeader;    //Event Header for Generator
  TObjArray*         fDetHeaders;   //Event Headers for detector specific information 

  ClassDef(AliHeader,5) //Alice event header    
};

#endif
