#ifndef ALIESDCALOFRIEND_H
#define ALIESDCALOFRIEND_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/*
 


Author: R. GUERNANE LPSC Grenoble CNRS/IN2P3
*/

#include <TObject.h>
#include <map>

class TArrayI;

class AliESDCalofriend : public TObject 
{
public:
  AliESDCalofriend();
  AliESDCalofriend(const AliESDCalofriend& ctrig);
  virtual ~AliESDCalofriend();
  
  AliESDCalofriend& operator=(const AliESDCalofriend& ctrig);
  
  Bool_t  IsEmpty() {return (fNEntries == 0);}
  
  virtual void Reset() {fCurrent = -1;}
  
  void    Allocate(Int_t size);
  void    DeAllocate(        ); 
  
  Bool_t  Add(Int_t id, Int_t type, Int_t nsamples, Int_t samples[]);
  
  void    GetId(             Int_t&   idx                     ) const;
  Int_t   GetId(                                              ) const;
  void    GetType(           Int_t&   type                    ) const;
  Int_t   GetType(                                            ) const;
  
  void    GetNSamples(       Int_t& nsamples                  ) const; 
  Int_t   GetNSamples(                                        ) const; 
  void    GetSamples(        Int_t  samples[]                 ) const;
  
  void    GetEntry(          Int_t  idx, Int_t type           );
  
  Int_t   GetEntries(                                         ) const {return fNEntries;}
  
  virtual Bool_t Next();
  
  virtual void Copy(TObject& obj) const;
  
  virtual void Print(const Option_t* opt) const;
  
private:
  
  Int_t    fNEntries;
  Int_t    fCurrent;
  
  Int_t*   fId;                             // [fNEntries]
  Int_t*   fType;                           // [fNEntries]
  Int_t*   fNSamples;                       // [fNEntries]
  TArrayI* fSamples;                        //
  
  std::map<std::pair<int, int>, int>  fDict;
  
  ClassDef(AliESDCalofriend, 1)
};
#endif

