#ifndef ALIGENEVENTHEADER_H
#define ALIGENEVENTHEADER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//---------------------------------------------------------------------
// Event header base class for generator. 
// Stores generated event information
// Author: andreas.morsch@cern.ch
//---------------------------------------------------------------------

#include <map>
#include <string>
#include <TNamed.h>
#include <TArrayF.h>

class AliGenEventHeader : public TNamed
{
 public:

  AliGenEventHeader(const char* name);
  AliGenEventHeader();
  virtual ~AliGenEventHeader() {}
  // Getters
  virtual Int_t           NProduced()       const  {return fNProduced;}
  virtual Float_t         InteractionTime() const  {return fInteractionTime;}
  virtual Float_t         EventWeight()     const  {return fEventWeight;}
  virtual void            PrimaryVertex(TArrayF &o) const;
  // Setters
  virtual void   SetNProduced(Int_t nprod)         {fNProduced = nprod;}
  virtual void   SetPrimaryVertex(const TArrayF &o);
  virtual void   SetInteractionTime(Float_t t)     {fInteractionTime = t;}
  virtual void   SetEventWeight(Float_t w)         {AddEventWeight(fEventWeightNameGenerator, w);}

  // named event weights
  virtual void    AddEventWeight(const TString &name, Float_t w);
  virtual Float_t GetEventWeight(const TString &name);
  virtual const std::map<std::string, Float_t>& GetEventWeights() const
    { return fEventWeights; }
	  
protected:
  Int_t     fNProduced;                 // Number stable or undecayed particles
  TArrayF   fVertex;                    // Primary Vertex Position
  Float_t   fInteractionTime;           // Time of the interaction
  Float_t   fEventWeight;               // Event weight

  std::map<std::string, Float_t> fEventWeights; // named event weights
  const TString fEventWeightNameGenerator;      //! name for generator level weight

  ClassDef(AliGenEventHeader, 5)        // Event header for primary event
};

#endif
