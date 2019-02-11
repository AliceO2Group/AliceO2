#ifndef AliVParticle_H
#define AliVParticle_H
/* Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//     base class for ESD and AOD particles
//     Author: Markus Oldenburg, CERN
//-------------------------------------------------------------------------

#include <Rtypes.h>
#include <TObject.h>
#include "AliVMisc.h"

class TLorentzVector;
class TParticle;

#include <float.h>

const Double_t kAlmost1=1. - Double_t(FLT_EPSILON);
const Double_t kAlmost0=Double_t(FLT_MIN);

const Double_t kB2C=-0.299792458e-3;
const Double_t kAlmost0Field=1.e-13;

class AliVParticle: public TObject {

public:
  AliVParticle() { }
  virtual ~AliVParticle() { }
  AliVParticle(const AliVParticle& vPart); 
  AliVParticle& operator=(const AliVParticle& vPart);

  // constructor for reinitialisation of vtable
  AliVParticle( AliVConstructorReinitialisationFlag ) :TObject(){}

  // kinematics
  virtual Double_t Px() const = 0;
  virtual Double_t Py() const = 0;
  virtual Double_t Pz() const = 0;
  virtual Double_t Pt() const = 0;
  virtual Double_t P()  const = 0;
  virtual Bool_t   PxPyPz(Double_t p[3]) const = 0;

  virtual void     Momentum(TLorentzVector &)  { ; }
  
  virtual Double_t Xv() const = 0;
  virtual Double_t Yv() const = 0;
  virtual Double_t Zv() const = 0;
  virtual Bool_t   XvYvZv(Double_t x[3]) const = 0;  
//virtual Double_t T()          const { return -1; } // Conflicts with AliAODTrack.h
  virtual Double_t Tv()         const { return 0 ; } 

  virtual Double_t OneOverPt()  const = 0;
  virtual Double_t Phi()        const = 0;
  virtual Double_t Theta()      const = 0;


  virtual Double_t E()          const = 0;
  virtual Double_t M()          const = 0;
  
  virtual Double_t Eta()        const = 0;
  virtual Double_t Y()          const = 0;
  
  virtual Short_t Charge()      const = 0;
  
  virtual Int_t   Label()       const { return -1; } 
  virtual Int_t   GetLabel()    const = 0;
  // PID
  virtual Int_t   PdgCode()     const = 0;       
  virtual const Double_t *PID() const = 0; // return PID object (to be defined, still)

  // Not possible GetStatus(), Long in AliVTrack, Int in AliMCParticle  
  virtual ULong64_t  GetStatus()    const { return 0  ; }
  virtual UInt_t  MCStatusCode() const { return 0  ; }
  
  virtual TParticle *Particle()  const { return NULL ; }

  /** Compare this class with an other instance of this class
   *  used in a TCollection::Sort()/TClonesArray::Sort()
   *  @param   obj  ptr to other instance
   *  @return  Returns 0 when equal, 1 when this is smaller
   *  and -1 when bigger -- sorts descending
   */
  Int_t Compare( const TObject* obj) const;
  
    
  /** Defines this class as being sortable in a TCollection
   *  @return     always kTRUE;
   */
  Bool_t IsSortable() const  { return kTRUE; }

  virtual void    SetFlag(UInt_t) {;}
  virtual UInt_t  GetFlag() const {return 0;}  

  // coordinate system conversions
  Bool_t   Local2GlobalMomentum(Double_t p[3], Double_t alpha) const;
  Bool_t   Local2GlobalPosition(Double_t r[3], Double_t alpha) const;
  Bool_t   Global2LocalMomentum(Double_t p[3], Short_t charge, Double_t &alpha) const;
  Bool_t   Global2LocalPosition(Double_t r[3], Double_t alpha) const;

  //Navigation
  virtual Int_t   GetMother()   const {return -1;}
  virtual Int_t   GetFirstDaughter()   const {return -1;}
  virtual Int_t   GetLastDaughter()    const {return -1;}
  // Cannot use GetDaughter because of AliAODRecoDecay
//virtual Int_t   GetDaughter(Int_t)      const {return -1;}
  virtual Int_t   GetDaughterLabel(Int_t) const {return -1;}
  virtual Int_t   GetNDaughters  ()       const {return 0 ;}
  
  virtual void    SetGeneratorIndex(Short_t) {;}
  virtual Short_t GetGeneratorIndex() const {return -1;}
  
  virtual void    SetPrimary(Bool_t)   { ; }
  virtual Bool_t  IsPrimary()          const { return 0  ; }
  
  virtual void    SetPhysicalPrimary(Bool_t ) { ; }
  virtual Bool_t  IsPhysicalPrimary()  const { return 0  ; }
  
  virtual void    SetSecondaryFromWeakDecay(Bool_t ) { ; }
  virtual Bool_t  IsSecondaryFromWeakDecay() const { return 0  ; }
  
  virtual void    SetSecondaryFromMaterial(Bool_t ) { ; }
  virtual Bool_t  IsSecondaryFromMaterial() const { return 0  ; }
  
  ClassDef(AliVParticle, 4)  // base class for particles
};

#endif
