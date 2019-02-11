#ifndef ALIMCPARTICLE_H
#define ALIMCPARTICLE_H
/* Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//     AliVParticle realisation for MC Particles
//     Author: Andreas Morsch, CERN
//-------------------------------------------------------------------------

#include <Rtypes.h>
#include <TParticle.h>
#include <TParticlePDG.h>
#include <TObjArray.h>

#include "AliTrackReference.h"
#include "AliVParticle.h"
#include "AliStack.h"

class AliMCParticle: public AliVParticle {
public:
    AliMCParticle();
    AliMCParticle(TParticle* part, TObjArray* rarray = 0, Int_t label=-1);
    virtual ~AliMCParticle();
    AliMCParticle(const AliMCParticle& mcPart); 
    AliMCParticle& operator=(const AliMCParticle& mcPart);
    
    // Kinematics
    virtual Double_t Px()        const;
    virtual Double_t Py()        const;
    virtual Double_t Pz()        const;
    virtual Double_t Pt()        const;
    virtual Double_t P()         const;
    virtual Bool_t   PxPyPz(Double_t p[3]) const;
   
    virtual void     Momentum(TLorentzVector & lv)  { fParticle->Momentum(lv) ; }
  
    virtual Double_t OneOverPt() const;
    virtual Double_t Phi()       const;
    virtual Double_t Theta()     const;
    
    virtual Double_t Xv()        const;
    virtual Double_t Yv()        const;
    virtual Double_t Zv()        const;
    virtual Bool_t   XvYvZv(Double_t x[3]) const;  
    virtual Double_t T()         const;
    virtual Double_t Tv()        const;
  
    virtual Double_t E()          const;
    virtual Double_t M()          const;
    
    virtual Double_t Eta()        const;
    virtual Double_t Y()          const;
    
    virtual Short_t Charge()      const;

    virtual Int_t    Label()       const;
    virtual Int_t    GetLabel()    const  {return Label();}
    virtual Int_t    PdgCode()     const  {return fParticle->GetPdgCode();}
    virtual UInt_t   MCStatusCode() const {return fParticle->GetStatusCode();}
    virtual TParticle* Particle()  const  {return fParticle;}
    
    // PID
    virtual const Double_t *PID() const {return 0;} // return PID object (to be defined, still)

    // Track References
    Int_t              GetNumberOfTrackReferences() const {return fNTrackRef;}
    AliTrackReference* GetTrackReference(Int_t i)
      {return dynamic_cast<AliTrackReference*>((*fTrackReferences)[i]);}

    // "Trackable" criteria
    Float_t  GetTPCTrackLength(Float_t bz, Float_t ptmin, Int_t &counter, Float_t deadWidth, Float_t zMax=230. );
    // Navigation
    virtual Int_t GetMother()       const {return fMother;}
    Int_t GetFirstDaughter()        const {return fFirstDaughter;}
    Int_t GetLastDaughter()         const {return fLastDaughter;}
    Int_t GetDaughterLabel(Int_t i) const {return fParticle->GetDaughter(i) ;}
    Int_t GetNDaughters()           const {return fParticle->GetNDaughters();}
    
    void  SetMother(Int_t idx)        {fMother        = idx;}
    void  SetFirstDaughter(Int_t idx) {fFirstDaughter = idx;}
    void  SetLastDaughter(Int_t idx)  {fLastDaughter  = idx;}
    void  SetLabel(Int_t label)       {fLabel         = label;}
    virtual void    SetGeneratorIndex(Short_t i) {fGeneratorIndex = i;}
    virtual Short_t GetGeneratorIndex() const {return fGeneratorIndex;}

    const AliStack*  GetStack()           const {return fStack;}
    void             SetStack(AliStack* st)     {fStack = st  ;}
    Bool_t     IsPhysicalPrimary()        const {return fStack->IsPhysicalPrimary(fLabel);} 
    Bool_t     IsSecondaryFromWeakDecay() const {return fStack->IsSecondaryFromWeakDecay(fLabel);}
    Bool_t     IsSecondaryFromMaterial()  const {return fStack->IsSecondaryFromMaterial(fLabel);}

 private:
    TParticle *fParticle;             // The wrapped TParticle
    TObjArray *fTrackReferences;      // Array to track references
    Int_t      fNTrackRef;            // Number of track references
    Int_t      fLabel;                // fParticle Label in the Stack
    Int_t      fMother;               // Mother particles
    Int_t      fFirstDaughter;        // First daughter
    Int_t      fLastDaughter;         // LastDaughter
    Short_t    fGeneratorIndex;       // !Generator index in cocktail  
    AliStack*  fStack;                //! stack the particle belongs to

  ClassDef(AliMCParticle,1)  // AliVParticle realisation for MCParticles
};

inline Double_t AliMCParticle::Px()        const {return fParticle->Px();}
inline Double_t AliMCParticle::Py()        const {return fParticle->Py();}
inline Double_t AliMCParticle::Pz()        const {return fParticle->Pz();}
inline Double_t AliMCParticle::Pt()        const {return fParticle->Pt();}
inline Double_t AliMCParticle::P()         const {return fParticle->P(); }
inline Double_t AliMCParticle::OneOverPt() const {return 1. / fParticle->Pt();}
inline Bool_t   AliMCParticle::PxPyPz(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }
inline Double_t AliMCParticle::Phi()       const {return fParticle->Phi();}
inline Double_t AliMCParticle::Theta()     const {return fParticle->Theta();}
inline Double_t AliMCParticle::Xv()        const {return fParticle->Vx();}
inline Double_t AliMCParticle::Yv()        const {return fParticle->Vy();}
inline Double_t AliMCParticle::Zv()        const {return fParticle->Vz();}
inline Bool_t   AliMCParticle::XvYvZv(Double_t x[3]) const { x[0] = Xv(); x[1] = Yv(); x[2] = Zv(); return kTRUE; }
inline Double_t AliMCParticle::T()         const {return fParticle->T();}
inline Double_t AliMCParticle::Tv()        const {return fParticle->T();}
inline Double_t AliMCParticle::E()         const {return fParticle->Energy();}
inline Double_t AliMCParticle::Eta()       const {return fParticle->Eta();}


inline Double_t AliMCParticle::M()         const
{
    TParticlePDG* pdg = fParticle->GetPDG();
    if (pdg) {
	return (pdg->Mass());
    } else {
	return (fParticle->GetCalcMass());
    }
}


inline Double_t AliMCParticle::Y()         const 
{
    Double_t e  = E();
    Double_t pz = Pz();
    
    if ( e - TMath::Abs(pz) > FLT_EPSILON ) {
	return 0.5*TMath::Log((e+pz)/(e-pz));
    } else { 
	return -999.;
    }
}

inline Short_t AliMCParticle::Charge()     const
{
    TParticlePDG* pdg = fParticle->GetPDG();
    if (pdg) {
	return (Short_t (pdg->Charge()));
    } else {
	return -99;
    }
}

inline Int_t AliMCParticle::Label()       const {return fLabel;}

#endif
