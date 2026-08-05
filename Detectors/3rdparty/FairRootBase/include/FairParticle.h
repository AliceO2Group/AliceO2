/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/// Class FAIRParticle
/// ------------------
/// Extended TParticle with persistent pointers to mother and daughters
/// particles  (Ivana Hrivnacova, 5.4.2002)
/// Used to define particles which will be added to Geant3/4 (M. Al-Turany)

#ifndef FAIR_PARTICLE_H
#define FAIR_PARTICLE_H

#include <Rtypes.h>            // for Int_t, Double_t, Bool_t, etc
#include <TMCParticleType.h>   // for TMCParticleType
#include <TObject.h>           // for TObject
#include <TRef.h>              // for TRef
#include <TRefArray.h>         // for TRefArray
#include <TString.h>           // for TString

class TParticle;

class FairParticle : public TObject
{
  public:
    FairParticle(Int_t id, TParticle* particle);
    FairParticle(Int_t id, TParticle* particle, FairParticle* mother);
    FairParticle(const char* name,
                 Int_t z,
                 Int_t a,
                 Int_t s,
                 Double_t mass,
                 Int_t q,
                 Bool_t stable,
                 Double_t decaytime);
    FairParticle(const char* name, Int_t z, Int_t a, Double_t mass, Int_t q, Bool_t stable, Double_t decaytime);
    FairParticle(Int_t pdg,
                 const TString name,
                 TMCParticleType mcType,
                 Double_t mass,
                 Double_t charge,
                 Double_t lifetime,
                 const TString pType = "Ion",
                 Double_t width = 0,
                 Int_t iSpin = 0,
                 Int_t iParity = 0,
                 Int_t iConjugation = 0,
                 Int_t iIsospin = 0,
                 Int_t iIsospinZ = 0,
                 Int_t gParity = 0,
                 Int_t lepton = 0,
                 Int_t baryon = 0,
                 Bool_t stable = kFALSE);

    FairParticle();

    virtual ~FairParticle();

    // methods
    void SetMother(FairParticle* particle);
    void AddDaughter(FairParticle* particle);
    virtual void Print(Option_t* option = "") const;
    void PrintDaughters() const;

    // get methods
    Int_t GetPDG() const;
    TParticle* GetParticle() const;
    FairParticle* GetMother() const;
    Int_t GetNofDaughters() const;
    FairParticle* GetDaughter(Int_t i) const;
    virtual const char* GetName() const { return fname.Data(); }
    TMCParticleType GetMCType() { return fmcType; }
    Double_t GetMass() { return fmass; }
    Double_t GetCharge() { return fcharge; }
    Double_t GetDecayTime() { return fDecayTime; }
    const TString& GetPType() { return fpType; }
    Double_t GetWidth() { return fwidth; }
    Int_t GetSpin() { return fiSpin; }
    Int_t GetiParity() { return fiParity; }
    Int_t GetConjugation() { return fiConjugation; }
    Int_t GetIsospin() { return fiIsospin; }
    Int_t GetIsospinZ() { return fiIsospinZ; }
    Int_t GetgParity() { return fgParity; }
    Int_t GetLepton() { return flepton; }
    Int_t GetBaryon() { return fbaryon; }
    Bool_t IsStable() { return fstable; }

  private:
    FairParticle(const FairParticle& P);
    FairParticle& operator=(const FairParticle&) { return *this; }
    // data members
    Int_t fpdg;
    TParticle* fParticle;
    TRef fMother;
    TRefArray fDaughters;
    const TString fname;
    TMCParticleType fmcType;
    Double_t fmass;
    Double_t fcharge;
    Double_t fDecayTime;
    const TString fpType;
    Double_t fwidth;
    Int_t fiSpin;
    Int_t fiParity;
    Int_t fiConjugation;
    Int_t fiIsospin;
    Int_t fiIsospinZ;
    Int_t fgParity;
    Int_t flepton;
    Int_t fbaryon;
    Bool_t fstable;

    ClassDef(FairParticle, 3);
};

#endif   // FAIR_PARTICLE_H
