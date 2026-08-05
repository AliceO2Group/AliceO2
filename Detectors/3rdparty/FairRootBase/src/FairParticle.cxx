/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                   FairParticle source file                    -----
// -----            Created 06/01/04  by M. Al-Turany                  -----
// -------------------------------------------------------------------------

// Extended TParticle with pointers to mother and daughters particles

#include "FairParticle.h"

#include <TDatabasePDG.h>      // for TDatabasePDG
#include <TMCParticleType.h>   // for TMCParticleType::kPTIon, etc
#include <TParticle.h>         // for TParticle
#include <TParticlePDG.h>      // for TParticlePDG
#include <iostream>            // for operator<<, basic_ostream, etc

using std::cout;
using std::endl;

ClassImp(FairParticle);

FairParticle::FairParticle(Int_t id, TParticle* particle)
    : TObject()
    , fpdg(id)
    , fParticle(particle)
    , fMother(0)
    , fDaughters(0)
    , fname(particle->GetName())
    , fmcType(kPTIon)
    , fmass(particle->GetMass())
    , fcharge(particle->GetPDG()->Charge())
    , fDecayTime(particle->GetPDG()->Lifetime())
    , fpType("Ion")
    , fwidth(particle->GetPDG()->Width())
    , fiSpin(static_cast<Int_t>(particle->GetPDG()->Spin()))
    , fiParity(particle->GetPDG()->Parity())
    , fiConjugation(0)
    , fiIsospin(static_cast<Int_t>(particle->GetPDG()->Isospin()))
    , fiIsospinZ(0)
    , fgParity(0)
    , flepton(0)
    , fbaryon(0)
    , fstable(particle->GetPDG()->Stable())
{}

FairParticle::FairParticle(const char* name,
                           Int_t z,
                           Int_t a,
                           Int_t s,
                           Double_t mass,
                           Int_t q,
                           Bool_t stable,
                           Double_t decaytime)
    : TObject()
    , fpdg(1000000000 + 10000000 * s + 10000 * z + 10 * a)
    , fParticle(0)
    , fMother(0)
    , fDaughters(0)
    , fname(name)
    , fmcType(kPTIon)
    , fmass(0)
    , fcharge(0)
    , fDecayTime(decaytime)
    , fpType("Ion")
    , fwidth(0)
    , fiSpin(0)
    , fiParity(0)
    , fiConjugation(0)
    , fiIsospin(0)
    , fiIsospinZ(0)
    , fgParity(0)
    , flepton(0)
    , fbaryon(0)
    , fstable(stable)
{

    TDatabasePDG* pdgDB = TDatabasePDG::Instance();
    if (!pdgDB->GetParticle(fpdg)) {
        if (mass == 0) {
            TParticlePDG* kProton = pdgDB->GetParticle(2212);
            Double_t kProtonMass = kProton->Mass();
            mass = a * kProtonMass;
        }
        pdgDB->AddParticle(name, name, mass, stable, 0, q, "kPTHadron", fpdg);
    }

    fParticle = new TParticle();
    fParticle->SetPdgCode(fpdg);

    fmcType = kPTHadron;
    fmass = mass;
    fcharge = fParticle->GetPDG()->Charge();
    fDecayTime = decaytime;
    fwidth = fParticle->GetPDG()->Width();
    fiSpin = static_cast<Int_t>(fParticle->GetPDG()->Spin());
    fiParity = fParticle->GetPDG()->Parity();
    fiConjugation = 0;
    fiIsospin = static_cast<Int_t>(fParticle->GetPDG()->Isospin());
    fiIsospinZ = 0;
    fgParity = 0;
    flepton = 0;
    fbaryon = 0;
    fstable = fParticle->GetPDG()->Stable();
}

FairParticle::FairParticle(const char* name,
                           Int_t z,
                           Int_t a,
                           Double_t mass,
                           Int_t q,
                           Bool_t stable,
                           Double_t decaytime)
    : TObject()
    , fpdg(10000000 + 10000 * z + 10 * a)
    , fParticle(0)
    , fMother(0)
    , fDaughters(0)
    , fname(name)
    , fmcType(kPTIon)
    , fmass(0)
    , fcharge(0)
    , fDecayTime(decaytime)
    , fpType("Ion")
    , fwidth(0)
    , fiSpin(0)
    , fiParity(0)
    , fiConjugation(0)
    , fiIsospin(0)
    , fiIsospinZ(0)
    , fgParity(0)
    , flepton(0)
    , fbaryon(0)
    , fstable(stable)
{
    //  fpdg= 10000000+10000* z +10 * a;

    //  fDecayTime= decaytime;
    TDatabasePDG* pdgDB = TDatabasePDG::Instance();

    if (!pdgDB->GetParticle(fpdg)) {
        if (mass == 0) {
            TParticlePDG* kProton = pdgDB->GetParticle(2212);
            Double_t kProtonMass = kProton->Mass();
            mass = a * kProtonMass;
        }
        pdgDB->AddParticle(name, name, mass, stable, 0, q, "kPTHadron", fpdg);
    }

    fParticle = new TParticle();
    fParticle->SetPdgCode(fpdg);

    fmcType = kPTHadron;
    fmass = mass;
    fcharge = fParticle->GetPDG()->Charge();
    fDecayTime = decaytime;
    fwidth = fParticle->GetPDG()->Width();
    fiSpin = static_cast<Int_t>(fParticle->GetPDG()->Spin());
    fiParity = fParticle->GetPDG()->Parity();
    fiConjugation = 0;
    fiIsospin = static_cast<Int_t>(fParticle->GetPDG()->Isospin());
    fiIsospinZ = 0;
    fgParity = 0;
    flepton = 0;
    fbaryon = 0;
    fstable = fParticle->GetPDG()->Stable();
}

FairParticle::FairParticle(Int_t id, TParticle* particle, FairParticle* mother)
    : TObject()
    , fpdg(id)
    , fParticle(particle)
    , fMother(mother)
    , fDaughters()
    , fname(particle->GetName())
    , fmcType(kPTIon)
    , fmass(0)
    , fcharge(0)
    , fDecayTime(0)
    , fpType("Ion")
    , fwidth(0)
    , fiSpin(0)
    , fiParity(0)
    , fiConjugation(0)
    , fiIsospin(0)
    , fiIsospinZ(0)
    , fgParity(0)
    , flepton(0)
    , fbaryon(0)
    , fstable(kTRUE)

{}

FairParticle::FairParticle(Int_t pdg,
                           const TString name,
                           TMCParticleType mcType,
                           Double_t mass,
                           Double_t charge,
                           Double_t lifetime,
                           const TString pType,
                           Double_t width,
                           Int_t iSpin,
                           Int_t iParity,
                           Int_t iConjugation,
                           Int_t iIsospin,
                           Int_t iIsospinZ,
                           Int_t gParity,
                           Int_t lepton,
                           Int_t baryon,
                           Bool_t stable)
    : TObject()
    , fpdg(pdg)
    , fParticle(new TParticle())
    , fMother(0)
    , fDaughters(0)
    , fname(name)
    , fmcType(mcType)
    , fmass(mass)
    , fcharge(charge)
    , fDecayTime(lifetime)
    , fpType(pType)
    , fwidth(width)
    , fiSpin(iSpin)
    , fiParity(iParity)
    , fiConjugation(iConjugation)
    , fiIsospin(iIsospin)
    , fiIsospinZ(iIsospinZ)
    , fgParity(gParity)
    , flepton(lepton)
    , fbaryon(baryon)
    , fstable(stable)
{

    if (!TDatabasePDG::Instance()->GetParticle(fpdg)) {
        TDatabasePDG::Instance()->AddParticle(fname, fname, fmass, fstable, fwidth, fcharge, pType, fpdg);
    }

    fParticle->SetPdgCode(fpdg);
}

FairParticle::FairParticle()
    : TObject()
    , fpdg(0)
    , fParticle(0)
    , fMother(0)
    , fDaughters()
    , fname("ion")
    , fmcType(kPTIon)
    , fmass(0)
    , fcharge(0)
    , fDecayTime(0)
    , fpType("Ion")
    , fwidth(0)
    , fiSpin(0)
    , fiParity(0)
    , fiConjugation(0)
    , fiIsospin(0)
    , fiIsospinZ(0)
    , fgParity(0)
    , flepton(0)
    , fbaryon(0)
    , fstable(kTRUE)
{}

FairParticle::~FairParticle() { delete fParticle; }

//
// public methods
//

void FairParticle::SetMother(FairParticle* particle)
{
    // Adds particles daughter
    // ---

    fMother.SetObject(particle);
}

void FairParticle::AddDaughter(FairParticle* particle)
{
    // Adds particles daughter
    // ---

    fDaughters.Add(particle);
}

void FairParticle::Print(Option_t*) const
{
    // Prints particle properties.
    // ---
    //  return;

    cout << "Particle: " << fParticle->GetName() << "  with ID:  " << fpdg << endl;

    // fParticle->Print();

    if (GetMother()) {
        cout << "Mother:    " << GetMother()->GetParticle()->GetName() << "  with ID: " << GetMother()->GetPDG()
             << endl;
    } else {
        cout << "Primary    " << endl;
    }

    cout << "Number of daughters: " << GetNofDaughters() << endl;
    cout << endl;
}

void FairParticle::PrintDaughters() const
{
    // Prints particles daughters.
    // ---

    for (Int_t i = 0; i < GetNofDaughters(); i++) {
        cout << i << "th daughter: " << endl;
        GetDaughter(i)->Print();
    }
}

Int_t FairParticle::GetPDG() const
{
    // Returs particle ID.
    // ---

    return fpdg;
}

TParticle* FairParticle::GetParticle() const
{
    // Returns particle definition (TParticle).
    // ---

    return fParticle;
}

FairParticle* FairParticle::GetMother() const
{
    // Returns particle definition (TParticle).
    // ---

    return static_cast<FairParticle*>(fMother.GetObject());
}

Int_t FairParticle::GetNofDaughters() const
{
    // Returns number of daughters.
    // ---

    return fDaughters.GetEntriesFast();
}

FairParticle* FairParticle::GetDaughter(Int_t i) const
{
    // Returns i-th daughter.
    // ---

    if (i < 0 || i >= GetNofDaughters()) {
        Fatal("GetDaughter", "Index out of range");
    }

    return static_cast<FairParticle*>(fDaughters.At(i));
}
