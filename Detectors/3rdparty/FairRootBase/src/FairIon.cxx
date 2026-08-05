/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                         FairIon source file                    -----
// -------------------------------------------------------------------------
#include "FairIon.h"

#include "FairLogger.h"

#include <TDatabasePDG.h>
#include <TParticlePDG.h>

FairIon::FairIon()
    : TNamed()
    , fZ(0)
    , fA(0)
    , fQ(0)
    , fExcEnergy(0)
    , fMass(0)
    , fLogger(FairLogger::GetLogger())
{}

FairIon::FairIon(const char* name, Int_t z, Int_t a, Int_t q, Double_t e, Double_t mass)
    : TNamed(name, "User defined ion")
    , fZ(z)
    , fA(a)
    , fQ(q)
    , fExcEnergy(e)
    , fMass(0)
    , fLogger(FairLogger::GetLogger())
{
    TDatabasePDG* pdgDB = TDatabasePDG::Instance();
    TParticlePDG* kProton = pdgDB->GetParticle(2212);
    Double_t kProtonMass = kProton->Mass();

    if (mass == 0.) {
        fMass = kProtonMass * Double_t(a);
    } else {
        fMass = mass;
    }
    LOG(info) << "New Ion " << name << " Z=" << z << "  A=" << a << "  Charge=" << q << "  Mass=" << fMass << " GeV ";
}

FairIon::~FairIon(){};

ClassImp(FairIon);
