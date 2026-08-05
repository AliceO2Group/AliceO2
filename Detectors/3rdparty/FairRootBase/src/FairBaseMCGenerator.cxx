/*
 * FairSimpleMCGenerator.cxx
 *
 *  Created on: 16 mar 2020
 *      Author: Daniel Wielanek
 *		E-mail: daniel.wielanek@gmail.com
 *		Warsaw University of Technology, Faculty of Physics
 */
#include "FairBaseMCGenerator.h"

#include "FairLogger.h"

#include <TDatabasePDG.h>
#include <TParticlePDG.h>
#include <TRandom.h>

FairBaseMCGenerator::FairBaseMCGenerator()
    : FairGenerator()
    , fX(0)
    , fY(0)
    , fZ(0)
    , fPDGType(0)
    , fMult(0)
    , fVertexSmear(kBox)
    , fPDGMass(0.0)
    , fVx(0.0)
    , fVy(0.0)
    , fVz(0.0)
    , fVex(0.0)
    , fVey(0.0)
    , fVez(0.0)
{}

void FairBaseMCGenerator::SetVertex(Double_t vx,
                                    Double_t vy,
                                    Double_t vz,
                                    Double_t evx,
                                    Double_t evy,
                                    Double_t evz,
                                    eVertexSmear VertexSmear)
{
    fX = fVx = vx;
    fY = fVy = vy;
    fZ = fVz = vz;
    fVex = evx;
    fVey = evy;
    fVez = evz;
    fVertexSmear = VertexSmear;
}

FairBaseMCGenerator::~FairBaseMCGenerator()
{
    // TODO Auto-generated destructor stub
}

Bool_t FairBaseMCGenerator::Init()
{
    if (fMult <= 0)
        return kFALSE;
    TDatabasePDG *pid = TDatabasePDG::Instance();
    TParticlePDG *p = pid->GetParticle(fPDGType);
    if (p != nullptr) {
        LOG(info) << this->ClassName() << ": particle with PDG =" << GetPDGType() << " Found";
        fPDGMass = p->Mass();
    }
    return kTRUE;
}

void FairBaseMCGenerator::SetPDGType(Int_t pdg) { fPDGType = pdg; }

void FairBaseMCGenerator::GenerateEventParameters()
{
    switch (fVertexSmear) {
        case kGauss: {
            fX = gRandom->Gaus(fVx, fVex);
            fY = gRandom->Gaus(fVy, fVey);
            fZ = gRandom->Gaus(fVz, fVez);
        } break;
        case kBox: {
            fX = gRandom->Uniform(fVx - fVex, fVx + fVex);
            fY = gRandom->Uniform(fVy - fVey, fVy + fVey);
            fZ = gRandom->Uniform(fVz - fVez, fVz + fVez);
        } break;
        case kExp: {
            if (gRandom->Uniform() < 0.5) {
                fX = fVx + gRandom->Exp(fVex);
            } else {
                fX = fVx - gRandom->Exp(fVex);
            }
            if (gRandom->Uniform() < 0.5) {
                fY = fVy + gRandom->Exp(fVey);
            } else {
                fY = fVy - gRandom->Exp(fVey);
            }
            if (gRandom->Uniform() < 0.5) {
                fZ = fVz + gRandom->Exp(fVez);
            } else {
                fZ = fVz - gRandom->Exp(fVez);
            }
        } break;
    }
    LOG(info) << this->ClassName() << ": Event,  vertex = (" << fX << "," << fY << "," << fZ << ") cm,  multiplicity "
              << fMult;
}
