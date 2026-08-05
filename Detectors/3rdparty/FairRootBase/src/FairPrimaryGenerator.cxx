/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH *
 *                                                                              *
 *              This software is distributed under the terms of the *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE" *
 ********************************************************************************/

#include "FairPrimaryGenerator.h"

#include "FairGenerator.h"       // for FairGenerator
#include "FairGenericStack.h"    // for FairGenericStack
#include "FairLogger.h"          // for FairLogger, MESSAGE_ORIGIN
#include "FairMCEventHeader.h"   // for FairMCEventHeader

#include <TDatabasePDG.h>   // for TDatabasePDG
#include <TIterator.h>      // for TIterator
#include <TMath.h>          // for Sqrt
#include <TObject.h>        // for TObject
#include <TParticlePDG.h>   // for TParticlePDG
#include <TRandom.h>        // for TRandom, gRandom
#include <iostream>         // for operator<<, basic_ostream, etc

using std::cerr;
using std::cout;
using std::endl;

Int_t FairPrimaryGenerator::fTotPrim = 0;

FairPrimaryGenerator::FairPrimaryGenerator()
    : TNamed()
    , fBeamX0(0.)
    , fBeamY0(0.)
    , fBeamSigmaX(0.)
    , fBeamSigmaY(0.)
    , fBeamAngleX0(0.)
    , fBeamAngleY0(0.)
    , fBeamAngleX(0.)
    , fBeamAngleY(0.)
    , fBeamAngleSigmaX(0.)
    , fBeamAngleSigmaY(0.)
    , fBeamDirection(TVector3(0., 0., 1.))
    , fPhiMin(0.)
    , fPhiMax(0.)
    , fPhi(0.)
    , fTargetZ(new Double_t[1])
    , fNrTargets(1)
    , fTargetDz(0)
    , fVertex(TVector3(0., 0., 0.))
    , fNTracks(0)
    , fSmearVertexZ(kFALSE)
    , fSmearGausVertexZ(kFALSE)
    , fSmearVertexXY(kFALSE)
    , fSmearGausVertexXY(kFALSE)
    , fBeamAngle(kFALSE)
    , fEventPlane(kFALSE)
    , fStack(nullptr)
    , fGenList(new TObjArray())
    , fListIter(fGenList->MakeIterator())
    , fEvent(nullptr)
    , fdoTracking(kTRUE)
    , fMCIndexOffset(0)
    , fEventNr(0)
{
    fTargetZ[0] = 0.;
}

FairPrimaryGenerator::FairPrimaryGenerator(const char *name, const char *title)
    : TNamed(name, title)
    , fBeamX0(0)
    , fBeamY0(0)
    , fBeamSigmaX(0)
    , fBeamSigmaY(0)
    , fBeamAngleX0(0)
    , fBeamAngleY0(0)
    , fBeamAngleX(0)
    , fBeamAngleY(0)
    , fBeamAngleSigmaX(0)
    , fBeamAngleSigmaY(0)
    , fBeamDirection(TVector3(0., 0., 1.))
    , fPhiMin(0.)
    , fPhiMax(0.)
    , fPhi(0.)
    , fTargetZ(new Double_t[1])
    , fNrTargets(1)
    , fTargetDz(0)
    , fVertex(TVector3(0., 0., 0.))
    , fNTracks(0)
    , fSmearVertexZ(kFALSE)
    , fSmearGausVertexZ(kFALSE)
    , fSmearVertexXY(kFALSE)
    , fSmearGausVertexXY(kFALSE)
    , fBeamAngle(kFALSE)
    , fEventPlane(kFALSE)
    , fStack(nullptr)
    , fGenList(new TObjArray())
    , fListIter(fGenList->MakeIterator())
    , fEvent(nullptr)
    , fdoTracking(kTRUE)
    , fMCIndexOffset(0)
    , fEventNr(0)
{
    fTargetZ[0] = 0.;
}

FairPrimaryGenerator::FairPrimaryGenerator(const FairPrimaryGenerator &rhs)
    : TNamed(rhs)
    , fBeamX0(rhs.fBeamX0)
    , fBeamY0(rhs.fBeamY0)
    , fBeamSigmaX(rhs.fBeamSigmaX)
    , fBeamSigmaY(rhs.fBeamSigmaY)
    , fBeamAngleX0(rhs.fBeamAngleX0)
    , fBeamAngleY0(rhs.fBeamAngleY0)
    , fBeamAngleX(rhs.fBeamAngleX)
    , fBeamAngleY(rhs.fBeamAngleY)
    , fBeamAngleSigmaX(rhs.fBeamAngleSigmaX)
    , fBeamAngleSigmaY(rhs.fBeamAngleSigmaY)
    , fBeamDirection(rhs.fBeamDirection)
    , fPhiMin(rhs.fPhiMin)
    , fPhiMax(rhs.fPhiMax)
    , fPhi(rhs.fPhi)
    , fTargetZ(new Double_t[1])
    , fNrTargets(rhs.fNrTargets)
    , fTargetDz(rhs.fTargetDz)
    , fVertex(rhs.fVertex)
    , fNTracks(rhs.fNTracks)
    , fSmearVertexZ(rhs.fSmearVertexZ)
    , fSmearGausVertexZ(rhs.fSmearGausVertexZ)
    , fSmearVertexXY(rhs.fSmearVertexXY)
    , fSmearGausVertexXY(rhs.fSmearGausVertexXY)
    , fBeamAngle(rhs.fBeamAngle)
    , fEventPlane(rhs.fEventPlane)
    , fStack(nullptr)
    , fGenList(new TObjArray())
    , fListIter(fGenList->MakeIterator())
    , fEvent(nullptr)
    , fdoTracking(rhs.fdoTracking)
    , fMCIndexOffset(rhs.fMCIndexOffset)
    , fEventNr(rhs.fEventNr)
{
    fTargetZ[0] = rhs.fTargetZ[0];
}

Bool_t FairPrimaryGenerator::Init()
{
    /** Initialize list of generators*/
    for (Int_t i = 0; i < fGenList->GetEntries(); i++) {
        FairGenerator *gen = static_cast<FairGenerator *>(fGenList->At(i));
        if (gen) {
            gen->Init();
        }
    }
    return kTRUE;
}

FairPrimaryGenerator::~FairPrimaryGenerator()
{

    delete[] fTargetZ;
    fGenList->Delete();
    delete fGenList;
    delete fListIter;
}

FairPrimaryGenerator &FairPrimaryGenerator::operator=(const FairPrimaryGenerator &rhs)
{
    // check assignment to self
    if (this != &rhs) {

        // base class assignment
        TNamed::operator=(rhs);

        // assignment operator
        fBeamX0 = rhs.fBeamX0;
        fBeamY0 = rhs.fBeamY0;
        fBeamSigmaX = rhs.fBeamSigmaX;
        fBeamSigmaY = rhs.fBeamSigmaY;
        fBeamAngleX0 = rhs.fBeamAngleX0;
        fBeamAngleY0 = rhs.fBeamAngleY0;
        fBeamAngleX = rhs.fBeamAngleX;
        fBeamAngleY = rhs.fBeamAngleY;
        fBeamAngleSigmaX = rhs.fBeamAngleSigmaX;
        fBeamAngleSigmaY = rhs.fBeamAngleSigmaY;
        fBeamDirection = rhs.fBeamDirection;
        fPhiMin = rhs.fPhiMin;
        fPhiMax = rhs.fPhiMax;
        fPhi = rhs.fPhi;
        fTargetZ = new Double_t[1];
        fNrTargets = rhs.fNrTargets;
        fTargetDz = rhs.fTargetDz;
        fVertex = rhs.fVertex;
        fNTracks = rhs.fNTracks;
        fSmearVertexZ = rhs.fSmearVertexZ;
        fSmearGausVertexZ = rhs.fSmearGausVertexZ;
        fSmearVertexXY = rhs.fSmearVertexXY;
        fSmearGausVertexXY = rhs.fSmearGausVertexXY;
        fBeamAngle = rhs.fBeamAngle;
        fEventPlane = rhs.fEventPlane;
        fStack = nullptr;
        fGenList = new TObjArray();
        fListIter = fGenList->MakeIterator();
        fEvent = nullptr;
        fdoTracking = rhs.fdoTracking;
        fMCIndexOffset = rhs.fMCIndexOffset;
        fEventNr = rhs.fEventNr;
        fTargetZ[0] = rhs.fTargetZ[0];
    }

    return *this;
}

Bool_t FairPrimaryGenerator::GenerateEvent(FairGenericStack *pStack)
{
    // Check for MCEventHeader
    if (!fEvent) {
        LOG(fatal) << "No MCEventHeader branch!";
        return kFALSE;
    } else {
        // Initialise
        fStack = pStack;
        fNTracks = 0;
        fEvent->Reset();

        // Create beam angle
        // Here we only randomly generate two angles (anglex, angley)
        // for the event and later on (in AddTrack())
        // all our particles will be rotated accordingly.
        if (fBeamAngle) {
            MakeBeamAngle();
        }

        // Create event vertex
        MakeVertex();
        fEvent->SetVertex(fVertex);

        // Create event plane
        // Randomly generate an angle by which each track added (in AddTrack())
        // to the event is rotated around the z-axis
        if (fEventPlane) {
            MakeEventPlane();
        }

        // Call the ReadEvent methods from all registered generators
        fListIter->Reset();
        TObject *obj = 0;
        FairGenerator *gen = 0;
        while ((obj = fListIter->Next())) {
            gen = dynamic_cast<FairGenerator *>(obj);
            if (!gen) {
                return kFALSE;
            }
            const char *genName = gen->GetName();
            fMCIndexOffset = fNTracks;   // number tracks before generator is called
            Bool_t test = gen->ReadEvent(this);
            if (!test) {
                LOG(error) << "ReadEvent failed for generator " << genName;
                return kFALSE;
            }
        }

        fTotPrim += fNTracks;
        // Screen output

        // Set the event number if not set already by one of the dedicated generators
        if (-1 == fEvent->GetEventID()) {
            fEventNr++;
            fEvent->SetEventID(fEventNr);
        }
        LOG(debug) << "(Event " << fEvent->GetEventID() << ") " << fNTracks << " primary tracks from vertex ("
                   << fVertex.X() << ", " << fVertex.Y() << ", " << fVertex.Z() << ") with beam angle (" << fBeamAngleX
                   << ", " << fBeamAngleY << ") ";

        fEvent->SetNPrim(fNTracks);

        return kTRUE;
    }
}

void FairPrimaryGenerator::AddTrack(Int_t pdgid,
                                    Double_t px_raw,
                                    Double_t py_raw,
                                    Double_t pz_raw,
                                    Double_t vx,
                                    Double_t vy,
                                    Double_t vz,
                                    Int_t parent,
                                    Bool_t wanttracking,
                                    Double_t e,
                                    Double_t tof,
                                    Double_t weight,
                                    TMCProcess proc)
{
    // ---> Add event vertex to track vertex
    vx += fVertex.X();
    vy += fVertex.Y();
    vz += fVertex.Z();

    // cout << "FairPrimaryGenerator::AddTrack +Z" <<  fVertex.Z() << " pdgid " <<
    // pdgid << "?"<< wanttracking << endl;

    TVector3 mom(px_raw, py_raw, pz_raw);

    if (fEventPlane) {
        // Rotate the track (event) by the randomly generated angle which is fixed
        // for the complete event.
        // The coordinate system is not changed by this rotation.
        mom.RotateZ(fPhi);
    }

    if (fBeamAngle) {
        // Rotate the track (event) from the rotated beam direction system into
        // the fixed global experiment coordinate system
        mom.RotateUz(fBeamDirection.Unit());
    }

    // ---> Convert K0 and AntiK0 into K0s and K0L
    if (pdgid == 311 || pdgid == -311) {
        Double_t test = gRandom->Uniform(0., 1.);
        if (test >= 0.5) {
            pdgid = 310;
        }   // K0S
        else {
            pdgid = 130;
        }   // K0L
    }

    // ---> Check whether particle type is in PDG Database
    TDatabasePDG *pdgBase = TDatabasePDG::Instance();
    if (!pdgBase) {
        Fatal("FairPrimaryGenerator", "No TDatabasePDG instantiated");
    } else {
        TParticlePDG *pdgPart = pdgBase->GetParticle(pdgid);
        if (!pdgPart) {
            if (e < 0) {
                cerr << "\033[5m\033[31m -E FairPrimaryGenerator: PDG code " << pdgid
                     << " not found in database.\033[0m " << endl;
                cerr << "\033[5m\033[31m -E FairPrimaryGenerator: Discarding particle "
                        "\033[0m "
                     << endl;
                cerr << "\033[5m\033[31m -E FairPrimaryGenerator: now MC Index is "
                        "corrupted \033[0m "
                     << endl;
                return;
            } else {
                cout << "\033[5m\033[31m -W FairPrimaryGenerator: PDG code " << pdgid
                     << " not found in database. This warning can be savely "
                        "ignored.\033[0m "
                     << endl;
            }
        }
    }
    // ---> Get mass and calculate energy of particle
    if (e < 0) {
        Double_t mass = pdgBase->GetParticle(pdgid)->Mass();
        e = TMath::Sqrt(mom.Mag2() + mass * mass);
    }   // else, use the value of e given to the function

    // ---> Set all other parameters required by PushTrack
    Int_t doTracking = 0;   // Go to tracking
    if (fdoTracking && wanttracking) {
        doTracking = 1;
    }
    Int_t dummyparent = -1;   // Primary particle (now the value is -1 by default)
    Double_t polx = 0.;       // Polarisation
    Double_t poly = 0.;
    Double_t polz = 0.;
    Int_t ntr = 0;      // Track number; to be filled by the stack
    Int_t status = 0;   // Generation status

    if (parent != -1) {
        parent += fMCIndexOffset;
    }   // correct for tracks which are in list before generator is called
    // Add track to stack
    fStack->PushTrack(doTracking,
                      dummyparent,
                      pdgid,
                      mom.X(),
                      mom.Y(),
                      mom.Z(),
                      e,
                      vx,
                      vy,
                      vz,
                      tof,
                      polx,
                      poly,
                      polz,
                      proc,
                      ntr,
                      weight,
                      status,
                      parent);
    fNTracks++;
}

FairPrimaryGenerator *FairPrimaryGenerator::ClonePrimaryGenerator() const
{
    FairPrimaryGenerator *newPrimaryGenerator = new FairPrimaryGenerator(*this);

    /** Clone generators in the list*/
    for (Int_t i = 0; i < fGenList->GetEntries(); i++) {
        FairGenerator *gen = static_cast<FairGenerator *>(fGenList->At(i));
        if (gen) {
            newPrimaryGenerator->AddGenerator(gen->CloneGenerator());
        }
    }

    return newPrimaryGenerator;
}

void FairPrimaryGenerator::SetBeam(Double_t x0, Double_t y0, Double_t sigmaX, Double_t sigmaY)
{
    fBeamX0 = x0;
    fBeamY0 = y0;
    fBeamSigmaX = sigmaX;
    fBeamSigmaY = sigmaY;
}

void FairPrimaryGenerator::SetBeamAngle(Double_t beamAngleX0,
                                        Double_t beamAngleY0,
                                        Double_t beamAngleSigmaX,
                                        Double_t beamAngleSigmaY)
{
    fBeamAngleX0 = beamAngleX0;
    fBeamAngleY0 = beamAngleY0;
    fBeamAngleSigmaX = beamAngleSigmaX;
    fBeamAngleSigmaY = beamAngleSigmaY;
    fBeamAngle = kTRUE;
}

void FairPrimaryGenerator::SetEventPlane(Double_t phiMin, Double_t phiMax)
{
    fPhiMin = phiMin;
    fPhiMax = phiMax;
    fEventPlane = kTRUE;
}

void FairPrimaryGenerator::SetTarget(Double_t z, Double_t dz)
{

    fTargetZ[0] = z;
    fTargetDz = dz;
}

void FairPrimaryGenerator::SetMultTarget(Int_t nroftargets, Double_t *targetpos, Double_t dz)
{

    delete[] fTargetZ;

    fNrTargets = nroftargets;

    fTargetZ = new Double_t[nroftargets];
    for (Int_t i = 0; i < nroftargets; i++) {
        fTargetZ[i] = targetpos[i];
    }
    fTargetDz = dz;
}

void FairPrimaryGenerator::MakeVertex()
{
    Double_t vx = fBeamX0;
    Double_t vy = fBeamY0;
    Double_t vz;
    if (1 == fNrTargets) {
        vz = fTargetZ[0];
    } else {
        Int_t Target = static_cast<Int_t>(gRandom->Uniform(fNrTargets));
        vz = fTargetZ[Target];
    }

    if (fSmearVertexZ)
        vz = gRandom->Uniform(vz - fTargetDz / 2., vz + fTargetDz / 2.);

    if (fSmearGausVertexZ) {
        vz = gRandom->Gaus(vz, fTargetDz);
    }

    if (fSmearGausVertexXY) {
        if (fBeamSigmaX != 0.) {
            vx = gRandom->Gaus(fBeamX0, fBeamSigmaX);
        }
        if (fBeamSigmaY != 0.) {
            vy = gRandom->Gaus(fBeamY0, fBeamSigmaY);
        }
    }

    if (fSmearVertexXY) {
        if (fBeamSigmaX != 0.) {
            vx = gRandom->Uniform(vx - fBeamSigmaX / 2., vx + fBeamSigmaX / 2.);
        }
        if (fBeamSigmaY != 0.) {
            vy = gRandom->Uniform(vy - fBeamSigmaY / 2., vy + fBeamSigmaY / 2.);
        }
    }

    fVertex = TVector3(vx, vy, vz);
}

void FairPrimaryGenerator::MakeBeamAngle()
{
    fBeamAngleX = gRandom->Gaus(fBeamAngleX0, fBeamAngleSigmaX);
    fBeamAngleY = gRandom->Gaus(fBeamAngleY0, fBeamAngleSigmaY);
    fBeamDirection.SetXYZ(TMath::Tan(fBeamAngleX), TMath::Tan(fBeamAngleY), 1.);
    fEvent->SetRotX(fBeamAngleX);
    fEvent->SetRotY(fBeamAngleY);
}

void FairPrimaryGenerator::MakeEventPlane()
{
    fPhi = gRandom->Uniform(fPhiMin, fPhiMax);
    fEvent->SetRotZ(fPhi);
}

void FairPrimaryGenerator::SmearVertexZ(Bool_t flag)
{
    fSmearVertexZ = flag;
    if (fSmearVertexZ && fSmearGausVertexZ) {
        fSmearGausVertexZ = kFALSE;
    }
}

void FairPrimaryGenerator::SmearGausVertexZ(Bool_t flag)
{
    fSmearGausVertexZ = flag;
    if (fSmearGausVertexZ && fSmearVertexZ) {
        fSmearVertexZ = kFALSE;
    }
}
void FairPrimaryGenerator::SmearVertexXY(Bool_t flag)
{
    fSmearVertexXY = flag;
    if (fSmearVertexXY && fSmearGausVertexXY) {
        fSmearGausVertexXY = kFALSE;
    }
}
void FairPrimaryGenerator::SmearGausVertexXY(Bool_t flag)
{
    fSmearGausVertexXY = flag;
    if (fSmearGausVertexXY && fSmearVertexXY) {
        fSmearVertexXY = kFALSE;
    }
}

ClassImp(FairPrimaryGenerator);
