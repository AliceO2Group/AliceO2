/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                         FairMesh source file                  -----
// -----          original author                  D.Bertini           -----
// -----          adapted april 2010               O.Hartmann          -----
// -------------------------------------------------------------------------

#include "FairMesh.h"

#include <TMath.h>     // for Sqrt
#include <TString.h>   // for TString, operator+
#include <iostream>    // for operator<<, basic_ostream, etc

using namespace std;

FairMesh::FairMesh()
    : TObject()
    , fXmin(0.0)
    , fYmin(0.)
    , fZmin(0.)
    , fXmax(0.)
    , fYmax(0.)
    , fZmax(0.)
    , NXbin(0)
    , NYbin(0)
    , NZbin(0)
    , fBinVolume(0.)
    , fDiag(0.)
    , fMeshTid(nullptr)
    , fMeshFlu(nullptr)
    , fMeshSEU(nullptr)
    , fhname("")
{}

FairMesh::FairMesh(const char* fname)
    : TObject()
    , fXmin(0.0)
    , fYmin(0.)
    , fZmin(0.)
    , fXmax(0.)
    , fYmax(0.)
    , fZmax(0.)
    , NXbin(0)
    , NYbin(0)
    , NZbin(0)
    , fBinVolume(0.)
    , fDiag(0.)
    , fMeshTid(nullptr)
    , fMeshFlu(nullptr)
    , fMeshSEU(nullptr)
    , fhname(fname)
{
    //  fhname = fname;
}

void FairMesh::calculate()
{
    // compute numbers

    TString tid = fhname + "TID";
    fMeshTid = new TH2D(tid.Data(), "TID", NXbin, fXmin, fXmax, NYbin, fYmin, fYmax);
    fMeshTid->Sumw2();

    TString flu = fhname + "FLU";
    fMeshFlu = new TH2D(flu.Data(), "Fluence", NXbin, fXmin, fXmax, NYbin, fYmin, fYmax);
    fMeshFlu->Sumw2();

    TString seu = fhname + "SEU";
    fMeshSEU = new TH2D(seu.Data(), "SEU", NXbin, fXmin, fXmax, NYbin, fYmin, fYmax);
    fMeshSEU->Sumw2();

    // reseting
    fMeshTid->Reset();
    fMeshFlu->Reset();
    fMeshSEU->Reset();

    fBinVolume = ((fXmax - fXmin) / NXbin) * ((fYmax - fYmin) / NYbin) * ((fZmax - fZmin) / NZbin);

    fDiag = TMath::Sqrt(((fXmax - fXmin) / NXbin) * ((fXmax - fXmin) / NXbin)
                        + ((fYmax - fYmin) / NYbin) * ((fYmax - fYmin) / NYbin)
                        + ((fZmax - fZmin) / NZbin) * ((fZmax - fZmin) / NZbin));
}

FairMesh::~FairMesh() {}

void FairMesh::print()
{
    cout << " Xmin " << fXmin << " Xmax " << fXmax << endl;
    cout << " Ymin " << fYmin << " Ymax " << fYmax << endl;
    cout << " Zmin " << fZmin << " Zmax " << fZmax << endl;
    cout << " NX  " << NXbin << " NY " << NXbin << endl;
}

ClassImp(FairMesh);
