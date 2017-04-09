/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    Pipe  file                               -----
// -----                Created by M. Al-Turany  June 2014             -----
// -------------------------------------------------------------------------

#include "DetectorsPassive/Pipe.h"
#include "TGeoManager.h"   // for TGeoManager, gGeoManager
#include "TGeoMaterial.h"  // for TGeoMaterial
#include "TGeoMedium.h"    // for TGeoMedium
#include "TGeoPcon.h"      // for TGeoPcon
#include "TGeoVolume.h"    // for TGeoVolume

using namespace o2::Passive;

Pipe::~Pipe()
= default;
Pipe::Pipe()
  : FairModule()
{
}

Pipe::Pipe(const char * name, const char * title)
  : FairModule(name ,title)
{
}

Pipe::Pipe(const Pipe& rhs)
  = default;

Pipe& Pipe::operator=(const Pipe& rhs)
{
  // self assignment
  if (this == &rhs) return *this;

  // base class assignment
  FairModule::operator=(rhs);

  return *this;
}

// -----  ConstructGeometry  --------------------------------------------------
void Pipe::ConstructGeometry()
{
     TGeoVolume *top=gGeoManager->GetTopVolume();

    // define some materials
     auto *matCarbon    = new TGeoMaterial("C", 12.011, 6.0, 2.265);
     auto *matVacuum    = new TGeoMaterial("Vacuum", 0, 0, 0);

    // define some media
     auto *Carbon     = new TGeoMedium("C", 3, matCarbon);
     auto *Vacuum     = new TGeoMedium("Vacuum", 4, matVacuum);


    Int_t nSects=2;
    Double_t z[] = { -100, 300};    // in cm
    Double_t r[] = { 2.5, 2.5};    // in cm
    Double_t Thickness = 0.05;     // thickness of beam pipe [cm]
    auto* shape = new TGeoPcon(0., 360., nSects);
    for (Int_t iSect = 0; iSect < nSects; iSect++) {
        shape->DefineSection(iSect, z[iSect], r[iSect], r[iSect]+Thickness);
    }

    // ---> Volume
    auto* pipe = new TGeoVolume("Pipe", shape, Carbon);

    // --Now create the same but diameter less by Thikness and vacuum instead of Carbon
    auto* Vshape = new TGeoPcon(0., 360., nSects);
    for (Int_t iSect = 0; iSect < nSects; iSect++) {
        Vshape->DefineSection(iSect, z[iSect], r[iSect], r[iSect]);
    }

    // ---> Volume
    auto* Vpipe = new TGeoVolume("Pipe", shape, Vacuum);

    top->AddNode(pipe, 1);
    top->AddNode(Vpipe, 1);


}

// ----------------------------------------------------------------------------
FairModule* Pipe::CloneModule() const
{
  return new Pipe(*this);
}

ClassImp(o2::Passive::Pipe)
