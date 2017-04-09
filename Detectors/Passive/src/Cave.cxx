
/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    Cave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------
#include "DetectorsPassive/Cave.h"
#include "FairGeoInterface.h"  // for FairGeoInterface
#include "FairGeoLoader.h"     // for FairGeoLoader
#include "include/DetectorsPassive/GeoCave.h"
#include "TString.h"           // for TString
#include <cstddef>            // for NULL

using namespace o2::Passive;

ClassImp(o2::Passive::Cave)



void Cave::ConstructGeometry()
{
  FairGeoLoader* loader=FairGeoLoader::Instance();
  FairGeoInterface* GeoInterface =loader->getGeoInterface();
  auto* MGeo=new GeoCave();
  MGeo->setGeomFile(GetGeometryFileName());
  GeoInterface->addGeoModule(MGeo);
  Bool_t rc = GeoInterface->readSet(MGeo);
  if ( rc ) { MGeo->create(loader->getGeoBuilder()); }

}
Cave::Cave()
:FairModule()
{
}

Cave::~Cave()
= default;
Cave::Cave(const char* name,  const char* Title)
  : FairModule(name ,Title)
{
  mWorld[0] = 0;
  mWorld[1] = 0;
  mWorld[2] = 0;
}

Cave::Cave(const Cave& rhs)
  : FairModule(rhs)
{
  mWorld[0] = rhs.mWorld[0];
  mWorld[1] = rhs.mWorld[1];
  mWorld[2] = rhs.mWorld[2];
}

Cave& Cave::operator=(const Cave& rhs)
{
  // self assignment
  if (this == &rhs) return *this;

  // base class assignment
  FairModule::operator=(rhs);

  // assignment operator
  mWorld[0] = rhs.mWorld[0];
  mWorld[1] = rhs.mWorld[1];
  mWorld[2] = rhs.mWorld[2];

  return *this;
}

FairModule* Cave::CloneModule() const
{
  return new Cave(*this);
}
