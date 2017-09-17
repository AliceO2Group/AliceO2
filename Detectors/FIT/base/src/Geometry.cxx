// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iomanip>

#include <TGeoBBox.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TList.h>
#include "FITBase/Geometry.h"

#include <FairLogger.h>

ClassImp(o2::FIT::Geometry);

using namespace o2::FIT;

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;
const Char_t* Geometry::sDefaultGeometryName = "fit";

Geometry::Geometry(const Geometry& geo)
  : TNamed(geo)
{
  
}
Geometry::Geometry(const Text_t* name, const Text_t* title, const Text_t* mcname, const Text_t* mctitle)
  : TNamed(name, title)
{
 
}
Geometry& Geometry::operator=(const Geometry& /*rvalue*/)
{
  // LOG(FATAL) << "assignment operator, not implemented\n";
  return *this;
}
Geometry::~Geometry()
{
  if (this == sGeom) {
    // LOG(ERROR) << "Do not call delete on me\n";
    return;
  }
}
Geometry* Geometry::GetInstance()
{
  Geometry* rv = static_cast<Geometry*>(sGeom);
  return rv;
}
