// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012
/// \author rafael.pezzi@cern.ch - adapted to ALICE 3 EndCaps 14/02/2021

// ATTENTION: In opposite to old AliITSgeomTGeo, all indices start from 0, not from 1!!!

#include "FT3Base/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"
#include "MathUtils/Cartesian.h"

#include "FairLogger.h" // for LOG

#include <TGeoBBox.h>         // for TGeoBBox
#include <TGeoManager.h>      // for gGeoManager, TGeoManager
#include <TGeoPhysicalNode.h> // for TGeoPNEntry, TGeoPhysicalNode
#include <TGeoShape.h>        // for TGeoShape
#include <TMath.h>            // for Nint, ATan2, RadToDeg
#include <TString.h>          // for TString, Form
#include "TClass.h"           // for TClass
#include "TGeoMatrix.h"       // for TGeoHMatrix
#include "TGeoNode.h"         // for TGeoNode, TGeoNodeMatrix
#include "TGeoVolume.h"       // for TGeoVolume
#include "TMathBase.h"        // for Max
#include "TObjArray.h"        // for TObjArray
#include "TObject.h"          // for TObject

#include <cctype>  // for isdigit
#include <cstdio>  // for snprintf, NULL, printf
#include <cstring> // for strstr, strlen

using namespace TMath;
using namespace o2::ft3;
using namespace o2::detectors;

ClassImp(o2::ft3::GeometryTGeo);

std::unique_ptr<o2::ft3::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName = "FT3V";      ///< Mother volume name
std::string GeometryTGeo::sLayerName = "FT3Layer";   ///< Layer name
std::string GeometryTGeo::sChipName = "FT3Chip";     ///< Sensor name
std::string GeometryTGeo::sSensorName = "FT3Sensor"; ///< Sensor name

//__________________________________________________________________________
GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : o2::itsmft::GeometryTGeo(DetID::FT3)
{
  // default c-tor, if build is true, the structures will be filled and the transform matrices
  // will be cached
  if (sInstance) {
    LOG(FATAL) << "Invalid use of public constructor: o2::ft3::GeometryTGeo instance exists";
    // throw std::runtime_error("Invalid use of public constructor: o2::ft3::GeometryTGeo instance exists");
  }

  if (build) {
    Build(loadTrans);
  }
}

//__________________________________________________________________________
void GeometryTGeo::Build(int loadTrans)
{
  if (isBuilt()) {
    LOG(WARNING) << "Already built";
    return; // already initialized
  }

  if (!gGeoManager) {
    // RSTODO: in future there will be a method to load matrices from the CDB
    LOG(FATAL) << "Geometry is not loaded";
  }

  fillMatrixCache(loadTrans);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameLayer(Int_t d, Int_t lr)
{
  return Form("%s/%s%d", composeSymNameFT3(d), getFT3LayerPattern(), lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameChip(Int_t d, Int_t lr)
{
  return Form("%s/%s%d", composeSymNameLayer(d, lr), getFT3ChipPattern(), lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameSensor(Int_t d, Int_t lr)
{
  return Form("%s/%s%d", composeSymNameChip(d, lr), getFT3SensorPattern(), lr);
}

//__________________________________________________________________________
void GeometryTGeo::fillMatrixCache(int mask)
{
  // populate matrix cache for requested transformations
  //
}
