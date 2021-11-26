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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Class containing parameters common to simulation and reconstruction       //
//                                                                           //
// Request an instance with CommonParam::instance()                          //
// Then request the needed values                                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TGeoGlobalMagField.h>
#include <TMath.h>

#include <FairLogger.h>
#include "TRDBase/CommonParam.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/SimParam.h"

#include "Field/MagneticField.h"

using namespace o2::trd;
using namespace o2::trd::constants;
using namespace o2::trd::garfield;

ClassImp(CommonParam);

CommonParam* CommonParam::mgInstance = nullptr;

//_ singleton implementation __________________________________________________
CommonParam* CommonParam::instance()
{
  //
  // Singleton implementation
  // Returns an instance of this class, it is created if neccessary
  //

  if (mgInstance == nullptr) {
    mgInstance = new CommonParam();
  }

  return mgInstance;
}


//_____________________________________________________________________________
void CommonParam::cacheMagField()
{
  // The magnetic field strength
  const o2::field::MagneticField* fld = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!fld) {
    LOG(fatal) << "Magnetic field is not initialized!";
    return;
  }
  mField = 0.1 * fld->solenoidField(); // kGauss -> Tesla
}




void CommonParam::setXenon()
{
  mGasMixture = kXenon;
  SimParam::instance()->reInit();
}

void CommonParam::setArgon()
{
  mGasMixture = kArgon;
  SimParam::instance()->reInit();
}
