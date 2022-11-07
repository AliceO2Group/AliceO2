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

#include "FairDetector.h"      // for FairDetector
#include <fairlogger/Logger.h> // for LOG, LOG_IF
#include "FairRootManager.h"   // for FairRootManager
#include "FairRun.h"           // for FairRun
#include "FairRuntimeDb.h"     // for FairRuntimeDb
#include "FairVolume.h"        // for FairVolume
#include "FairRootManager.h"

#include "TGeoManager.h"     // for TGeoManager, gGeoManager
#include "TGeoTube.h"        // for TGeoTube
#include "TGeoPcon.h"        // for TGeoPcon
#include "TGeoVolume.h"      // for TGeoVolume, TGeoVolumeAssembly
#include "TString.h"         // for TString, operator+
#include "TVirtualMC.h"      // for gMC, TVirtualMC
#include "TVirtualMCStack.h" // for TVirtualMCStack

#include "ITSBase/DescriptorInnerBarrel.h"

#include <cstdio> // for NULL, snprintf

using namespace o2::its;

/// \cond CLASSIMP
ClassImp(DescriptorInnerBarrel);
/// \endcond

//________________________________________________________________
DescriptorInnerBarrel::DescriptorInnerBarrel() : TObject()
{
  //
  // Default constructor
  //
}

//________________________________________________________________
DescriptorInnerBarrel::DescriptorInnerBarrel(int nlayers) : TObject(), fNumLayers(nlayers)
{
  //
  // Standard constructor
  //
}

//________________________________________________________________
void DescriptorInnerBarrel::GetConfigurationWrapperVolume(double& minradius, double& maxradius, double& zspan)
{
  minradius = fWrapperMinRadius;
  maxradius = fWrapperMaxRadius;
  zspan = fWrapperZSpan;
}

//________________________________________________________________
TGeoTube* DescriptorInnerBarrel::DefineWrapperVolume()
{
  TGeoTube* wrap = new TGeoTube(fWrapperMinRadius, fWrapperMaxRadius, fWrapperZSpan / 2.);
  return wrap;
}