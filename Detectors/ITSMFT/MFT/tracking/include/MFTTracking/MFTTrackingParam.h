// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R. Pezzi - May 2020

#ifndef ALICEO2_MFT_TRACKINGPARAM_H_
#define ALICEO2_MFT_TRACKINGPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mft
{

enum MFTTrackingSeed {
  AB, //
  CE,
  DH
};

enum MFTTrackModel {
  Helix,
  Quadratic,
  Linear
};

// **
// ** Parameters for MFT tracking configuration
// **
struct MFTTrackingParam : public o2::conf::ConfigurableParamHelper<MFTTrackingParam> {
  Int_t seed = MFTTrackingSeed::DH;
  Int_t trackmodel = MFTTrackModel::Helix;
  double sigmaboost = 1e0;
  double seedH_k = 1.0;
  double MFTRadLenghts = 0.041; // MFT average material budget within acceptance
  bool verbose = false;

  O2ParamDef(MFTTrackingParam, "MFTTracking");
};

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_TRACKINGPARAM_H_
