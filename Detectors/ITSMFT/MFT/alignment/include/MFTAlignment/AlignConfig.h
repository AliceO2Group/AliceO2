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

/// \file AlignConfig.h
/// \author arakotoz@cern.ch
/// \brief Configuration file for MFT standalone alignment

#ifndef ALICEO2_MFT_ALIGN_CONFIG_H
#define ALICEO2_MFT_ALIGN_CONFIG_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mft
{

struct AlignConfig : public o2::conf::ConfigurableParamHelper<AlignConfig> {
  int minPoints = 6;                  ///< mininum number of clusters in a track used for alignment
  Int_t chi2CutNStdDev = 3;           ///< Number of standard deviations for chi2 cut
  Double_t residualCutInitial = 100.; ///< Cut on residual on first iteration
  Double_t residualCut = 100.;        ///< Cut on residual for other iterations
  Double_t allowedVarDeltaX = 0.5;    ///< allowed max delta in x-translation (cm)
  Double_t allowedVarDeltaY = 0.5;    ///< allowed max delta in y-translation (cm)
  Double_t allowedVarDeltaZ = 0.5;    ///< allowed max delta in z-translation (cm)
  Double_t allowedVarDeltaRz = 0.01;  ///< allowed max delta in rotation around z-axis (rad)
  Double_t chi2CutFactor = 256.;      ///< used to reject outliers i.e. bad tracks with sum(chi2) > Chi2DoFLim(fNStdDev, nDoF) * fChi2CutFactor

  O2ParamDef(AlignConfig, "MFTAlignment");
};

} // namespace mft
} // namespace o2

#endif
