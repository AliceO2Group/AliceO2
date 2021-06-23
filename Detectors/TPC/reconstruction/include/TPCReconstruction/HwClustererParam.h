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

/// \file HwClustererParam.h
/// \brief Implementation of the parameter class for the hardware clusterer
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_HwClustererParam_H_
#define ALICEO2_TPC_HwClustererParam_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

struct HwClustererParam : public o2::conf::ConfigurableParamHelper<HwClustererParam> {
  unsigned peakChargeThreshold = 2;         ///< Charge threshold for the central peak in ADC counts
  unsigned contributionChargeThreshold = 0; ///< Charge threshold for the contributing pads in ADC counts
  unsigned firstTimeBin = 0;                ///< First time bin to process
  unsigned lastTimeBin = 200000;            ///< Last time bin to process
  short splittingMode = 0;                  ///< cluster splitting mode, 0 no splitting, 1 for minimum contributes half to both, 2 for miminum corresponds to left/older cluster,
  bool isContinuousReadout = true;          ///< Switch for continuous readout
  bool rejectSinglePadClusters = false;     ///< Switch to reject single pad clusters, sigmaPad2Pre == 0
  bool rejectSingleTimeClusters = false;    ///< Switch to reject single time clusters, sigmaTime2Pre == 0
  bool rejectLaterTimebin = false;          ///< Switch to reject peaks in later timebins of the same pad

  O2ParamDef(HwClustererParam, "TPCHwClusterer");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_HwClustererParam_H_
