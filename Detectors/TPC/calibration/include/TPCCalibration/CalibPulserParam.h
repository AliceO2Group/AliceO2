// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibPulserParam.h
/// \brief Implementation of the parameter class for the hardware clusterer
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_CalibPulserParam_H_
#define ALICEO2_TPC_CalibPulserParam_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

struct CalibPulserParam : public o2::conf::ConfigurableParamHelper<CalibPulserParam> {
  int NbinsT0{200};     ///< Number of bins for T0 reference histogram
  float XminT0{-2};     ///< xmin   of T0 reference histogram
  float XmaxT0{2};      ///< xmax   of T0 reference histogram
  int NbinsQtot{200};   ///< Number of bins for Qtot reference histogram
  float XminQtot{50};   ///< xmin   of Qtot reference histogram
  float XmaxQtot{700};  ///< xmax   of Qtot reference histogram
  int NbinsWidth{100};  ///< Number of bins for width reference histogram
  float XminWidth{0.1}; ///< xmin   of width reference histogram
  float XmaxWidth{5.1}; ///< xmax   of width reference histogram
  int FirstTimeBin{10}; ///< first time bin used in analysis
  int LastTimeBin{490}; ///< first time bin used in analysis
  int ADCMin{5};        ///< minimum adc value
  int ADCMax{1023};     ///< maximum adc value
  int PeakIntMinus{2};  ///< lower bound from maximum for the peak integration, mean and std dev. calc
  int PeakIntPlus{2};   ///< upper bound from maximum for the peak integration, mean and std dev. calc
  int MinimumQtot{20};  ///< minimal Qtot accepted as pulser signal

  O2ParamDef(CalibPulserParam, "TPCCalibPulser");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_HwClustererParam_H_
