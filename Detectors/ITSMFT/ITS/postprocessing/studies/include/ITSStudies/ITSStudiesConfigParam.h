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

// #ifndef ALICEO2_ITSDPLTRACKINGPARAM_H_
// #define ALICEO2_ITSDPLTRACKINGPARAM_H_

#ifndef O2_AVGCLUSSIZE_STUDY_PARAM_H
#define O2_AVGCLUSSIZE_STUDY_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{
namespace study
{

struct AvgClusSizeStudyParamConfig : public o2::conf::ConfigurableParamHelper<AvgClusSizeStudyParamConfig> {

  // K0s ID cuts
  double Rmin = 0.5;        // lower limit on V0 decay length
  double Rmax = 5.4;        // upper limit on V0 decay length
  double cosPAmin = 0.995;  // lower limit on cosine of pointing angle
  double prongDCAmax = 0.2; // upper limit on DCA between two daughter prongs
  double dauPVDCAmin = 0.2; // lower limit on DCA between prong and primary vertex

  // Plotting options
  bool performFit = false;   // determine if fit to K0s mass spectrum will be done (set to false in the case of low statistics)
  bool generatePlots = true; // TODO: not yet tested

  // Plot parameters
  double etaMin = -1.5; // lower edge of lowest bin for eta binning on average cluster size
  double etaMax = 1.5;  // upper edge for highest bin for eta binning on average cluster size
  int etaNBins = 5;     // number of eta bins

  O2ParamDef(AvgClusSizeStudyParamConfig, "AvgClusSizeStudyParam");
};

} // namespace study
} // namespace its
} // namespace o2
#endif
