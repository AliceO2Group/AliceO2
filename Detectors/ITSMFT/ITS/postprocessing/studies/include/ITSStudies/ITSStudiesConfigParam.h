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

#ifndef ITS_STUDY_CONFIG_PARAM_H
#define ITS_STUDY_CONFIG_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{
namespace study
{
struct ITSCheckTracksParamConfig : public o2::conf::ConfigurableParamHelper<ITSCheckTracksParamConfig> {
  std::string outFileName = "TrackCheckStudy.root";
  size_t effHistBins = 100;
  unsigned short trackLengthMask = 0x7f;
  float effPtCutLow = 0.01;
  float effPtCutHigh = 10.;

  O2ParamDef(ITSCheckTracksParamConfig, "ITSCheckTracksParam");
};

struct ITSAvgClusSizeParamConfig : public o2::conf::ConfigurableParamHelper<ITSAvgClusSizeParamConfig> {

  // K0s ID cuts
  float Rmin = 0.5;        // lower limit on V0 decay length
  float Rmax = 5.4;        // upper limit on V0 decay length
  float cosPAmin = 0.995;  // lower limit on cosine of pointing angle
  float prongDCAmax = 0.2; // upper limit on DCA between two daughter prongs
  float dauPVDCAmin = 0.2; // lower limit on DCA between prong and primary vertex

  // Plotting options
  bool performFit = false;   // determine if fit to K0s mass spectrum will be done (set to false in the case of low statistics)
  bool generatePlots = true; // TODO: not yet tested

  // Average cluster size plot: eta binning parameters
  float etaMin = -1.5; // lower edge of lowest bin for eta binning on average cluster size
  float etaMax = 1.5;  // upper edge for highest bin for eta binning on average cluster size
  int etaNBins = 5;    // number of eta bins

  // Average cluster size plot: cluster size binning parameters
  float sizeMax = 15; // upper edge of highest bin for average cluster size
  int sizeNBins = 20; // number of cluster size bins

  O2ParamDef(ITSAvgClusSizeParamConfig, "ITSAvgClusSizeParam");
};
} // namespace study
} // namespace its
} // namespace o2

#endif