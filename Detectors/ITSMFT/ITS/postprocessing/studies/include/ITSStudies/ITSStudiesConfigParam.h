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
  // Data parameters
  double b = 5; // Solenoid field in kG (+/-)

  // K0s ID cuts
  std::string targetV0 = "K0"; // target V0; set as "K0" or "Lambda"
  float tgV0window = 0.02;     // half-width of mass window for target V0 mass hypothesis testing (GeV)
  float bgV0window = 0.01;     // half-width of mass window for background V0 mass hypothesis testing (GeV)
  float Rmin = 0.;             // lower limit on V0 decay length (cm?)
  float Rmax = 5.4;            // upper limit on V0 decay length (cm?)
  float cosPAmin = 0.995;      // lower limit on cosine of pointing angle
  float prongDCAmax = 0.2;     // upper limit on DCA between two daughter prongs (cm?)
  float dauPVDCAmin = 0.2;     // lower limit on DCA between prong and primary vertex (cm?)
  float v0PVDCAmax = 0.2;      // upper limit on DCA between V0 and primary vertex (cm?)
  int dauNClusMin = 0;         // lower limit on number of ITS clusters on daughter tracks TODO: not yet implemented

  // Kinematic cut disable flags, false="leave this cut on"; NOTE: may be a better way to implement this with std::bitset<8>
  bool disableCosPA = false;
  bool disableRmin = false;
  bool disableRmax = false;
  bool disableProngDCAmax = false;
  bool disableDauPVDCAmin = false;
  bool disableV0PVDCAmax = false;
  bool disableDauNClusmin = false; // TODO: not yet implemented
  bool disableMassHypoth = true;   // applies to both target and background V0 cuts

  // Plotting options
  bool generatePlots = true;                                        // flag to generate plots
  std::string outFileName = "o2standalone_cluster_size_study.root"; // filename for the ROOT output of this study

  // Average cluster size plot: eta binning parameters
  float etaMin = -1.5; // lower edge of lowest bin for eta binning on average cluster size
  float etaMax = 1.5;  // upper edge for highest bin for eta binning on average cluster size
  int etaNBins = 5;    // number of eta bins

  // Average cluster size plot: cluster size binning parameters
  float sizeMax = 15; // upper edge of highest bin for average cluster size
  int sizeNBins = 20; // number of cluster size bins

  O2ParamDef(ITSAvgClusSizeParamConfig, "ITSAvgClusSizeParam");
};

struct PIDStudyParamConfig : public o2::conf::ConfigurableParamHelper<PIDStudyParamConfig> {
  std::string outFileName = "its_PIDStudy.root";
  // default: average 2023 from C. Sonnabend, Nov 2023: ([0.217553   4.02762    0.00850178 2.33324    0.880904  ])
  // to-do: grab from CCDB when available
  float mBBpars[5] = {0.217553, 4.02762, 0.00850178, 2.33324, 0.880904};
  float mBBres = 0.07; // default: 7% resolution
  O2ParamDef(PIDStudyParamConfig, "PIDStudyParam");
};

struct ITSImpactParameterParamConfig : public o2::conf::ConfigurableParamHelper<ITSImpactParameterParamConfig> {
  std::string outFileName = "its_ImpParameter.root";
  int minNumberOfContributors = 0;
  bool applyTrackCuts = false;
  bool useAllTracks = false;
  bool generatePlots = false;

  O2ParamDef(ITSImpactParameterParamConfig, "ITSImpactParameterParam");
};

} // namespace study
} // namespace its
} // namespace o2

#endif