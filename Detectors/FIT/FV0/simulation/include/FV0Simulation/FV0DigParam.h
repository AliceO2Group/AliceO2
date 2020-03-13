// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FV0_SIMPARAMS_H_
#define O2_FV0_SIMPARAMS_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace fv0
{
// parameters of FV0 digitization / transport simulation

struct FV0DigParam : public o2::conf::ConfigurableParamHelper<FV0DigParam> {
  float intrinsicTimeRes = 0.91;       // time resolution
  float photoCathodeEfficiency = 0.23; // quantum efficiency = nOfPhotoE_emitted_by_photocathode / nIncidentPhotons
  float lightYield = 0.1;              // light collection efficiency to be tuned using collision data
  float pmtGain = 5e4;                 // value for PMT R5924-70 at default FV0 gain
  float pmtTransitTime = 9.5;          // PMT response time (corresponds to 1.9 ns rise time)
  float pmtTransparency = 0.25;        // Transparency of the first dynode of the PMT
  float pmtNbOfSecElec = 9.0;          // Number of secondary electrons emitted from first dynode (per ph.e.)
  float shapeConst = 0.029;            // Crystal ball const parameter
  float shapeMean = 10.2;              // Crystal ball mean  parameter
  float shapeAlpha = -0.34;            // Crystal ball alpha parameter
  float shapeN = 7.6e06;               // Crystal ball N     parameter
  float shapeSigma = 3.013;            // Crystal ball sigma parameter
  float timeShiftCfd = 3.3;            // TODO: adjust after PM design for FV0 is fixed
  int photoelMin = 0;                  // integration lower limit
  int photoelMax = 30;                 // integration upper limit
  float singleMipThreshold = 3.0;      // in [MeV] of deposited energy
  float waveformNbins = 2000;          // number of bins for the analog pulse waveform
  float waveformBinWidth = 0.09765625; // number of bins for the analog (25.0 / 256.0)

  //Optimization-related, derived constants
  float oneOverPmtTransitTime2 = 1.0 / (pmtTransitTime * pmtTransitTime);

  O2ParamDef(FV0DigParam, "FV0DigParam");
};
} // namespace fv0
} // namespace o2

#endif
