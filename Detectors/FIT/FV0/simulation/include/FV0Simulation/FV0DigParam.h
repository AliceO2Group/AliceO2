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

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace fv0
{
// parameters of FV0 digitization / transport simulation

struct FV0DigParam : public o2::conf::ConfigurableParamHelper<FV0DigParam> {
  float intrinsicTimeRes = 0.91;       // time resolution
  float photoCathodeEfficiency = 0.23; // quantum efficiency = nOfPhotoE_emitted_by_photocathode / nIncidentPhotons
  float lightYield = 0.01;             // light collection efficiency to be tuned using collision data [1%]
  float pmtGain = 5e4;                 // value for PMT R5924-70 at default FV0 gain
  float pmtTransitTime = 9.5;          // PMT response time (corresponds to 1.9 ns rise time)
  float pmtTransparency = 0.25;        // Transparency of the first dynode of the PMT
  float shapeConst = 1.18059e-14;      // Crystal ball const parameter
  float shapeMean = 1.70518e+01;       // Crystal ball mean  parameter
  float shapeAlpha = -6.56586e-01;     // Crystal ball alpha parameter
  float shapeN = 2.36408e+00;          // Crystal ball N     parameter
  float shapeSigma = 3.55445;          // Crystal ball sigma parameter
  float timeShiftCfd = 3.3;            // TODO: adjust after PM design for FV0 is fixed
  float singleMipThreshold = 3.0;      // in [MeV] of deposited energy
  float waveformNbins = 10000;         // number of bins for the analog pulse waveform
  float waveformBinWidth = 0.01302;    // number of bins for the analog
  float timeCompensate = 23.25;        // in ns
  float chargeIntBinMin = (timeCompensate - 6.0) / waveformBinWidth;  //Charge integration offset (cfd mean time - 6 ns)
  float chargeIntBinMax = (timeCompensate + 14.0) / waveformBinWidth; //Charge integration offset (cfd mean time + 14 ns)
  bool isIntegrateFull = false;
  float cfdCheckWindow = 2.5; // in ns
  int avgPhElectron = 201;

  //Optimization-related, derived constants
  float oneOverPmtTransitTime2 = 1.0 / (pmtTransitTime * pmtTransitTime);

  O2ParamDef(FV0DigParam, "FV0DigParam");
};
} // namespace fv0
} // namespace o2

#endif
