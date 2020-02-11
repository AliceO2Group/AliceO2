// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterizerParameters.h
/// \brief Definition of the EMCAL clusterizer parameter class
#ifndef ALICEO2_EMCAL_CLUSTERIZERPARAMETERS_H
#define ALICEO2_EMCAL_CLUSTERIZERPARAMETERS_H

#include "Rtypes.h"

namespace o2
{

namespace emcal
{

/// \class ClusterizerParameters
/// \brief Contains all parameters to set up the clusterizer
/// \ingroup EMCALreconstruction
/// \author Rudiger Haake (Yale)
class ClusterizerParameters
{
 public:
  ClusterizerParameters(double timeCut, double timeMin, double timeMax, bool doEnergyGradientCut, double gradientCut, double thresholdSeedE, double thresholdCellE);
  ~ClusterizerParameters() = default;

  double getTimeCut() { return mTimeCut; }
  double getTimeMin() { return mTimeMin; }
  double getTimeMax() { return mTimeMax; }
  double getGradientCut() { return mGradientCut; }
  double getDoEnergyGradientCut() { return mDoEnergyGradientCut; }
  double getThresholdSeedEnergy() { return mThresholdSeedEnergy; }
  double getThresholdCellEnergy() { return mThresholdCellEnergy; }

 private:
  double mTimeCut = 0;               ///<  maximum time difference between the digits inside EMC cluster
  double mTimeMin = 0;               ///<  minimum time of physical signal in a cell/digit
  double mTimeMax = 0;               ///<  maximum time of physical signal in a cell/digit
  double mGradientCut = 0;           ///<  minimum energy difference to distinguish local maxima in a cluster
  bool mDoEnergyGradientCut = false; ///<  cut on energy gradient
  double mThresholdSeedEnergy = 0;   ///<  minimum energy to seed a EC digit in a cluster
  double mThresholdCellEnergy = 0;   ///<  minimum energy for a digit to be a member of a cluster
  ClassDefNV(ClusterizerParameters, 1);
};

} // namespace emcal
} // namespace o2
#endif /* ALICEO2_EMCAL_CLUSTERIZERPARAMETERS_H */
