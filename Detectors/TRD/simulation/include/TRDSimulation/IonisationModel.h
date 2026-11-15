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

#ifndef ALICEO2_TRD_IONISATIONMODEL_H_
#define ALICEO2_TRD_IONISATIONMODEL_H_

#include "TRDSimulation/SimParam.h"

class TRandom;

namespace o2
{
namespace trd
{

/// Primary ionisation in the TRD gas from tabulated collision spectra, independent of the transport engine's
/// energy-loss model. The tables hold the number of collisions per cm with an energy transfer below the delta-ray
/// production cut of the gas (10 keV) and their energy distribution, as functions of beta*gamma, taken from
/// Geant4's photo-absorption ionisation (PAI) cross-section for the Xe/CO2 and Ar/CO2 mixtures.
class IonisationModel
{
 public:
  explicit IonisationModel(SimParam::GasMixture gas);

  /// Energy (eV) lost in collisions along a step of length (cm) at beta*gamma, for a particle of charge squared z2
  double stepEnergy(double length, double betaGamma, double z2, TRandom& rnd) const;

  /// Mean number of collisions per cm for unit charge
  double collisionsPerCm(double betaGamma) const;

  /// Mean energy loss (eV/cm) below the delta-ray cut for unit charge, from the tables
  double restrictedDEdx(double betaGamma) const;

  /// Lowest beta*gamma of the tables
  static double minBetaGamma();

  /// Delta-ray production cut (eV) the tables were made for
  static double deltaRayCut();

 private:
  double sampleCollision(int row, double u) const;

  int mNEnergy = 0;
  const double* mEnergy = nullptr;       ///< energy grid (eV)
  const double* mCollisions = nullptr;   ///< collisions per cm, per beta*gamma row
  const double* mDEdx = nullptr;         ///< restricted dE/dx (eV/cm), per beta*gamma row
  const float* mFractionAbove = nullptr; ///< fraction of collisions above each grid energy, per row
};

} // namespace trd
} // namespace o2

#endif
