// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGEM.cxx
/// \brief Implementation of the parameter class for the GEM stack
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

// Remark: This file has been modified by Viktor Ratza in order to
// implement the efficiency models for the collection and the
// extraction efficiency.

#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ModelGEM.h"

using namespace o2::TPC;

ParameterGEM::ParameterGEM()
  : mGeometry(), mDistance(), mPotential(), mElectricField(), mAbsoluteGain(), mCollectionEfficiency(), mExtractionEfficiency(), mTotalGainStack(1644.f), mKappaStack(1.2295f), mEfficiencyStack(0.473805f), mAmplificationMode(AmplificationMode::EffectiveMode)
{
  mGeometry = { { 0, 2, 2, 0 } };
  mDistance = { { 4.f, 0.2f, 0.2f, 0.2f, 0.2f } };
  mPotential = { { 270.f, 250.f, 270.f, 340.f } };
  mElectricField = { { 0.4f, 4.f, 2.f, 0.1f, 4.f } };

  // Fixed values for absolute gains
  mAbsoluteGain = { { 14.f, 8.f, 53.f, 240.f } };

  // Fixed values for efficiencies
  mCollectionEfficiency = { { 1.f, 0.2f, 0.25f, 1.f } };
  mExtractionEfficiency = { { 0.65f, 0.55f, 0.12f, 0.6f } };

  // Use model calculations in order to calculate electron efficiencies, gain curves oder energy resolution
  // and total effective gain of a predefined stack.
  ModelGEM Model;

  for (int n = 1; n <= 4; ++n) {
    // Calculate electron efficiencies
    // setCollectionEfficiency(Model.getElectronCollectionEfficiency(getElectricField(n), getPotential(n), getGeometry(n)), n);
    // setExtractionEfficiency(Model.getElectronExtractionEfficiency(getElectricField(n + 1), getPotential(n), getGeometry(n)), n);

    // Calculate absolute gains
    // Model.setAttachment(0.0);
    // Model.setAbsGainScalingFactor(1.49); // Scale gain curves for tuning
    // setAbsoluteGain(Model.getAbsoluteGain(getPotential(n), getGeometry(n)), n);

    // Get the energy resolution & total eff. gain of stack
    // Model.setStackProperties(mGeometry, mDistance, mPotential, mElectricField);
    // Model.getStackEnergyResolution();
    // Model.getStackEffectiveGain();
  }
}
