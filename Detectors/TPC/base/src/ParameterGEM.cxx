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
#include "TPCBase/EfficiencyGEM.h"

using namespace o2::TPC;

ParameterGEM::ParameterGEM()
  : mGeometry(), mPotential(), mElectricField(), mAbsoluteGain(), mCollectionEfficiency(), mExtractionEfficiency()
{
  mGeometry = { { 1, 3, 3, 1 } };
  mPotential = { { 270.f, 250.f, 270.f, 340.f } };
  mElectricField = { { 0.4f, 4.f, 2.f, 0.1f, 4.f } };
  mAbsoluteGain = { { 14.f, 8.f, 53.f, 240.f } };

  // Fixed values for efficiencies
  // mCollectionEfficiency = {{1.f, 0.2f, 0.25f, 1.f}};
  // mExtractionEfficiency = {{0.65f, 0.55f, 0.12f, 0.6f}};

  // Load efficiencies for stack configuration
  EfficiencyGEM EffGEM;

  float ElecFieldAbove;
  float ElecFieldBelow;
  float ElecFieldGEM;
  float EffColl;
  float EffExtr;

  for (int n = 1; n <= 4; ++n) {
    EffGEM.setGeometry(getGeometry(n));

    ElecFieldAbove = getElectricField(n);                                                // in kV/cm
    ElecFieldBelow = getElectricField(n + 1);                                            // in kV/cm
    ElecFieldGEM = (0.001 * getPotential(n)) / (0.0001 * EffGEM.getGeometryThickness()); // in kV/cm

    EffColl = EffGEM.getCollectionEfficiency(ElecFieldAbove / ElecFieldGEM);
    EffExtr = EffGEM.getExtractionEfficiency(ElecFieldBelow / ElecFieldGEM);

    setCollectionEfficiency(EffColl, n);
    setExtractionEfficiency(EffExtr, n);
  }
}
