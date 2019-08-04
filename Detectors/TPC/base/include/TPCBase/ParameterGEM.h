// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGEM.h
/// \brief Definition of the parameter class for the GEM stack
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

// Remark: This file has been modified by Viktor Ratza in order to
// implement the electron efficiency models for the collection and the
// extraction efficiency.

#ifndef ALICEO2_TPC_ParameterGEM_H_
#define ALICEO2_TPC_ParameterGEM_H_

#include <array>
#include <cmath>
#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

enum class AmplificationMode : char {
  FullMode = 0,      ///< Full 4-GEM simulation of all efficiencies etc.
  EffectiveMode = 1, ///< Effective amplification mode using one polya distribution only
};

struct ParameterGEM : public o2::conf::ConfigurableParamHelper<ParameterGEM> {

  /// Get the effective gain of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Effective gain of a given GEM in the stack
  float getEffectiveGain(int gem) const
  {
    return CollectionEfficiency[gem] * AbsoluteGain[gem] * ExtractionEfficiency[gem];
  }

  /// \todo Remove hard-coded number of GEMs in a stack
  /// \todo O2ParamDef takes no std::vectors/arrays, therefore the c-style
  int Geometry[4] = {0, 2, 2, 0};                                ///< GEM geometry (0 standard, 1 medium, 2 large)
  float Distance[5] = {4.f, 0.2f, 0.2f, 0.2f, 0.2f};             ///< Distances between cathode/anode and stages (in cm)
  float Potential[4] = {270.f, 250.f, 270.f, 340.f};             ///< Potential (in Volts)
  float ElectricField[5] = {0.4f, 4.f, 2.f, 0.1f, 4.f};          ///< Electric field configuration (in kV/cm)
  float AbsoluteGain[4] = {14.f, 8.f, 53.f, 240.f};              ///< Absolute gain
  float CollectionEfficiency[4] = {1.f, 0.2f, 0.25f, 1.f};       ///< Collection efficiency
  float ExtractionEfficiency[4] = {0.65f, 0.55f, 0.12f, 0.6f};   ///< Extraction efficiency
  float TotalGainStack = 1644.f;                                 ///< Total gain of the stack for the EffectiveMode
  float KappaStack = 1.2295f;                                    ///< Variable steering the energy resolution of the full stack for the EffectiveMode
  float EfficiencyStack = 0.473805f;                             ///< Variable steering the single electron efficiency of the full stack for the EffectiveMode
  AmplificationMode AmplMode = AmplificationMode::EffectiveMode; ///< Amplification mode [FullMode / EffectiveMode]

  O2ParamDef(ParameterGEM, "TPCGEMParam");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterGEM_H_
