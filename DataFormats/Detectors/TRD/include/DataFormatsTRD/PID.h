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

/// @file  PID.h
/// @author Felix Schlepper

#ifndef TRD_PID_H
#define TRD_PID_H

#include <array>
#include <unordered_map>
#include <string>

namespace o2
{
namespace trd
{

/// Option for available PID policies.
enum class PIDPolicy : unsigned int {
  // Classical Algorithms
  LQ1D = 0, ///< 1-Dimensional Likelihood model
  LQ3D,     ///< 3-Dimensional Likelihood model

  // ML models
  XGB, ///< XGBOOST

  // Do not add anything after this!
  NMODELS,         ///< Count of all models
  Test,            ///< Load object for testing
  Dummy,           ///< Dummy object outputting -1.f
  DEFAULT = Dummy, ///< The default option
};

/// Transform PID policy from string to enum.
static const std::unordered_map<std::string, PIDPolicy> PIDPolicyString{
  // Classical Algorithms
  {"LQ1D", PIDPolicy::LQ1D},
  {"LQ3D", PIDPolicy::LQ3D},

  // ML models
  {"XGB", PIDPolicy::XGB},

  // General
  {"TEST", PIDPolicy::Test},
  {"DUMMY", PIDPolicy::Dummy},
  // Default
  {"default", PIDPolicy::DEFAULT},
};

/// Transform PID policy from string to enum.
static const char* PIDPolicyEnum[] = {
  "LQ1D",
  "LQ3D",
  "XGBoost",
  "NMODELS",
  "Test",
  "Dummy",
  "default(=TODO)"};

using PIDValue = float;

} // namespace trd
} // namespace o2

#endif
