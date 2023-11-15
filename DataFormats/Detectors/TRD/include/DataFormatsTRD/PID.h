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
#include <iostream>

namespace o2
{
namespace trd
{

/// Option for available PID policies.
enum class PIDPolicy : unsigned int {
  // Classical Algorithms
  LQ1D = 0, ///< 1-Dimensional Likelihood model
  LQ2D,     ///< 2-Dimensional Likelihood model
  LQ3D,     ///< 3-Dimensional Likelihood model

#ifdef TRDPID_WITH_ONNX
  // ML models
  XGB, ///< XGBOOST
  PY,  ///< Pytorch
#endif

  // Do not add anything after this!
  NMODELS,         ///< Count of all models
  Dummy,           ///< Dummy object outputting -1.f
  DEFAULT = Dummy, ///< The default option
};

inline std::ostream& operator<<(std::ostream& os, const PIDPolicy& policy)
{
  std::string name;
  switch (policy) {
    case PIDPolicy::LQ1D:
      name = "LQ1D";
      break;
    case PIDPolicy::LQ2D:
      name = "LQ2D";
      break;
    case PIDPolicy::LQ3D:
      name = "LQ3D";
      break;
#ifdef TRDPID_WITH_ONNX
    case PIDPolicy::XGB:
      name = "XGBoost";
      break;
    case PIDPolicy::PY:
      name = "PyTorch";
      break;
#endif
    case PIDPolicy::Dummy:
      name = "Dummy";
      break;
    default:
      name = "Default";
  }
  os << name;
  return os;
}

/// Transform PID policy from string to enum.
static const std::unordered_map<std::string, PIDPolicy> PIDPolicyString{
  // Classical Algorithms
  {"LQ1D", PIDPolicy::LQ1D},
  {"LQ2D", PIDPolicy::LQ2D},
  {"LQ3D", PIDPolicy::LQ3D},

#ifdef TRDPID_WITH_ONNX
  // ML models
  {"XGB", PIDPolicy::XGB},
  {"PY", PIDPolicy::PY},
#endif

  // General
  {"DUMMY", PIDPolicy::Dummy},
  // Default
  {"default", PIDPolicy::DEFAULT},
};

} // namespace trd
} // namespace o2

#endif
