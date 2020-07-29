// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   ParamBase.h
/// \author Nicolo' Jacazio
///
/// Set of utilities to handle the parametrization of the PID response for each detector
/// These are the basic storage elements to be kept in the CCDB
///

#ifndef O2_FRAMEWORK_PARAMBASE_H_
#define O2_FRAMEWORK_PARAMBASE_H_

// ROOT includes
#include "Rtypes.h"

namespace o2::pid
{

/// \brief Class to handle the parameters of a given detector response
template <typename T, unsigned int size>
class Parameters
{
 public:
  Parameters() = default;
  ~Parameters() = default;

  /// Setter for parametrization parameters
  void Set(unsigned int param_index, T value) { param_index < size ? mPar[param_index] = value : 0.f; }
  /// Setter for parametrization parameters
  void Set(T const value[size])
  {
    for (unsigned int i = 0; i < size; i++)
      Set(i, value[i]);
  }
  /// Getter for parametrization parameters
  T Get(unsigned int param_index) const { return param_index < size ? mPar[param_index] : -999.f; }
  /// Getter for parametrization parameters
  std::array<T, size> Get() const { return mPar; }

 private:
  /// parameters
  std::array<T, size> mPar;
};

/// \brief Class to handle the parameters and the parametrization of a given detector response
template <typename T, unsigned int size, T (*functional_form)(T, const std::array<T, size>)>
class Parametrization
{
 public:
  Parametrization() = default;
  ~Parametrization() = default;

  /// Getter for parametrization values
  T GetValue(T x) const { return functional_form(x, mParameters.Get()); }

  /// Parameters of the parametrization
  Parameters<T, size> mParameters = Parameters<T, size>();
};

} // namespace o2::pid

#endif // O2_FRAMEWORK_PARAMBASE_H_
