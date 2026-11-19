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

/// \file AlmiraSignal.h
/// \brief Charge dependence of the ALMIRA pixel front-end signal: time walk and time over threshold

#ifndef ALICEO2_TRKFT3_ALMIRASIGNAL_H
#define ALICEO2_TRKFT3_ALMIRASIGNAL_H

#include <cmath>
#include <cstdio>

namespace o2
{
namespace trkft3
{

/// @class AlmiraSignal
/// @brief Time walk and time over threshold of the ALMIRA pixel front-end as a function of the collected charge.
/// Both are parametrised as f(Q) = a * (1 - exp(-Q / b)) + c, with Q the collected charge in electrons.
/// The fit parameters are kept in seconds (as in the original fit), the getters return ns.
class AlmiraSignal
{
 public:
  /// time walk: delay of the leading edge wrt the charge arrival, in ns
  float getTimeWalk(float nEle) const { return eval(nEle, TWa, TWb, TWc); }
  /// time over threshold, in ns
  float getToT(float nEle) const { return eval(nEle, ToTa, ToTb, ToTc); }

  void print() const
  {
    printf("  f(Q) = a * (1 - exp(-Q / b)) + c, Q in electrons\n");
    printf("  time walk          : a = %.3e s, b = %.3e e, c = %.3e s\n", TWa, TWb, TWc);
    printf("  time over threshold: a = %.3e s, b = %.3e e, c = %.3e s\n", ToTa, ToTb, ToTc);
  }

 private:
  static float eval(float nEle, float a, float b, float c) { return (a * (1.f - std::exp(-nEle / b)) + c) * kSecToNS; }

  static constexpr float kSecToNS = 1.e9f;

  // parametrization parameters
  static constexpr float TWa = -8.51e-07f;  ///< s (± 2.56e-08)
  static constexpr float TWb = 7.35e+01f;   ///< electrons (± 1.67e+00)
  static constexpr float TWc = 8.94e-07f;   ///< s (± 2.61e-08)
  static constexpr float ToTa = 1.17e-05f;  ///< s (± 2.68e-08)
  static constexpr float ToTb = 9.44e+02f;  ///< electrons (± 6.06e+00)
  static constexpr float ToTc = -3.15e-08f; ///< s (± 1.57e-08)
};

} // namespace trkft3
} // namespace o2

#endif
