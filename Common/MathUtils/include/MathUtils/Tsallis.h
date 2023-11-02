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

#ifndef ALICEO2_TSALLIS_H
#define ALICEO2_TSALLIS_H

namespace o2
{

namespace math_utils
{

struct Tsallis {
  /// Tsallis/Hagedorn function describing charged pt spectra (m s = 62.4 GeV to 13 TeV) as in https://iopscience.iop.org/article/10.1088/2399-6528/aab00f/pdf
  /// https://github.com/alisw/AliPhysics/blob/523f2dc8b45d913e9b7fda9b27e746819cbe5b09/PWGPP/AliAnalysisTaskFilteredTree.h#L145
  /// \param pt     - transverse momentum
  /// \param mass   - mass of particle
  /// \param sqrts  - centre of mass energy
  /// \return       - invariant yields of the charged particle *pt
  ///    n(sqrts)= a + b/sqrt(s)                             - formula 6
  ///    T(sqrts)= c + d/sqrt(s)                             - formula 7
  ///    a = 6.81 ± 0.06       and b = 59.24 ± 3.53 GeV      - for charged particles page 3
  ///    c = 0.082 ± 0.002 GeV and d = 0.151 ± 0.048 (GeV)   - for charged particles page 4
  static float tsallisCharged(float pt, float mass, float sqrts);

  /// Random downsampling trigger function using Tsallis/Hagedorn spectra fit (sqrt(s) = 62.4 GeV to 13 TeV) as in https://iopscience.iop.org/article/10.1088/2399-6528/aab00f/pdf
  /// \return flat q/pt trigger
  /// \param pt pat of particle
  /// \param factorPt defines the sampling
  /// \param sqrts centre of mass energy
  /// \param weight weight which is internally calculated
  /// \param rnd random value between (0->1) used to check for sampling
  /// \param mass particles mass (use pion if not known)
  static bool downsampleTsallisCharged(float pt, float factorPt, float sqrts, float& weight, float rnd, float mass = 0.13957);
};

} // namespace math_utils
} // namespace o2

#endif
