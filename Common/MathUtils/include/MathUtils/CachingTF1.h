// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief Extension to ROOT::TF1 allowing to cache integral
/// @author Sandro Wenzel, sandro.wenzel@cern.ch

#ifndef ALICEO2_CACHINGTF1_H
#define ALICEO2_CACHINGTF1_H

#include <TF1.h>

namespace o2
{
namespace base
{
class CachingTF1 : public TF1
{
  ///
  /// Class extending TF1 with capability to store expensive
  /// internal caches (integral, etc) when streaming out
  /// This can immensely speed up the construction of the TF1
  /// (e.g., for the purpose of random number generation from arbitrary distributions)
  ///
 public:
  using TF1::TF1;
  ~CachingTF1() override = default;

  // get reading access to fIntegral member
  std::vector<double> const& getIntegralVector() const { return fIntegral; }

 private:
  // in the original TF1 implementation, these members
  // are marked transient; by simply introducing something that
  // points to them they will now be written correctly to disc
  std::vector<double>* mIntegralCache = &fIntegral;
  std::vector<double>* mAlphaCache = &fAlpha;
  std::vector<double>* mBetaCache = &fBeta;
  std::vector<double>* mGammaCache = &fGamma;
  ClassDefOverride(CachingTF1, 1);
};
} // namespace base
} // namespace o2

#endif
