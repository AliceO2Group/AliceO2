// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_PADRESPONSE_H_
#define ALICEO2_TRD_PADRESPONSE_H_

#include <array>
#include "TRDBase/TRDCommonParam.h" // For kNLayer

namespace o2
{
namespace trd
{
class PadResponse
{
 public:
  PadResponse() { samplePRF(); }
  ~PadResponse() = default;
  int getPRF(double, double, int, double*) const; // Get the Pad Response Function (PRF)

 private:
  void samplePRF();                               // Initialized the PRF
  static constexpr int mPRFbin{500};              // Number of bins for the PRF
  static constexpr float mPRFlo{-1.5};            // Lower boundary of the PRF
  static constexpr float mPRFhi{1.5};             // Higher boundary of the PRF
  float mPRFwid;                                  // Bin width of the sampled PRF
  int mPRFpad;                                    // Distance to next pad in PRF
  std::array<float, kNlayer * mPRFbin> mPRFsmp{}; // Sampled pad response
};
} // namespace trd
} // namespace o2

#endif
