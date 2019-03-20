// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFTDPLDIGITWRITERPARAM_H_
#define ALICEO2_ITSMFTDPLDIGITWRITERPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"
#include <string_view>

namespace o2
{
namespace ITSMFT
{
template <int N>
struct DPLDigitWriterParam : public o2::conf::ConfigurableParamHelper<DPLDigitWriterParam<N>> {
  static constexpr std::string_view DefFileName[2] = { "itsdigits.root", "mftdigits.root" };
  static constexpr std::string_view ParamName[2] = { "ITSDigitWriterParam", "MFTDigitWriterParam" };

  std::string file = DefFileName[N].data(); // output file name
  std::string treeDigits = "o2sim";         // tree name for digits
  std::string treeROF = "ROF";              // tree name for ROFs
  std::string treeMC2ROF = "MC2ROF";        // tree name for ROFs

 private:
  void sanityCheck()
  {
    static_assert(N == 0 || N == 1, "only 0(ITS) or 1(MFT) are allowed");
  }

  // boilerplate stuff + make principal key
  O2ParamDef(DPLDigitWriterParam, ParamName[N].data());
};

template <int N>
O2ParamImpl(o2::ITSMFT::DPLDigitWriterParam<N>);

} // namespace ITSMFT
} // namespace o2

#endif
