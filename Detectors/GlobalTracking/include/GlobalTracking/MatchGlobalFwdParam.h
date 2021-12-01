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

#ifndef ALICEO2_GLOBALFWD_MATCHINGPARAM_H_
#define ALICEO2_GLOBALFWD_MATCHINGPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace globaltracking
{

// **
// ** Parameters for Global forward matching
// **
struct GlobalFwdMatchingParam : public o2::conf::ConfigurableParamHelper<GlobalFwdMatchingParam> {
  bool useMIDMatch = false;
  std::string matchFcn = "matchALL";  ///< MFT-MCH matching score evaluation
  std::string cutFcn = "cutDisabled"; ///< MFT-MCH candicate cut function
  double matchPlaneZ = -77.5;         ///< MFT-MCH matching plane z coordinate

  bool isMatchUpstream() const { return matchFcn.find("matchUpstream") < matchFcn.length(); }
  bool saveAll = false; ///< Option to save all MFTMCH pair combinations.

  O2ParamDef(GlobalFwdMatchingParam, "FwdMatching");
};

} // end namespace globaltracking
} // end namespace o2

#endif // ALICEO2_GLOBALFWD_MATCHINGPARAM_H_
