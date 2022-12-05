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

enum SaveMode { kBestMatch = 0,
                kSaveAll,
                kSaveTrainingData };

struct GlobalFwdMatchingParam : public o2::conf::ConfigurableParamHelper<GlobalFwdMatchingParam> {

  std::string matchFcn = "matchALL";                      ///< MFT-MCH matching score evaluation
  std::string extMatchFuncFile = "FwdMatchFunc.C";        ///< File name for external input matching function
  std::string extMatchFuncName = "getMatchingFunction()"; ///< Name of external matching function getter
  std::string extCutFuncName = "getCutFunction()";        ///< Name of external cut function getter
  std::string cutFcn = "cutDisabled";                     ///< MFT-MCH candicate cut function
  bool MCMatching = false;                                ///< MFT-MCH matching computed from MCLabels
  double matchPlaneZ = -77.5;                             ///< MFT-MCH matching plane z coordinate
  bool useMIDMatch = false;                               ///< Use input from MCH-MID matching
  Int_t saveMode = kBestMatch;                            ///< Global Forward Tracks save mode
  float MFTRadLength = 0.042;                             ///< MFT thickness in radiation length
  float alignResidual = 1.;                               ///< Alignment residual for cluster position uncertainty

  bool
    isMatchUpstream() const
  {
    return matchFcn.find("matchUpstream") < matchFcn.length();
  }
  bool matchingExternalFunction() const { return matchFcn.find("matchExternal") < matchFcn.length(); }
  bool cutExternalFunction() const { return cutFcn.find("cutExternal") < cutFcn.length(); }

  O2ParamDef(GlobalFwdMatchingParam, "FwdMatching");
};

} // end namespace globaltracking
} // end namespace o2

#endif // ALICEO2_GLOBALFWD_MATCHINGPARAM_H_
