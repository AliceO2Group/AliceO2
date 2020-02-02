// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_QEDGENINFO_H
#define ALICEO2_QEDGENINFO_H

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

//  @file   QEDGenInfo.h
//  @author ruben.shahoyan@cern.ch
//  @brief  Summary of QED generation: cuts and estimated x-section

namespace o2
{
namespace eventgen
{
struct QEDGenParam : public o2::conf::ConfigurableParamHelper<QEDGenParam> {

  float yMin = -6.f;    ///< min Y
  float yMax = 6.f;     ///< max Y
  float ptMin = 0.4e-3; ///< min pT
  float ptMax = 10.f;   ///< min pT
  //
  float xSectionQED = -1; ///< estimated QED x-section in barns
  float xSectionHad = 8.; ///< reference hadronic x-section for the same system

  // boilerplate stuff + make principal key
  O2ParamDef(QEDGenParam, "QEDGenParam");
};

} // namespace eventgen
} // namespace o2

#endif
