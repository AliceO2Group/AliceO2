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

/// \file DescriptorInnerBarrelITS3Param.h
/// \brief Definition of the DescriptorInnerBarrelITS3Param class

#ifndef ALICEO2_ITS3_DESCRIPTORINNERBARRELITS3PARAM_H
#define ALICEO2_ITS3_DESCRIPTORINNERBARRELITS3PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace its3
{
/**
 ** a parameter class/struct to keep the settings of
 ** the DescriptorInnerBarrelITS3 class
 ** allow the user to modify them
 **/

struct DescriptorInnerBarrelITS3Param : public o2::conf::ConfigurableParamHelper<DescriptorInnerBarrelITS3Param> {
  std::string mVersion{""};
  O2ParamDef(DescriptorInnerBarrelITS3Param, "DescriptorInnerBarrelITS3");
};

} // namespace its3
} // end namespace o2

#endif // ALICEO2_ITS3_DESCRIPTORINNERBARRELITS3PARAM_H