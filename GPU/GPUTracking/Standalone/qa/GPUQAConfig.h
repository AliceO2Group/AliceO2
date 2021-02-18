// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUQAConfig.h
/// \author David Rohr

#ifndef GPUQACONFIG_H
#define GPUQACONFIG_H

#include "GPUCommonDef.h"

#include "utils/qconfig.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
typedef GPUSettingsQA GPUQAConfig;
extern GPUSettingsStandalone configStandalone;
}
} // namespace GPUCA_NAMESPACE

#endif
