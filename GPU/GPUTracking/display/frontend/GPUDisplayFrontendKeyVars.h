// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendKeyVars.h
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDKEYVARS_H
#define GPUDISPLAYFRONTENDKEYVARS_H

#include <cstdint>

namespace o2::gpu
{
class GPUDisplayFrontendKeyVars
{
 public:
  // Constant key codes for special mKeys (to unify different treatment in X11 / Windows / GLUT / etc.)
#define GPUDisplayFrontendAddKey(code, key) static constexpr int32_t KEY_##key = code
#define GPUDisplayFrontendKeyDownMap(code, key) static constexpr int32_t KEY_MAP_##key = code
#include "GPUDisplayFrontendKeys.h"
};
} // namespace o2::gpu

#endif
