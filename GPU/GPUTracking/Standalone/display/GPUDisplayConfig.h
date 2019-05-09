// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayConfig.h
/// \author David Rohr

#ifndef GPUDISPLAYCONFIG_H
#define GPUDISPLAYCONFIG_H

#include "GPUCommonDef.h"

#if !defined(GPUCA_STANDALONE)
#define QCONFIG_CPP11_INIT
#endif
#include "utils/qconfig.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
typedef structConfigGL GPUDisplayConfig;
}
} // namespace GPUCA_NAMESPACE

#endif
