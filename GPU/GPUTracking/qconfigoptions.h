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

/// \file qconfigoptions.h
/// \author David Rohr

#ifndef QCONFIG_INTERNAL
#include "GPUSettingsTypes.h"
#endif

// clang-format off
BeginNamespace(o2::gpu)
#if !defined(QCONFIGOPTIONS_H_LIST) || defined(QCONFIG_INTERNAL)
#define QCONFIGOPTIONS_H_LIST
#include "GPUSettingsList.h"
#endif

#if !defined(QCONFIGOPTIONS_H_PARAM) || defined(QCONFIG_INTERNAL)
#define QCONFIGOPTIONS_H_PARAM
#include "GPUSettingsParam.h"
#endif
EndNamespace() // o2::gpu
