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

/// \file GPUSettingsParam.h
/// \author David Rohr

// clang-format off

// Derrived parameters used in GPUParam
BeginHiddenConfig(GPUSettingsParam, param)
AddVariableRTC(continuousTracking, int8_t, 0) // Continuous tracking, estimate bz and errors for abs(z) = 125cm during seeding
AddVariableRTC(dodEdx, int8_t, 0)             // Do dEdx computation
EndConfig()
