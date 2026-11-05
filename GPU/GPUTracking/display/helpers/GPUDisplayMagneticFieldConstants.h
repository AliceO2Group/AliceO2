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

/// \file GPUDisplayMagneticFieldConstants.h
/// \author David Rohr
/// \brief Array sizes of the magnetic field parameterization, shared by GPUDisplayMagneticField (C++) and the field shader (GLSL), so only plain #defines

#ifndef GPUDISPLAYMAGNETICFIELDCONSTANTS_H
#define GPUDISPLAYMAGNETICFIELDCONSTANTS_H

#define GPUCA_DISPLAY_FIELD_DIMENSIONS 3
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_Z_SEGMENTS 32
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_P_SEGMENTS 512
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_R_SEGMENTS 4096
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_Z_SEGMENTS 128
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_Y_SEGMENTS 2048
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_X_SEGMENTS 16384
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_PARAMETERIZATIONS 2048
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_ROWS 16384
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_COLUMNS 65536
#define GPUCA_DISPLAY_FIELD_MAX_SOLENOID_COEFFICIENTS 131072
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_PARAMETERIZATIONS 2048
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_ROWS 16384
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_COLUMNS 65536
#define GPUCA_DISPLAY_FIELD_MAX_DIPOLE_COEFFICIENTS 262144
#define GPUCA_DISPLAY_FIELD_MAX_CHEBYSHEV_ORDER 32

#endif
