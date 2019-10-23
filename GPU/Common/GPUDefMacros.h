// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDefMacros.h
/// \author David Rohr

// clang-format off
#ifndef GPUDEFMACROS_H
#define GPUDEFMACROS_H

#define GPUCA_M_STRIP_A(...) __VA_ARGS__
#define GPUCA_M_STRIP(X) GPUCA_M_STRIP_A X

#define GPUCA_M_STR_X(a) #a
#define GPUCA_M_STR(a) GPUCA_M_STR_X(a)

#define GPUCA_M_CAT_A(a, b) a ## b
#define GPUCA_M_CAT(...) GPUCA_M_CAT_A(__VA_ARGS__)
#define GPUCA_M_CAT3_A(a, b, c) a ## b ## c
#define GPUCA_M_CAT3(...) GPUCA_M_CAT3_A(__VA_ARGS__)

#define GPUCA_M_FIRST_A(a, ...) a
#define GPUCA_M_FIRST(...) GPUCA_M_FIRST_A(__VA_ARGS__)
#define GPUCA_M_SHIFT_A(a, ...) __VA_ARGS__
#define GPUCA_M_SHIFT(...) GPUCA_M_SHIFT(__VA_ARGS__)

#define GPUCA_M_STRIP_FIRST(a) GPUCA_M_FIRST(GPUCA_M_STRIP(a))

#endif
// clang-format on
