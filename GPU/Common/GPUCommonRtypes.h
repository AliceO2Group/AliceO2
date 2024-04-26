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

/// \file GPUCommonRtypes.h
/// \author David Rohr

#ifndef GPUCOMMONRTYPES_H
#define GPUCOMMONRTYPES_H

#include "GPUCommonDef.h"

#if defined(GPUCA_STANDALONE) || (defined(GPUCA_O2_LIB) && !defined(GPUCA_O2_INTERFACE)) || defined(GPUCA_GPUCODE) // clang-format off
  #if !defined(ROOT_Rtypes) && !defined(__CLING__)
    #define GPUCOMMONRTYPES_H_ACTIVE
    #define ClassDef(name,id)
    #define ClassDefNV(name, id)
    #define ClassDefOverride(name, id)
    #define ClassImp(name)
    #define templateClassImp(name)
    #ifndef GPUCA_GPUCODE_DEVICE
      typedef unsigned long long int ULong64_t;
      typedef unsigned int UInt_t;
      #include <iostream>
    #endif
  #endif
#else
  #include "Rtypes.h"
#endif // clang-format off

#endif
