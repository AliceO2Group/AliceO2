// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonRtypes.h
/// \author David Rohr

#ifndef GPUCOMMONRTYPES_H
#define GPUCOMMONRTYPES_H

#if defined(GPUCA_STANDALONE) || (defined(GPUCA_O2_LIB) && !defined(GPUCA_O2_INTERFACE)) || defined(GPUCA_GPULIBRARY) // clang-format off
  #if !defined(ROOT_Rtypes) && !defined(__CLING__)
    #define ClassDef(name,id)
    #define ClassDefNV(name, id)
    #define ClassDefOverride(name, id)
    #define ClassImp(name)
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
