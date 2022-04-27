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

/// \file qlibload.h
/// \author David Rohr

#ifndef QLIBLOAD_H
#define QLIBLOAD_H

#ifdef _WIN32
#define LIBRARY_EXTENSION ".dll"
#define LIBRARY_TYPE HMODULE
#define LIBRARY_LOAD(name) LoadLibraryEx(name, nullptr, nullptr)
#define LIBRARY_CLOSE FreeLibrary
#define LIBRARY_FUNCTION GetProcAddress
#else
#define LIBRARY_EXTENSION ".so"
#define LIBRARY_TYPE void*
#define LIBRARY_LOAD(name) dlopen(name, RTLD_NOW)
#define LIBRARY_CLOSE dlclose
#define LIBRARY_FUNCTION dlsym
#endif

#if defined(GPUCA_ALIROOT_LIB)
#define LIBRARY_PREFIX "Ali"
#elif defined(GPUCA_O2_LIB)
#define LIBRARY_PREFIX "O2"
#else
#define LIBRARY_PREFIX ""
#endif

#endif
