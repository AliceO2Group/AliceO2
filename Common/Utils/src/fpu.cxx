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

// To enable check for problematic LIBXML version uncomment this
// #define _PROTECT_LIBXML_

#ifdef __linux
#ifdef _PROTECT_LIBXML_
#include <libxml/xmlversion.h>
#if LIBXML_VERSION > 20912
#define _DUMMY_FEE_TRAP_
#endif // LIBXML_VERSION > 20912
#endif // _PROTECT_LIBXML_
#else  // __linux
#define _DUMMY_FEE_TRAP_
#endif // __linux

#ifdef _DUMMY_FEE_TRAP_
void trapfpe() {}
#else // _DUMMY_FEE_TRAP_
#define _GNU_SOURCE 1
#include <cfenv>
#include <cstdlib>
static void __attribute__((constructor)) trapfpe()
{
  // allows to disable set of particular FE's by setting corresponding bit of the O2_DISABLE_FPE_TRAP,
  // i.e. to enable only FE_DIVBYZERO use O2_DISABLE_FPE_TRAP=9
  //  const char* ev = std::getenv("O2_DISABLE_FPE_TRAP");
  //  int enabledFE = (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW) & ~(ev ? atoi(ev) : 0);
  const char* ev = std::getenv("O2_ENABLE_FPE_TRAP");
  int enabledFE = (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW) & (ev ? atoi(ev) : 0);
  if (enabledFE) {
    feenableexcept(enabledFE);
  }
}
#endif // _DUMMY_FEE_TRAP_
