// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionIncludes.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONINCLUDES_H
#define GPURECONSTRUCTIONINCLUDES_H

// Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#ifndef WIN32
#include <sys/syscall.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sched.h>
#endif

#include "GPUDef.h"
#include "GPULogging.h"

#include <iostream>
#include <fstream>

#if defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPULIBRARY)
#include "AliHLTDefinitions.h"
#include "AliHLTSystem.h"
#endif

#include "GPUReconstructionIncludesITS.h"

#define RANDOM_ERROR
//#define RANDOM_ERROR || rand() % 500 == 1

#endif
