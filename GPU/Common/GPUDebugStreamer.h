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

/// \file GPUDebugStreamer.h

#ifndef GPUDEBUGSTREAMER_H
#define GPUDEBUGSTREAMER_H

#include "GPUCommonDef.h"
#if defined(GPUCA_HAVE_O2HEADERS) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && defined(DEBUG_STREAMER)
#include "CommonUtils/DebugStreamer.h"
#endif

#endif
