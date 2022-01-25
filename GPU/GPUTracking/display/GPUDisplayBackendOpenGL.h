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

/// \file GPUDisplayBackendOpenGL.h
/// \author David Rohr

#ifndef GPUDISPLAYEXT_H
#define GPUDISPLAYEXT_H
#ifdef GPUCA_BUILD_EVENT_DISPLAY

#include "GPUDisplayBackend.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplayBackendOpenGL : public GPUDisplayBackend
{
  virtual int ExtInit();
  virtual bool CoreProfile();
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
