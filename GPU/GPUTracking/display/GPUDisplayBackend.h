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

/// \file GPUDisplayBackend.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKEND_H
#define GPUDISPLAYBACKEND_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUDisplayBackend
{
 public:
  GPUDisplayBackend() = default;
  virtual ~GPUDisplayBackend() = default;

  virtual int ExtInit() = 0;
  virtual bool CoreProfile() = 0;

  static GPUDisplayBackend* getBackend(const char* type);

 protected:
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
