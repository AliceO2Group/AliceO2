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

/// \file GPUDisplayGUIWrapper.h
/// \author David Rohr

#ifndef GPUDISPLAYGUIWRAPPER_H
#define GPUDISPLAYGUIWRAPPER_H

#include "GPUCommonDef.h"
#include <memory>

namespace GPUCA_NAMESPACE::gpu
{
struct GPUDisplayGUIWrapperObjects;

class GPUDisplayGUIWrapper
{
 public:
  GPUDisplayGUIWrapper();
  ~GPUDisplayGUIWrapper();
  bool isRunning() const;
  void UpdateTimer();

  int start();
  int stop();
  int focus();

 private:
  std::unique_ptr<GPUDisplayGUIWrapperObjects> mO;

  void guiThread();
};
} // namespace GPUCA_NAMESPACE::gpu
#endif // GPUDISPLAYGUIWRAPPER_H
