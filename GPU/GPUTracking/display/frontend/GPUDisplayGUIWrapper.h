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

namespace o2::gpu
{
struct GPUDisplayGUIChannel;
namespace internal
{
struct GPUDisplayGUIWrapperObjects;
} // namespace internal

class GPUDisplayGUIWrapper
{
 public:
  explicit GPUDisplayGUIWrapper(GPUDisplayGUIChannel& channel); // Starts the GUI thread, which owns the QApplication
  ~GPUDisplayGUIWrapper();
  bool isRunning() const;

  int32_t start();
  int32_t stop();
  int32_t focus(); // Bring the GUI window to the front

 private:
  GPUDisplayGUIChannel& mChannel;
  std::unique_ptr<internal::GPUDisplayGUIWrapperObjects> mO;

  void guiThread();
};
} // namespace o2::gpu
#endif // GPUDISPLAYGUIWRAPPER_H
