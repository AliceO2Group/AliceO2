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

/// \file GPUDisplayInterface.h
/// \author David Rohr

#ifndef GPUDISPLAYINTERFACE_H
#define GPUDISPLAYINTERFACE_H

#include "GPUSettings.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUChainTracking;
class GPUQA;
struct GPUParam;
class GPUDisplayInterface
{
 public:
  GPUDisplayInterface(const GPUDisplayInterface&) = delete;
  virtual ~GPUDisplayInterface();
  virtual int StartDisplay() = 0;
  virtual void ShowNextEvent(const GPUTrackingInOutPointers* ptrs = nullptr) = 0;
  virtual void WaitForNextEvent() = 0;
  virtual void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster) = 0;
  virtual void UpdateCalib(const GPUCalibObjectsConst* calib) = 0;
  virtual void UpdateParam(const GPUParam* param) = 0;
  static GPUDisplayInterface* getDisplay(GPUDisplayFrontendInterface* frontend, GPUChainTracking* chain, GPUQA* qa, const GPUParam* param = nullptr, const GPUCalibObjectsConst* calib = nullptr, const GPUSettingsDisplay* config = nullptr);

 protected:
  GPUDisplayInterface();
};

class GPUDisplayFrontendInterface
{
 public:
  virtual ~GPUDisplayFrontendInterface();
  static GPUDisplayFrontendInterface* getFrontend(const char* type);
  virtual void DisplayExit() = 0;
  virtual bool EnableSendKey() = 0;
  virtual int getDisplayControl() const = 0;
  virtual int getSendKey() const = 0;
  virtual int getNeedUpdate() const = 0;
  virtual void setDisplayControl(int v) = 0;
  virtual void setSendKey(int v) = 0;
  virtual void setNeedUpdate(int v) = 0;
  virtual const char* frontendName() const = 0;

 protected:
  GPUDisplayFrontendInterface();
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUDISPLAYINTERFACE_H
