// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendWindows.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDWINDOWS_H
#define GPUDISPLAYBACKENDWINDOWS_H

#include "GPUDisplayBackend.h"

class GPUDisplayBackendWindows : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendWindows() = default;
  virtual ~GPUDisplayBackendWindows() = default;

  virtual int StartDisplay() override;
  virtual void DisplayExit() override;
  virtual void SwitchFullscreen(bool set) override;
  virtual void ToggleMaximized(bool set) override;
  virtual void SetVSync(bool enable) override;
  virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

 private:
  virtual int OpenGLMain();
};

#endif
