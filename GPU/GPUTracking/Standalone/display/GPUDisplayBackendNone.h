// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendNone.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDNONE_H
#define GPUDISPLAYBACKENDNONE_H

#include "GPUDisplay.h"

class GPUDisplayBackendNone : public GPUDisplayBackend
{
	GPUDisplayBackendNone() = default;
	virtual ~GPUDisplayBackendNone() = default;
	
	virtual int StartDisplay() override {return 1;}
	virtual void DisplayExit() override {}
	virtual void SwitchFullscreen(bool set) override {}
	virtual void ToggleMaximized(bool set) override {}
	virtual void SetVSync(bool enable) override {}
	virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override {}
};
#endif
