// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendWaylandEv.cxx
/// \author David Rohr

// Functions of the Wayland frontend that need the evdev codes from linux/input-event-codes.h.
// That header defines macros KEY_UP, KEY_F1, ..., which replace GPUDisplayFrontend::KEY_* (e.g. KEY_F1 becomes 59 = ';'), so it must not be included in GPUDisplayFrontendWayland.cxx,
// and here only after all other headers, and the functions in this file must not use GPUDisplayFrontend::KEY_*.

#include "GPUDisplayFrontendWayland.h"
#include <linux/input-event-codes.h>

using namespace o2::gpu;

void GPUDisplayFrontendWayland::handlePointerButton(uint32_t button, bool pressed)
{
  if (button == BTN_RIGHT) {
    mMouseDnR = pressed;
  } else if (button == BTN_LEFT) {
    mMouseDn = pressed;
  } else if (button == BTN_MIDDLE) {
    mMouseDnM = pressed;
  }
  if (pressed) {
    mMouseDnX = mMouseMvX;
    mMouseDnY = mMouseMvY;
  }
}
