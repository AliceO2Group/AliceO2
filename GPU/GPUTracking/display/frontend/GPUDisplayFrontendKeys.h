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

/// \file GPUDisplayFrontendKeys.h
/// \author David Rohr

// Key codes for special keys (to unify different treatment in X11 / Windows / GLUT / etc.), as GPUDisplayFrontendAddKey(code, name) X-macros.
// No include guard: included multiple times with different definitions of GPUDisplayFrontendAddKey.

GPUDisplayFrontendAddKey(1, UP);
GPUDisplayFrontendAddKey(2, DOWN);
GPUDisplayFrontendAddKey(3, LEFT);
GPUDisplayFrontendAddKey(4, RIGHT);
GPUDisplayFrontendAddKey(5, PAGEUP);
GPUDisplayFrontendAddKey(6, PAGEDOWN);
GPUDisplayFrontendAddKey(8, SHIFT);
GPUDisplayFrontendAddKey(9, ALT);
GPUDisplayFrontendAddKey(10, RALT);
GPUDisplayFrontendAddKey(11, CTRL);
GPUDisplayFrontendAddKey(12, RCTRL);
GPUDisplayFrontendAddKey(13, ENTER); // fixed at 13
GPUDisplayFrontendAddKey(14, F1);
GPUDisplayFrontendAddKey(15, F2);
GPUDisplayFrontendAddKey(16, F3);
GPUDisplayFrontendAddKey(17, F4);
GPUDisplayFrontendAddKey(18, F5);
GPUDisplayFrontendAddKey(19, F6);
GPUDisplayFrontendAddKey(20, F7);
GPUDisplayFrontendAddKey(21, F8);
GPUDisplayFrontendAddKey(22, F9);
GPUDisplayFrontendAddKey(23, F10);
GPUDisplayFrontendAddKey(24, F11);
GPUDisplayFrontendAddKey(25, F12);
GPUDisplayFrontendAddKey(26, INSERT);
GPUDisplayFrontendAddKey(27, ESCAPE); // fixed at 27
GPUDisplayFrontendAddKey(28, HOME);
GPUDisplayFrontendAddKey(29, END);
GPUDisplayFrontendAddKey(32, SPACE); // fixed at 32

#ifndef GPUDisplayFrontendKeyDownMap
#define GPUDisplayFrontendKeyDownMap(...)
#endif

GPUDisplayFrontendKeyDownMap(0, FORWARD);
GPUDisplayFrontendKeyDownMap(1, BACKWARD);
GPUDisplayFrontendKeyDownMap(2, STRAFE_LEFT);
GPUDisplayFrontendKeyDownMap(3, STRAFE_RIGHT);
GPUDisplayFrontendKeyDownMap(4, STRAFE_UP);
GPUDisplayFrontendKeyDownMap(5, STRAFE_DOWN);
GPUDisplayFrontendKeyDownMap(6, ROTATE_LEFT);
GPUDisplayFrontendKeyDownMap(7, ROTATE_RIGHT);
GPUDisplayFrontendKeyDownMap(8, INCREASE_WIDTH);
GPUDisplayFrontendKeyDownMap(9, DECREASE_WIDTH);
GPUDisplayFrontendKeyDownMap(10, FOCUS_ORIGIN);
GPUDisplayFrontendKeyDownMap(11, FOCUS_ORIENT_UP);
GPUDisplayFrontendKeyDownMap(12, ROTATE_MODEL);
GPUDisplayFrontendKeyDownMap(13, ROTATE_AROUND_Z);
GPUDisplayFrontendKeyDownMap(14, CHANGE_ANIMATION_INTERVAL);
GPUDisplayFrontendKeyDownMap(15, ROTATE_YAW_LEFT);
GPUDisplayFrontendKeyDownMap(16, ROTATE_YAW_RIGHT);
GPUDisplayFrontendKeyDownMap(17, ROTATE_PITCH_UP);
GPUDisplayFrontendKeyDownMap(18, ROTATE_PITCH_DOWN);
GPUDisplayFrontendKeyDownMap(19, MAX); // Must be the maximum value!

#undef GPUDisplayFrontendAddKey
#undef GPUDisplayFrontendKeyDownMap
