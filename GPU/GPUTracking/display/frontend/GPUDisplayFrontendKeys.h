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
