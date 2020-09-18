// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayExt.h
/// \author David Rohr

#ifndef GPUDISPLAYEXT_H
#define GPUDISPLAYEXT_H
#ifdef GPUCA_BUILD_EVENT_DISPLAY

#include "GPUCommonDef.h"

#if defined(GPUCA_DISPLAY_GL3W) && !defined(GPUCA_DISPLAY_OPENGL_CORE)
#define GPUCA_DISPLAY_OPENGL_CORE
#endif

#ifdef GPUCA_DISPLAY_GL3W
#include "GL/gl3w.h"
#else
#include <GL/glew.h>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#ifdef GPUCA_DISPLAY_GL3W
static int GPUDisplayExtInit()
{
  return gl3wInit();
}
#else
static int GPUDisplayExtInit()
{
  return glewInit();
}
#endif
#ifdef GPUCA_DISPLAY_OPENGL_CORE
static constexpr bool GPUCA_DISPLAY_OPENGL_CORE_FLAGS = true;
#else
static constexpr bool GPUCA_DISPLAY_OPENGL_CORE_FLAGS = false;
#endif
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
