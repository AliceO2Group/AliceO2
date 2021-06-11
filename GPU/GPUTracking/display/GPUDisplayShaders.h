// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayShaders.h
/// \author David Rohr

#ifndef GPUDISPLAYSHADERS_H
#define GPUDISPLAYSHADERS_H

#include "GPUCommonDef.h"
namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUDisplayShaders {
  static constexpr const char* vertexShader = R"(
#version 450 core
layout (location = 0) in vec3 pos;
uniform mat4 ModelViewProj;

void main()
{
  gl_Position = ModelViewProj * vec4(pos.x, pos.y, pos.z, 1.0);
}
)";

  static constexpr const char* fragmentShader = R"(
#version 450 core
out vec4 fragColor;
uniform vec3 color;

void main()
{
    fragColor = vec4(color.x, color.y, color.z, 1.f);
}
)";
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
