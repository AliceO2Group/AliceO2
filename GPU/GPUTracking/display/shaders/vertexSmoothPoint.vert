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

/// \file vertexSmoothPoint.vert
/// \author David Rohr

#version 460 core
layout (location = 0) in vec3 pos;
layout (location = 1) flat out float pointSize;
#ifdef VULKAN
layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;
layout (push_constant) uniform pushSize { layout(offset = 16) float size; } ps;
#define MODEL_VIEW_PROJ um.ModelViewProj
#define POINT_SIZE ps.size
#else
layout (location = 0) uniform mat4 ModelViewProj;
layout (location = 5) uniform float size;
#define MODEL_VIEW_PROJ ModelViewProj
#define POINT_SIZE size
#endif

void main()
{
  gl_Position = MODEL_VIEW_PROJ * vec4(pos.x, pos.y, pos.z, 1.0);
  gl_PointSize = POINT_SIZE; // Ignored in OpenGL (no GL_PROGRAM_POINT_SIZE), there glPointSize applies
  pointSize = POINT_SIZE;
}
