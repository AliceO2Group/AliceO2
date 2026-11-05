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

/// \file vertexSmoothLine.vert
/// \author David Rohr

#version 460 core
layout (location = 0) in vec3 pos;
layout (location = 1) noperspective out vec2 centerPos; // Window position, interpolated along the line, i.e. the closest point on the center line for each fragment of a (wide) line
layout (location = 2) flat out float alphaOffset; // Half line width - 0.25, see fragmentSmoothLine.frag
#ifdef VULKAN
layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;
layout (push_constant) uniform pushLine { layout(offset = 20) float lineWidth; layout(offset = 24) vec2 viewport; } pl;
#define MODEL_VIEW_PROJ um.ModelViewProj
#define LINE_WIDTH pl.lineWidth
#define VIEWPORT pl.viewport
#else
layout (location = 0) uniform mat4 ModelViewProj;
layout (location = 6) uniform vec2 viewport;
layout (location = 7) uniform float lineWidth;
#define MODEL_VIEW_PROJ ModelViewProj
#define LINE_WIDTH lineWidth
#define VIEWPORT viewport
#endif

void main()
{
  gl_Position = MODEL_VIEW_PROJ * vec4(pos.x, pos.y, pos.z, 1.0);
  centerPos = (gl_Position.xy / gl_Position.w * 0.5f + 0.5f) * VIEWPORT;
  alphaOffset = LINE_WIDTH * 0.5f - 0.25f;
}
