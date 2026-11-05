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

/// \file fragmentSmoothLine.frag
/// \author David Rohr

#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 1) noperspective in vec2 centerPos;
layout (location = 2) flat in float alphaOffset;
#ifdef VULKAN
layout (push_constant) uniform pushColor { vec4 color; } pc;
#define COLOR pc.color
#else
layout (location = 4) uniform vec4 color;
#define COLOR color
#endif

void main() // Antialiased line edge (1 pixel ramp at the border of the line widened by SMOOTH_LINE_EXTRA_WIDTH), replaces GL_LINE_SMOOTH / Vulkan smoothLines
{
  vec2 p = gl_FragCoord.xy - centerPos;
  float alpha = clamp(alphaOffset - 0.5f * dot(p, p), 0.f, 1.f); // half width - (d^2 + 0.5) / 2 in pixels: sqrt(d^2) linearized around d = 1 pixel (no sqrt), 0.25 margin to make it a bit brighter, as for the points
  if (alpha <= 0.f) {
    discard; // No depth write outside the line
  }
  outColor = vec4(COLOR.x, COLOR.y, COLOR.z, alpha);
}
