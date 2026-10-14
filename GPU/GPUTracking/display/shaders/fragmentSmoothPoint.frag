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

/// \file fragmentSmoothPoint.frag
/// \author David Rohr

#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 1) flat in float pointSize;
#ifdef VULKAN
layout (push_constant) uniform pushColor { vec4 color; } pc;
#define COLOR pc.color
#else
layout (location = 4) uniform vec4 color;
#define COLOR color
#endif

void main() // Round point with antialiased edge, replaces GL_POINT_SMOOTH. The point size is increased by SMOOTH_POINT_EXTRA_SIZE, the outermost pixel is the alpha ramp
{
  vec2 p = gl_PointCoord - vec2(0.5f);
  float alpha = clamp(pointSize * (0.5f - 0.5f * pointSize * dot(p, p)) - 0.25f, 0.f, 1.f); // radius - (d^2 + 1) / 2 in pixels: sqrt(d^2) linearized around d = 1 pixel, avoids the sqrt, adding 0.25 margin to make it a bit brighter
  if (alpha <= 0.f) {
    discard; // No depth write outside the disc
  }
  outColor = vec4(COLOR.x, COLOR.y, COLOR.z, alpha);
}
