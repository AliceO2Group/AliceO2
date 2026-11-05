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

/// \file fragmentDownsample.frag
/// \author David Rohr

#version 460 core
layout (location = 0) out vec4 outColor;
#ifdef VULKAN
layout (binding = 2) uniform sampler2D tex;
layout (push_constant) uniform pushFactor { int factor; } pf;
#define FACTOR pf.factor
#else
layout (binding = 0) uniform sampler2D tex;
layout (location = 4) uniform int factor;
#define FACTOR factor
#endif

void main() // Exact box filter for FSAA downsampling by an integer factor: average of the factor x factor source pixels of this pixel (a linear blit only uses 2 x 2)
{
  const ivec2 base = ivec2(gl_FragCoord.xy) * FACTOR;
  vec3 sum = vec3(0.f);
  if ((FACTOR & 1) == 0) { // Bilinear sample at the corner of 4 source pixels returns their average
    const vec2 invSize = 1.f / vec2(textureSize(tex, 0));
    for (int y = 1; y < FACTOR; y += 2) {
      for (int x = 1; x < FACTOR; x += 2) {
        sum += texture(tex, vec2(base + ivec2(x, y)) * invSize).rgb;
      }
    }
    sum *= 4.f;
  } else {
    for (int y = 0; y < FACTOR; y++) {
      for (int x = 0; x < FACTOR; x++) {
        sum += texelFetch(tex, base + ivec2(x, y), 0).rgb;
      }
    }
  }
  outColor = vec4(sum / float(FACTOR * FACTOR), 1.f);
}
