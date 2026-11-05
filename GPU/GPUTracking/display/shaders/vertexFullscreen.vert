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

/// \file vertexFullscreen.vert
/// \author David Rohr

#version 460 core
#ifdef VULKAN
#define VERTEX_ID gl_VertexIndex
#else
#define VERTEX_ID gl_VertexID
#endif

void main() // Full-screen triangle from the vertex index (0..2), no vertex buffer
{
  vec2 p = vec2((VERTEX_ID << 1) & 2, VERTEX_ID & 2);
  gl_Position = vec4(p * 2.f - 1.f, 0.f, 1.f);
}
