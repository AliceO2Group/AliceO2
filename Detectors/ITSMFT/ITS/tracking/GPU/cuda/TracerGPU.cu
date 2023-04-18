// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITStrackingGPU/TracerGPU.h"

#if !defined(__HIPCC__) && defined(__USE_GPU_TRACER__)
#include "nvToolsExt.h"

constexpr uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
constexpr int num_colors = sizeof(colors) / sizeof(uint32_t);

namespace o2
{
namespace its
{
namespace gpu
{
Tracer::Tracer(const char* name, int color_id)
{
  color_id = color_id % num_colors;
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = colors[color_id];
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name;
  nvtxRangePushEx(&eventAttrib);
}

Tracer::~Tracer()
{
  nvtxRangePop();
}

} // namespace gpu
} // namespace its
} // namespace o2
#endif