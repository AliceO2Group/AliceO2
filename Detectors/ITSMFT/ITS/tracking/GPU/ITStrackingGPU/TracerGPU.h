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
///
#include "ITStracking/Definitions.h"

#if defined(__CUDACC__) && defined(__USE_GPU_TRACER__)
namespace o2
{
namespace its
{
namespace gpu
{
class Tracer
{
 public:
  Tracer(const char* name, int color_id = 0);
  ~Tracer();
};
} // namespace gpu
} // namespace its
} // namespace o2
#define RANGE(name, cid) o2::its::gpu::Tracer tracer(name, cid);
#else
#define RANGE(name, cid)
#endif