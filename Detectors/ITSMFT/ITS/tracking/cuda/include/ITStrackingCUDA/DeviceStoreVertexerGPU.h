// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file DeviceStoreVertexer.h
/// \brief
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_
#define O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Tracklet.h"
#include "ITStrackingCUDA/Array.h"
#include "ITStrackingCUDA/UniquePointer.h"
#include "ITStrackingCUDA/Vector.h"

namespace o2
{
namespace its
{
namespace GPU
{
class DeviceStoreVertexerGPU final
{
 public:
  DeviceStoreVertexerGPU() = default;
  UniquePointer<DeviceStoreVertexerGPU> initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumber>&);

 private:
  Array<Vector<Cluster>, constants::its::LayersNumberVertexer> mClsuters;
  Vector<Tracklet> mComb01;
  Vector<Tracklet> mConb12;
}
} // namespace GPU
} // namespace its
} // namespace o2
#endif //O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_
