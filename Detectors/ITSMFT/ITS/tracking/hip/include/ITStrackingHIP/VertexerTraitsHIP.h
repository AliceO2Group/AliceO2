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
/// \file VertexerTraitsHIP.h
/// \brief
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_VERTEXERTRAITS_HIP_H_
#define O2_ITS_TRACKING_VERTEXERTRAITS_HIP_H_

#include <vector>
#include <array>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingHIP/DeviceStoreVertexerHIP.h"
#include "ITStrackingHIP/UniquePointerHIP.h"

#ifdef _ALLOW_DEBUG_TREES_ITS_
#include "ITStracking/StandaloneDebugger.h"
#endif

namespace o2
{
namespace its
{
class ROframe;

using constants::its2::InversePhiBinSize;

class VertexerTraitsHIP : public VertexerTraits
{
 public:
#ifdef _ALLOW_DEBUG_TREES_ITS_
  VertexerTraitsHIP();
  ~VertexerTraitsHIP();
#else
  VertexerTraitsHIP();
  ~VertexerTraitsHIP() = default;
#endif
  void initialise(ROframe*) override;
  void computeTracklets() override;
  void computeTrackletMatching() override;
  void computeVertices() override;
#ifdef _ALLOW_DEBUG_TREES_ITS_
  void computeMCFiltering() override;
#endif

  // GPU-specific getters
  GPUd() static const int2 getBinsPhiRectWindow(const Cluster&, float maxdeltaphi);
  GPUhd() gpu::DeviceStoreVertexerHIP& getDeviceContext();
  GPUhd() gpu::DeviceStoreVertexerHIP* getDeviceContextPtr();

 protected:
  // #ifdef _ALLOW_DEBUG_TREES_ITS_
  //   StandaloneDebugger* mDebugger;
  // #endif
  gpu::DeviceStoreVertexerHIP mStoreVertexerGPU;
  gpu::UniquePointer<gpu::DeviceStoreVertexerHIP> mStoreVertexerGPUPtr;
};

GPUdi() const int2 VertexerTraitsHIP::getBinsPhiRectWindow(const Cluster& currentCluster, float phiCut)
{
  // This function returns the lowest PhiBin and the number of phi bins to be spanned, In the form int2{phiBinLow, PhiBinSpan}
  const int phiBinMin{constants::its2::getPhiBinIndex(
    math_utils::getNormalizedPhiCoordinate(currentCluster.phiCoordinate - phiCut))};
  const int phiBinSpan{static_cast<int>(MATH_CEIL(phiCut * InversePhiBinSize))};
  return int2{phiBinMin, phiBinSpan};
}

GPUhdi() gpu::DeviceStoreVertexerHIP& VertexerTraitsHIP::getDeviceContext()
{
  return *mStoreVertexerGPUPtr;
}

GPUhdi() gpu::DeviceStoreVertexerHIP* VertexerTraitsHIP::getDeviceContextPtr()
{
  return mStoreVertexerGPUPtr.get();
}

extern "C" VertexerTraits* createVertexerTraitsHIP();

} // namespace its
} // namespace o2
#endif /* O2_ITS_TRACKING_VERTEXERTRAITS_HIP_H_ */
