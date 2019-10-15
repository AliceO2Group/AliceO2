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
/// \file VertexerTraitsGPU.h
/// \brief
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_VERTEXER_TRAITS_GPU_H_
#define O2_ITS_TRACKING_VERTEXER_TRAITS_GPU_H_

#include <vector>
#include <array>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingCUDA/DeviceStoreVertexerGPU.h"
#include "ITStrackingCUDA/UniquePointer.h"

#ifdef _ALLOW_DEBUG_TREES_ITS_
#include "ITStracking/StandaloneDebugger.h"
#endif

namespace o2
{
namespace its
{
class ROframe;

using constants::index_table::InversePhiBinSize;

class VertexerTraitsGPU : public VertexerTraits
{
 public:
#ifdef _ALLOW_DEBUG_TREES_ITS_
  VertexerTraitsGPU();
  virtual ~VertexerTraitsGPU();
#else
  VertexerTraitsGPU();
  virtual ~VertexerTraitsGPU() = default;
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
  GPUhd() GPU::DeviceStoreVertexerGPU& getDeviceContext();

 protected:
  // #ifdef _ALLOW_DEBUG_TREES_ITS_
  //   StandaloneDebugger* mDebugger;
  // #endif
  GPU::DeviceStoreVertexerGPU mStoreVertexerGPU;
  GPU::UniquePointer<GPU::DeviceStoreVertexerGPU> mStoreVertexerGPUPtr;
};

inline GPUd() const int2 VertexerTraitsGPU::getBinsPhiRectWindow(const Cluster& currentCluster, float phiCut)
{
  // This function returns the lowest PhiBin and the number of phi bins to be spanned, In the form int2{phiBinLow, PhiBinSpan}
  const int phiBinMin{index_table_utils::getPhiBinIndex(
    math_utils::getNormalizedPhiCoordinate(currentCluster.phiCoordinate - phiCut))};
  const int phiBinSpan{static_cast<int>(MATH_CEIL(phiCut * InversePhiBinSize))};
  return int2{phiBinMin, phiBinSpan};
}

inline GPU::DeviceStoreVertexerGPU& VertexerTraitsGPU::getDeviceContext()
{
  return *mStoreVertexerGPUPtr;
}

extern "C" VertexerTraits* createVertexerTraitsGPU();

} // namespace its
} // namespace o2
#endif /* O2_ITS_TRACKING_VERTEXER_TRAITS_GPU_H_ */
