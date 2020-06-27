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
/// \file VertexerTraitsEC0GPU.h
/// \brief
/// \author matteo.concas@cern.ch

#ifndef O2_EC0_TRACKING_VERTEXER_TRAITS_GPU_H_
#define O2_EC0_TRACKING_VERTEXER_TRAITS_GPU_H_

#include <vector>
#include <array>

#include "EC0tracking/VertexerTraitsEC0.h"
#include "EC0tracking/Cluster.h"
#include "EC0tracking/Constants.h"
#include "EC0tracking/Definitions.h"
#include "EC0tracking/Tracklet.h"

#include "EC0trackingCUDA/DeviceStoreVertexerGPU.h"
#include "EC0trackingCUDA/UniquePointer.h"

#ifdef _ALLOW_DEBUG_TREES_ITS_
#include "EC0tracking/StandaloneDebugger.h"
#endif

namespace o2
{
namespace ecl
{
class ROframe;

using constants::index_table::InversePhiBinSize;

class VertexerTraitsEC0GPU : public VertexerTraitsEC0
{
 public:
#ifdef _ALLOW_DEBUG_TREES_ITS_
  VertexerTraitsEC0GPU();
  virtual ~VertexerTraitsEC0GPU();
#else
  VertexerTraitsEC0GPU();
  virtual ~VertexerTraitsEC0GPU() = default;
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
  GPU::DeviceStoreVertexerGPU mStoreVertexerGPU;
  GPU::UniquePointer<GPU::DeviceStoreVertexerGPU> mStoreVertexerGPUPtr;
};

inline GPUd() const int2 VertexerTraitsEC0GPU::getBinsPhiRectWindow(const Cluster& currentCluster, float phiCut)
{
  // This function returns the lowest PhiBin and the number of phi bins to be spanned, In the form int2{phiBinLow, PhiBinSpan}
  const int phiBinMin{index_table_utils::getPhiBinIndex(
    math_utils::getNormalizedPhiCoordinate(currentCluster.phiCoordinate - phiCut))};
  const int phiBinSpan{static_cast<int>(MATH_CEIL(phiCut * InversePhiBinSize))};
  return int2{phiBinMin, phiBinSpan};
}

inline GPU::DeviceStoreVertexerGPU& VertexerTraitsEC0GPU::getDeviceContext()
{
  return *mStoreVertexerGPUPtr;
}

extern "C" VertexerTraitsEC0* createVertexerTraitsEC0GPU();

} // namespace ecl
} // namespace o2
#endif /* O2_EC0_TRACKING_VERTEXER_TRAITS_GPU_H_ */
