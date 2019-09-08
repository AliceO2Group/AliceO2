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
/// \file DeviceStoreVertexerGPU.cu
/// \brief
/// \author matteo.concas@cern.ch

#include "ITStrackingCUDA/DeviceStoreVertexerGPU.h"

namespace o2
{
namespace its
{
namespace GPU
{
DeviceStoreVertexerGPU::initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumber>& clusters)
{
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    this->mClusters[iLayer] =
      Vector<Cluster>{&clusters[iLayer][0], static_cast<int>(clusters[iLayer].size())};
  }
}
} // namespace GPU
} // namespace its
} // namespace o2