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
/// \file Cluster.h
/// \brief A simple structure for the MFT cluster, used by the standalone track finder
///

#ifndef O2_MFT_CLUSTER_H_
#define O2_MFT_CLUSTER_H_

#include <array>
#include "ReconstructionDataFormats/BaseCluster.h"
#include "MFTTracking/IndexTableUtils.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace mft
{

struct Cluster : public o2::BaseCluster<float> {
  Cluster(const Float_t x, const Float_t y, const Float_t z, const Float_t phi, const Float_t r, const Int_t idx, const Int_t bin)
    : BaseCluster(1, x, y, z),
      phiCoordinate{phi},
      rCoordinate{r},
      clusterId{idx},
      indexTableBin{bin} {};
  Cluster(const Float_t x, const Float_t y, const Float_t z, const Int_t index);
  Cluster(const Int_t layerIndex, const Cluster& other);

  Float_t phiCoordinate;
  Float_t rCoordinate;
  Int_t clusterId;
  Int_t indexTableBin;
  Float_t sigmaY2;
  Float_t sigmaX2;
};

} // namespace mft
} // namespace o2

#endif /* O2_MFT_CLUSTER_H_ */
