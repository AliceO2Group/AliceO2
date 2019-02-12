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

#include "MFTTracking/MathUtils.h"
#include "MFTTracking/IndexTableUtils.h"

#include "AliTPCCommonDefGPU.h"

namespace o2
{
namespace MFT
{

struct Cluster final {
  Cluster(const Float_t x, const Float_t y, const Float_t z, const Float_t phi, const Float_t r, const Int_t idx, const Int_t bin)
    : xCoordinate{ x },
      yCoordinate{ y },
      zCoordinate{ z },
      phiCoordinate{ phi },
      rCoordinate{ r },
      clusterId{ idx },
      indexTableBin{ bin } {};
  Cluster(const Float_t x, const Float_t y, const Float_t z, const Int_t index);
  Cluster(const Int_t layerIndex, const Cluster& other);

  Float_t xCoordinate;
  Float_t yCoordinate;
  Float_t zCoordinate;
  Float_t phiCoordinate;
  Float_t rCoordinate;
  Int_t clusterId;
  Int_t indexTableBin;
};

} // namespace MFT
} // namespace o2

#endif /* O2_MFT_CLUSTER_H_ */
