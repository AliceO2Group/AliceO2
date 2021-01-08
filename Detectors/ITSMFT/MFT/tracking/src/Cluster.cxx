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
/// \file Cluster.cxx
///

#include "MFTTracking/Cluster.h"

#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian.h"

namespace o2
{
namespace mft
{

Cluster::Cluster(const Float_t x, const Float_t y, const Float_t z, const Int_t index)
  : BaseCluster(1, x, y, z),
    phiCoordinate{0.},
    rCoordinate{0.},
    clusterId{index},
    indexTableBin{0}
{
  auto clsPoint2D = math_utils::Point2D<Float_t>(x, y);
  rCoordinate = clsPoint2D.R();
  phiCoordinate = clsPoint2D.Phi();
  o2::math_utils::bringTo02PiGen(phiCoordinate);
}

} // namespace mft
} // namespace o2
