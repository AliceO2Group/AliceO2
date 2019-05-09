// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.cxx
/// \brief Implementation of the MCH cluster for internal use
///
/// \author Philippe Pillot, Subatech

#include "Cluster.h"

namespace o2
{
namespace mch
{

//_________________________________________________________________________
Cluster::Cluster(const ClusterStruct& cl) : mX(cl.x), mY(cl.y), mZ(cl.z), mEx(cl.ex), mEy(cl.ey), mUid(cl.uid)
{
  /// Constructor from the simple cluster structure
}

//__________________________________________________________________________
ClusterStruct Cluster::getClusterStruct() const
{
  /// return cluster in the flat structure

  ClusterStruct cluster{};

  cluster.x = mX;
  cluster.y = mY;
  cluster.z = mZ;
  cluster.ex = mEx;
  cluster.ey = mEy;
  cluster.uid = mUid;

  return cluster;
}

} // namespace mch
} // namespace o2
