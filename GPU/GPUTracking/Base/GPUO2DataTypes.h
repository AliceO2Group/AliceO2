// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2DataTypes.h
/// \author David Rohr

#ifndef O2_GPU_GPUO2DATATYPES_H
#define O2_GPU_GPUO2DATATYPES_H

#if defined(HAVE_O2HEADERS) && (!defined(__OPENCL__) || defined(__OPENCLCPP__))
#include "DataFormatsTPC/ClusterNative.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "TRDBase/TRDGeometryFlat.h"
#else
namespace o2
{
namespace tpc
{
struct ClusterNative {
  static float getTime() { return 0.f; }
  static float getPad() { return 0.f; }
  static int getFlags() { return 0; }
  unsigned char qTot, qMax;
};
struct ClusterNativeAccess {
  const ClusterNative* clustersLinear;
  const ClusterNative* clusters[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClusters[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClustersSector[GPUCA_NSLICES];
  unsigned int clusterOffset[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClustersTotal;
};
} // namespace tpc
namespace base
{
struct MatBudget {
};
class MatLayerCylSet
{
};
} // namespace base
namespace trd
{
class TRDGeometryFlat
{
};
} // namespace trd
} // namespace o2
#endif

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "GPUdEdxInfo.h"
#endif

#endif
