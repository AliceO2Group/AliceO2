// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file KrBoxClusterFinderParam.h
/// \brief Parameter class for the Kr box cluster finder
///
/// \author Philip Hauer, hauer@hiskp.uni-bonn.de

#ifndef ALICEO2_TPC_KrBoxClusterFinderParam_H_
#define ALICEO2_TPC_KrBoxClusterFinderParam_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

struct KrBoxClusterFinderParam : public o2::conf::ConfigurableParamHelper<KrBoxClusterFinderParam> {
  int MaxClusterSizeTime{3}; ///< "radius" of a cluster in time direction

  int MaxClusterSizeRowIROC{3};  ///< "radius" of a cluster in row direction in IROC
  int MaxClusterSizeRowOROC1{2}; ///< "radius" of a cluster in row direction in OROC1
  int MaxClusterSizeRowOROC2{2}; ///< "radius" of a cluster in row direction in OROC2
  int MaxClusterSizeRowOROC3{1}; ///< "radius" of a cluster in row direction in OROC3

  int MaxClusterSizePadIROC{5};  ///< "radius" of a cluster in pad direction in IROC
  int MaxClusterSizePadOROC1{3}; ///< "radius" of a cluster in pad direction in OROC1
  int MaxClusterSizePadOROC2{3}; ///< "radius" of a cluster in pad direction in OROC2
  int MaxClusterSizePadOROC3{3}; ///< "radius" of a cluster in pad direction in OROC3

  float QThresholdMax{30.0};    ///< the Maximum charge in a cluster must exceed this value or it is discarded
  float QThreshold{1.0};        ///< every charge which is added to a cluster must exceed this value or it is discarded
  int MinNumberOfNeighbours{2}; ///< amount of direct neighbours required for a cluster maximum

  O2ParamDef(KrBoxClusterFinderParam, "TPCKrBoxClusterFinder");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_KrBoxClusterFinderParam_H_