// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

#include <string>

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

  float CutMinSigmaTime{0};      ///< Min sigma time to accept cluster
  float CutMaxSigmaTime{1000};   ///< Min sigma time to accept cluster
  float CutMinSigmaPad{0};       ///< Min sigma pad to accept cluster
  float CutMaxSigmaPad{1000};    ///< Min sigma pad to accept cluster
  float CutMinSigmaRow{0};       ///< Min sigma row to accept cluster
  float CutMaxSigmaRow{1000};    ///< Min sigma row to accept cluster
  float CutMaxQtot{1e10};        ///< Max Qtot to accept cluster
  float CutQtot0{1e10};          ///< Max Qtot at zero size for Qtot vs. size correlation cut
  float CutQtotSizeSlope{0};     ///< Max Qtot over size slope for Qtot vs. size correlation cut
  unsigned char CutMaxSize{255}; ///< Max cluster size in number of digits
  bool ApplyCuts{false};         ///< if to apply cluster cuts above

  std::string GainMapFile{};          ///< gain map file to apply during reconstruction
  std::string GainMapName{"GainMap"}; ///< gain map file to apply during reconstruction

  O2ParamDef(KrBoxClusterFinderParam, "TPCKrBoxClusterFinder");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_KrBoxClusterFinderParam_H_
