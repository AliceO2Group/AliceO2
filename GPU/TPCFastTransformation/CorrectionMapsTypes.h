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

/// \file CorrectionMapsTypes.h
/// \brief Helper class for enums and structs related to the correction maps
/// \author matthias.kleiner@cern.ch

#ifndef TPC_CORRECTION_MAPS_TYPES_H_
#define TPC_CORRECTION_MAPS_TYPES_H_

namespace o2::tpc
{
enum class LumiScaleType : int {
  Unset = -1,    ///< init value
  NoScaling = 0, ///< no scaling, use map as is
  CTPLumi = 1,   ///< use CTP luminosity for scaling
  TPCScaler = 2  ///< use TPC scaler for scaling
};

enum class LumiScaleMode : int {
  Unset = -1,         ///< init value
  Linear = 0,         ///< map(lumi) = (mean_map - referenceMap) * lumiScale + referenceMap
  DerivativeMap = 1,  ///< map(lumi) = mean_map + lumiScale * (derivativeMap) where derivativeMap = (mean_map_A - mean_map_B)
  DerivativeMapMC = 2 ///< same DerivativeMap, but for MC
};

struct CorrectionMapsGloOpts {
  LumiScaleType lumiType = LumiScaleType::Unset; ///< what estimator to used for corrections scaling: 0: no scaling, 1: CTP, 2: IDC
  LumiScaleMode lumiMode = LumiScaleMode::Unset; ///< what corrections method to use: 0: classical scaling, 1: Using of the derivative map, 2: Using of the derivative map for MC
  bool enableMShapeCorrection = false;
  bool requestCTPLumi = true;         ///< request CTP Lumi regardless of what is used for corrections scaling
  bool checkCTPIDCconsistency = true; ///< check the selected CTP or IDC scaling source being consistent with mean scaler of the map
};
} // namespace o2::tpc
#endif
