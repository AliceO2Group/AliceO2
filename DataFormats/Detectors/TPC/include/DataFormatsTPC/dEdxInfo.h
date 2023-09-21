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

/// \file dEdxInfo.h
/// \author David Rohr

#ifndef ALICEO2_DATAFORMATSTPC_DEDXINFO_H
#define ALICEO2_DATAFORMATSTPC_DEDXINFO_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace tpc
{
struct dEdxInfo {
  float dEdxTotIROC = 0.f;
  float dEdxTotOROC1 = 0.f;
  float dEdxTotOROC2 = 0.f;
  float dEdxTotOROC3 = 0.f;
  float dEdxTotTPC = 0.f;
  float dEdxMaxIROC = 0.f;
  float dEdxMaxOROC1 = 0.f;
  float dEdxMaxOROC2 = 0.f;
  float dEdxMaxOROC3 = 0.f;
  float dEdxMaxTPC = 0.f;
  unsigned char NHitsIROC = 0;
  unsigned char NHitsSubThresholdIROC = 0;
  unsigned char NHitsOROC1 = 0;
  unsigned char NHitsSubThresholdOROC1 = 0;
  unsigned char NHitsOROC2 = 0;
  unsigned char NHitsSubThresholdOROC2 = 0;
  unsigned char NHitsOROC3 = 0;
  unsigned char NHitsSubThresholdOROC3 = 0;
  ClassDefNV(dEdxInfo, 1);
};
} // namespace tpc
} // namespace o2

#endif
