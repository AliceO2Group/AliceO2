// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file dEdxInfo.h
/// \author David Rohr

#ifndef DEDXINFO_H
#define DEDXINFO_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace TPC
{
struct dEdxInfo {
  float dEdxTotIROC;
  float dEdxTotOROC1;
  float dEdxTotOROC2;
  float dEdxTotOROC3;
  float dEdxTotTPC;
  float dEdxMaxIROC;
  float dEdxMaxOROC1;
  float dEdxMaxOROC2;
  float dEdxMaxOROC3;
  float dEdxMaxTPC;
  unsigned char NHitsIROC;
  unsigned char NHitsSubThresholdIROC;
  unsigned char NHitsOROC1;
  unsigned char NHitsSubThresholdOROC1;
  unsigned char NHitsOROC2;
  unsigned char NHitsSubThresholdOROC2;
  unsigned char NHitsOROC3;
  unsigned char NHitsSubThresholdOROC3;
  ClassDefNV(dEdxInfo, 1);
};
} // namespace TPC
} // namespace o2

#endif
