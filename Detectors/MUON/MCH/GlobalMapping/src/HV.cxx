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

#include "HV.h"

#include "MCHConditions/Chamber.h"
#include "MCHConditions/DetectionElement.h"
#include "MCHConditions/Number.h"
#include "MCHGlobalMapping/DsIndex.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/Mapper.h"

namespace o2::mch::dcs
{

double findXmean(const o2::mch::mapping::Segmentation& seg, int dsId)
{
  double xmean = 0.0;
  int n = 0;
  for (auto ch = 0; ch < 64; ++ch) {
    auto pad = seg.findPadByFEE(dsId, ch);
    if (seg.isValid(pad)) {
      double x = seg.padPositionX(pad);
      n++;
      xmean += x;
    }
  }
  return xmean / n;
}

std::set<int> hvAliasToDsIndices(std::string_view alias)
{
  const auto chamber = aliasToChamber(alias);
  bool slat = isSlat(chamber);
  int deId = aliasToDetElemId(alias).value();
  std::set<int> indices;
  for (auto dsIndex = 0; dsIndex < o2::mch::NumberOfDualSampas; ++dsIndex) {
    const auto& dd = getDsDetId(dsIndex);
    if (dd.deId() != deId) {
      continue;
    }
    if (slat) {
      indices.emplace(dsIndex);
    } else {
      const auto& seg = o2::mch::mapping::segmentation(deId);
      auto xref = 10 * findXmean(seg, dd.dsId());
      int sector{-1};
      double x0, x1, x2, x3;
      if (dd.deId() < 300) {
        x0 = -10,
        x1 = 291.65;
        x2 = 585.65;
        x3 = 879.65;

      } else {
        x0 = -140;
        x1 = 283.75;
        x2 = 606.25;
        x3 = 1158.75;
      }
      if (xref < x0) {
        throw std::invalid_argument("x<x0");
      }
      if (xref < x1) {
        sector = 0;
      } else if (xref < x2) {
        sector = 1;
      } else if (xref < x3) {
        sector = 2;
      }
      const auto number = aliasToNumber(alias);
      if (sector == number % 10) {
        indices.emplace(dsIndex);
      }
    }
  }
  return indices;
}

} // namespace o2::mch::dcs
