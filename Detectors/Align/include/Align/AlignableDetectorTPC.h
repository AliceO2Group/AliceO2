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

/// @file   AlignableDetectorTPC.h
/// @author ruben.shahoyan@cern.ch
/// @brief  TPC detector wrapper

#ifndef ALIGNABLEDETECTORTPC_H
#define ALIGNABLEDETECTORTPC_H

#include "Align/AlignableDetector.h"

namespace o2
{
namespace align
{

class AlignableDetectorTPC final : public AlignableDetector
{
 public:
  //
  AlignableDetectorTPC() = default;
  AlignableDetectorTPC(Controller* ctr);
  ~AlignableDetectorTPC() final = default;
  void defineVolumes() final;
  void Print(const Option_t* opt = "") const final;
  //
  int processPoints(GIndex gid, int npntCut, bool inv) final;

  void setTrackTimeStamp(float t) { mTrackTimeStamp = t; }
  float getTrackTimeStamp() const { return mTrackTimeStamp; }

 protected:
  //
  float mTrackTimeStamp = 0.f; // use track timestamp in \mus

  ClassDef(AlignableDetectorTPC, 1);
};
} // namespace align
} // namespace o2
#endif
