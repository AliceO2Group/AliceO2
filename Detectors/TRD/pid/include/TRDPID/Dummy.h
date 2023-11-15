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

/// \file Dummy.h
/// \brief This file provides a dummy model, which only outputs -1.f
/// \author Felix Schlepper

#ifndef O2_TRD_DUMMY_H
#define O2_TRD_DUMMY_H

#include "Rtypes.h"
#include "TRDPID/PIDBase.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "Framework/ProcessingContext.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

namespace o2
{
namespace trd
{

/// This is the ML Base class which defines the interface all machine learning
/// models.
class Dummy final : public PIDBase
{
  using PIDBase::PIDBase;

 public:
  ~Dummy() = default;

  /// Do absolutely nothing.
  void init(o2::framework::ProcessingContext& pc) final{};

  /// Everything below 0.f indicates nothing available.
  float process(const TrackTRD& trk, const o2::globaltracking::RecoContainer& input, bool isTPCTRD) const final
  {
    return -1.f;
  };

 private:
  ClassDefNV(Dummy, 1);
};

} // namespace trd
} // namespace o2

#endif
