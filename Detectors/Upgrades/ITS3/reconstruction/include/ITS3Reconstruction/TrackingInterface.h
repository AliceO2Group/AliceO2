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

#ifndef O2_ITS3_TRACKINGINTERFACE
#define O2_ITS3_TRACKINGINTERFACE

#include "ITStracking/TrackingInterface.h"
#include "ITS3Reconstruction/TopologyDictionary.h"

namespace o2::its3
{

class ITS3TrackingInterface final : public its::ITSTrackingInterface
{
 public:
  using its::ITSTrackingInterface::ITSTrackingInterface;

  void setClusterDictionary(const o2::its3::TopologyDictionary* d) { mDict = d; }
  void updateTimeDependentParams(framework::ProcessingContext& pc) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 protected:
  void loadROF(gsl::span<itsmft::ROFRecord>& trackROFspan,
               gsl::span<const itsmft::CompClusterExt> clusters,
               gsl::span<const unsigned char>::iterator& pattIt,
               const dataformats::MCTruthContainer<MCCompLabel>* mcLabels) final;

 private:
  const o2::its3::TopologyDictionary* mDict{nullptr};
};

} // namespace o2::its3

#endif
