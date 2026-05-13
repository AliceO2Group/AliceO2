// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file ClustererACTS.h
/// \brief Definition of the TRK cluster finder
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-01
///

#ifndef ALICEO2_TRK_CLUSTERERACTS_H
#define ALICEO2_TRK_CLUSTERERACTS_H

#include "TRKReconstruction/Clusterer.h"

namespace o2::trk
{

class GeometryTGeo;

class ClustererACTS : public Clusterer
{
 public:
  void process(gsl::span<const Digit> digits,
               gsl::span<const DigROFRecord> digitROFs,
               std::vector<o2::trk::Cluster>& clusters,
               std::vector<unsigned char>& patterns,
               std::vector<o2::trk::ROFRecord>& clusterROFs,
               const ConstDigitTruth* digitLabels = nullptr,
               ClusterTruth* clusterLabels = nullptr,
               gsl::span<const DigMC2ROFRecord> digMC2ROFs = {},
               std::vector<o2::trk::MC2ROFRecord>* clusterMC2ROFs = nullptr) override;

 private:
};

} // namespace o2::trk

#endif
