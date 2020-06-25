// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrivialClustererTask.h
/// \brief Definition of the ITS cluster finder task

#ifndef ALICEO2_ITS_TRIVIALCLUSTERERTASK
#define ALICEO2_ITS_TRIVIALCLUSTERERTASK

#include "FairTask.h"

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/TrivialClusterer.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace itsmft
{
class Digit;
}

namespace its
{
class TrivialClustererTask : public FairTask
{
  using Digit = o2::itsmft::Digit;
  using Cluster = o2::itsmft::Cluster;

 public:
  TrivialClustererTask(Bool_t useMCTruth = kTRUE);
  ~TrivialClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:
  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; ///< ITS geometry
  TrivialClusterer mTrivialClusterer;                  ///< Cluster finder

  const std::vector<Digit>* mDigitsArray = nullptr;                         ///< Array of digits
  std::vector<Cluster>* mClustersArray = nullptr;                           ///< Array of clusters
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; ///< MC labels

  ClassDefOverride(TrivialClustererTask, 2);
};
} // namespace its
} // namespace o2

#endif /* ALICEO2_ITS_TRIVIALCLUSTERERTASK */
