// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the HMPID cluster finder
#ifndef ALICEO2_HMPID_CLUSTERER_H
#define ALICEO2_HMPID_CLUSTERER_H

#include <utility>
#include <vector>
#include "HMPIDBase/Cluster.h"
#include "HMPIDBase/Digit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{

namespace hmpid
{
class Clusterer
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using Cluster = o2::hmpid::Cluster;
  using Digit = o2::hmpid::Digit;

 public:
  Clusterer() = default;
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(std::vector<Digit> const& digits, std::vector<o2::hmpid::Cluster>& clusters, MCLabelContainer const* digitMCTruth);

  void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* truth) { mClsLabels = truth; }

 private:
  //void processChamber(std::vector<Cluster>& clusters, MCLabelContainer const* digitMCTruth);
  //void fetchMCLabels(const Digit* dig, std::array<Label, Cluster::maxLabels>& labels, int& nfilled) const;

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; // Cluster MC labels

  Digit* mContributingDigit[6];    //! array of digits contributing to the cluster; this will not be stored, it is temporary to build the final cluster
  int mNumberOfContributingDigits; //! number of digits contributing to the cluster; this will not be stored, it is temporary to build the final cluster
  void addContributingDigit(Digit* dig);
  void buildCluster(Cluster& c, MCLabelContainer const* digitMCTruth);
};

} // namespace hmpid
} // namespace o2
#endif /* ALICEO2_TOF_CLUSTERER_H */
