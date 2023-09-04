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

/// \file Clusterer.h
/// \brief Definition of the HMPID cluster finder
#ifndef ALICEO2_HMPID_CLUSTERER_H
#define ALICEO2_HMPID_CLUSTERER_H

#include <utility>
#include <vector>
#include "HMPIDBase/Param.h"
#include "DataFormatsHMP/Cluster.h"
#include "DataFormatsHMP/Digit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "TMatrixF.h" // ef: added

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

  // void process(std::vector<Digit> const& digits, std::vector<o2::hmpid::Cluster>& clusters, MCLabelContainer const* digitMCTruth);

  // void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* truth) { mClsLabels = truth; }

  static void Dig2Clu(gsl::span<const o2::hmpid::Digit> digs, std::vector<o2::hmpid::Cluster>& clus, float* pUserCut, bool isUnfold = kTRUE); // digits->clusters
  static void FormClu(Cluster& pClu, int pDig, gsl::span<const o2::hmpid::Digit> digs, TMatrixF& pDigMap);                                    // cluster formation recursive algorithm
  static int UseDig(int padX, int padY, TMatrixF& pDigMap);                                                                                   // use this pad's digit to form a cluster
  inline bool IsDigSurvive(Digit* pDig) const;                                                                                                // check for sigma cut

 private:
  // void processChamber(std::vector<Cluster>& clusters, MCLabelContainer const* digitMCTruth);
  // void fetchMCLabels(const Digit* dig, std::array<Label, Cluster::maxLabels>& labels, int& nfilled) const;

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; // Cluster MC labels

  // Digit* mContributingDigit[6];    //! array of digits contributing to the cluster; this will not be stored, it is temporary to build the final cluster
  // int mNumberOfContributingDigits; //! number of digits contributing to the cluster; this will not be stored, it is temporary to build the final cluster
  // std::vector<o2::hmpid::Digit*> mDigs;
  // std::vector<o2::hmpid::Cluster*> mClus;
  //  void addContributingDigit(Digit* dig);
  //  void buildCluster(Cluster& c, MCLabelContainer const* digitMCTruth);
};

} // namespace hmpid
} // namespace o2
#endif /* ALICEO2_TOF_CLUSTERER_H */
