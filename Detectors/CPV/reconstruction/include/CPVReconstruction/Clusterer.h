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
/// \brief Definition of the CPV cluster finder
#ifndef ALICEO2_CPV_CLUSTERER_H
#define ALICEO2_CPV_CLUSTERER_H
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Cluster.h"
#include "CPVReconstruction/FullCluster.h"
#include "CPVCalib/CalibParams.h"
#include "CPVCalib/BadChannelMap.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsCPV/TriggerRecord.h"

namespace o2
{
namespace cpv
{

class Clusterer
{
 public:
  Clusterer() = default;
  ~Clusterer() = default;

  void initialize();
  void process(gsl::span<const Digit> digits, gsl::span<const TriggerRecord> dtr,
               const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& dmc,
               std::vector<Cluster>* clusters, std::vector<TriggerRecord>* trigRec,
               o2::dataformats::MCTruthContainer<o2::MCCompLabel>* cluMC);

  void makeClusters(gsl::span<const Digit> digits);
  void evalCluProperties(gsl::span<const Digit> digits, std::vector<Cluster>* clusters,
                         const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& dmc,
                         o2::dataformats::MCTruthContainer<o2::MCCompLabel>* cluMC);

  float responseShape(float dx, float dz); // Parameterization of EM shower
  void propagateMC(bool toRun = true) { mRunMC = toRun; }

  void makeUnfoldings(gsl::span<const Digit> digits); // Find and unfold clusters with few local maxima
  void unfoldOneCluster(FullCluster& iniClu, char nMax, gsl::span<int> digitId, gsl::span<const Digit> digits);

 protected:
  static constexpr short NLMMax = 10; ///< maximal number of local maxima in cluster

  bool mRunMC = false;                ///< Process MC info
  int mFirstDigitInEvent;             ///< Range of digits from one event
  int mLastDigitInEvent;              ///< Range of digits from one event
  std::vector<FullCluster> mClusters; ///< internal vector of clusters
  std::vector<Digit> mDigits;         ///< vector of transient digits for cell processing

  std::vector<std::vector<float>> meInClusters = std::vector<std::vector<float>>(10, std::vector<float>(NLMMax));
  std::vector<std::vector<float>> mfij = std::vector<std::vector<float>>(10, std::vector<float>(NLMMax));
};
} // namespace cpv
} // namespace o2

#endif
