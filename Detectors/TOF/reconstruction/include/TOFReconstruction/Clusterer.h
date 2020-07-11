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
/// \brief Definition of the TOF cluster finder
#ifndef ALICEO2_TOF_CLUSTERER_H
#define ALICEO2_TOF_CLUSTERER_H

#include <utility>
#include <vector>
#include "DataFormatsTOF/Cluster.h"
#include "TOFBase/Geo.h"
#include "TOFReconstruction/DataReader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TOFCalibration/CalibTOFapi.h"

namespace o2
{

namespace tof
{
class Clusterer
{
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using Cluster = o2::tof::Cluster;
  using StripData = o2::tof::DataReader::StripData;
  using Digit = o2::tof::Digit;
  using CalibApi = o2::tof::CalibTOFapi;

 public:
  Clusterer() = default;
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(DataReader& r, std::vector<Cluster>& clusters, MCLabelContainer const* digitMCTruth);

  void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* truth) { mClsLabels = truth; }

  void setCalibApi(CalibApi* calibApi)
  {
    mCalibApi = calibApi;
    Printf("mCalibApi = %p", mCalibApi);
  }

  void setFirstOrbit(uint32_t orb);
  uint32_t getFirstOrbit() const { return mFirstOrbit; }

 private:
  void calibrateStrip();
  void processStrip(std::vector<Cluster>& clusters, MCLabelContainer const* digitMCTruth);
  //void fetchMCLabels(const Digit* dig, std::array<Label, Cluster::maxLabels>& labels, int& nfilled) const;

  StripData mStripData; ///< single strip data provided by the reader

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; // Cluster MC labels

  Digit* mContributingDigit[6];    //! array of digits contributing to the cluster; this will not be stored, it is temporary to build the final cluster
  int mNumberOfContributingDigits; //! number of digits contributing to the cluster; this will not be stored, it is temporary to build the final cluster
  void addContributingDigit(Digit* dig);
  void buildCluster(Cluster& c, MCLabelContainer const* digitMCTruth);
  CalibApi* mCalibApi = nullptr; //! calib api to handle the TOF calibration
  uint32_t mFirstOrbit = 0;      //! 1st orbit of the TF
  uint64_t mBCOffset = 0;        //! 1st orbit of the TF converted to BCs
};

} // namespace tof
} // namespace o2
#endif /* ALICEO2_TOF_CLUSTERER_H */
