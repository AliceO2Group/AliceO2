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
/// \brief Definition of the TOF cluster finder
#ifndef ALICEO2_TOF_CLUSTERER_H
#define ALICEO2_TOF_CLUSTERER_H

#include <utility>
#include <vector>
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "TOFBase/Geo.h"
#include "TOFReconstruction/DataReader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TOFBase/CalibTOFapi.h"
#include "DataFormatsTOF/Diagnostic.h"

namespace o2
{

namespace tof
{
class Clusterer
{
  using MCLabelContainer = o2::dataformats::MCLabelContainer;
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
  }

  void addCalibFromCluster(int ch1, int8_t dch, float dtime, short tot1, short tot2);

  void setFirstOrbit(uint64_t orb);
  uint64_t getFirstOrbit() const { return mFirstOrbit; }

  void setCalibFromCluster(bool val = 1) { mCalibFromCluster = val; }
  bool isCalibFromCluster() const { return mCalibFromCluster; }

  void setDeltaTforClustering(float val) { mDeltaTforClustering = val; }
  float getDeltaTforClustering() const { return mDeltaTforClustering; }
  std::vector<o2::tof::CalibInfoCluster>* getInfoFromCluster() { return &mCalibInfosFromCluster; }
  void addDiagnostic(const Diagnostic& dia)
  {
    mDiagnosticFrequency.merge(&dia);
    mDiagnosticFrequency.getNoisyMap(mIsNoisy);
  };
  void clearDiagnostic()
  {
    memset(mIsNoisy, 0, Geo::NCHANNELS * sizeof(mIsNoisy[0]));
    mDiagnosticFrequency.clear();
  }

  bool areCalibStored() const { return mAreCalibStored; }
  void setCalibStored(bool val = true) { mAreCalibStored = val; }

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
  uint64_t mFirstOrbit = 0;      //! 1st orbit of the TF
  uint64_t mBCOffset = 0;        //! 1st orbit of the TF converted to BCs

  float mDeltaTforClustering = 5000; //! delta time (in ps) accepted for clustering
  bool mCalibFromCluster = false;    //! if producing calib from clusters
  Diagnostic mDiagnosticFrequency;   //! diagnostic frquency in current TF
  bool mIsNoisy[Geo::NCHANNELS];     //! noisy channel map

  std::vector<o2::tof::CalibInfoCluster> mCalibInfosFromCluster;

  bool mAreCalibStored = false;
};

} // namespace tof
} // namespace o2
#endif /* ALICEO2_TOF_CLUSTERER_H */
