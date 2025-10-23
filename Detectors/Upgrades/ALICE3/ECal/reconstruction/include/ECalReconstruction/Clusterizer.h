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

/// \file Clusterizer.h
/// \brief Class for cluster finding and unfolding
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_CLUSTERIZER_H
#define ALICEO2_ECAL_CLUSTERIZER_H

#include <gsl/span>
#include <vector>
#include <DataFormatsECal/Digit.h>
#include <DataFormatsECal/Cluster.h>

using o2::ecal::Cluster;
using o2::ecal::Digit;
class TF1;

namespace o2
{
namespace ecal
{
class Clusterizer
{
 public:
  Clusterizer(bool applyCorrectionZ = 1, bool applyCorrectionE = 1);
  ~Clusterizer() = default;
  void initialize() {};
  void addDigitToCluster(Cluster& cluster, int row, int col, const gsl::span<const Digit>& digits);
  void findClusters(const gsl::span<const Digit>& digits, std::vector<Cluster>& foundClusters, std::vector<Cluster>& unfoldedClusters);
  void makeClusters(const gsl::span<const Digit>& digits, std::vector<Cluster>& clusters);
  void makeUnfoldings(std::vector<Cluster>& foundClusters, std::vector<Cluster>& unfoldedClusters);
  void unfoldOneCluster(Cluster* iniClu, int nMax, int* digitId, float* maxAtEnergy, std::vector<Cluster>& unfoldedClusters);
  void evalClusters(std::vector<Cluster>& clusters);
  int getNumberOfLocalMax(Cluster& clu, int* maxAt, float* maxAtEnergy);
  double showerShape(double dx, double dz, bool isCrystal);
  void setLogWeight(double logWeight) { mLogWeight = logWeight; }
  void setClusteringThreshold(double threshold) { mClusteringThreshold = threshold; }
  void setCrystalDigitThreshold(double threshold) { mCrystalDigitThreshold = threshold; }
  void setSamplingDigitThreshold(double threshold) { mSamplingDigitThreshold = threshold; }
  void setCrystalEnergyCorrectionPars(std::vector<double> pars) { mCrystalEnergyCorrectionPars = pars; }
  void setSamplingEnergyCorrectionPars(std::vector<double> pars) { mSamplingEnergyCorrectionPars = pars; }
  void setCrystalZCorrectionPars(std::vector<double> pars) { mCrystalZCorrectionPars = pars; }
  void setSamplingZCorrectionPars(std::vector<double> pars) { mSamplingZCorrectionPars = pars; }

 private:
  std::vector<std::vector<int>> mDigitIndices;       // 2D map of digit indices used for recursive cluster finding
  bool mUnfoldClusters = true;                       // to perform cluster unfolding
  double mCrystalDigitThreshold = 0.040;             // minimal energy of crystal digit
  double mSamplingDigitThreshold = 0.100;            // minimal energy of sampling digit
  double mClusteringThreshold = 0.050;               // minimal energy of digit to start clustering (GeV)
  double mClusteringTimeGate = 1e9;                  // maximal time difference between digits to be accepted to clusters (in ns)
  int mNLMMax = 30;                                  // maximal number of local maxima in unfolding
  double mLogWeight = 4.;                            // cutoff used in log. weight calculation
  double mUnfogingEAccuracy = 1.e-4;                 // accuracy of energy calculation in unfoding prosedure (GeV)
  double mUnfogingXZAccuracy = 1.e-2;                // accuracy of position calculation in unfolding procedure (cm)
  int mNMaxIterations = 100;                         // maximal number of iterations in unfolding procedure
  double mLocalMaximumCut = 0.015;                   // minimal height of local maximum over neighbours
  bool mApplyCorrectionZ = 1;                        // apply z-correction
  bool mApplyCorrectionE = 1;                        // apply energy-correction
  TF1* fCrystalShowerShape;                          //! Crystal shower shape
  TF1* fSamplingShowerShape;                         //! Sampling shower shape
  TF1* fCrystalRMS;                                  //! Crystal RMS
  std::vector<double> mCrystalEnergyCorrectionPars;  // crystal energy-correction parameters
  std::vector<double> mSamplingEnergyCorrectionPars; // sampling energy-correction parameters
  std::vector<double> mCrystalZCorrectionPars;       // crystal z-correction parameters
  std::vector<double> mSamplingZCorrectionPars;      // sampling z-correction parameters
};

} // namespace ecal
} // namespace o2

#endif // ALICEO2_ECAL_CLUSTERIZER_H
