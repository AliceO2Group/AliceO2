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

/// \file ClusterFindergem.h
/// \brief Definition of a class to reconstruct clusters with the gem MLEM algorithm
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_CLUSTERFINDERGEM_H_
#define O2_MCH_CLUSTERFINDERGEM_H_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gsl/span>

#include <TH2D.h>

#include "DataFormatsMCH/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHPreClustering/PreClusterFinder.h"
#include "ClusterFinderOriginal.h"
#include "MCHClustering/ClusterPEM.h"

// GG Added include
#include "ClusterDump.h"
#include "clusterProcessing.h"

namespace o2
{
namespace mch
{
class PadOriginal;
class ClusterOriginal;

// GG class MathiesonOriginal;

class ClusterFinderGEM
{
 public:
  ClusterFinderGEM();
  ~ClusterFinderGEM();

  ClusterFinderGEM(const ClusterFinderGEM&) = delete;
  ClusterFinderGEM& operator=(const ClusterFinderGEM&) = delete;
  ClusterFinderGEM(ClusterFinderGEM&&) = delete;
  ClusterFinderGEM& operator=(ClusterFinderGEM&&) = delete;

  //
  // GG method called by the process workflow ( ClusterFinderGEMSpec )
  //

  void init(int mode, bool run2Config);
  void deinit();
  void reset();
  void fillGEMInputData(gsl::span<const Digit>& digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster);
  void releasePreCluster();
  //
  void findClusters(gsl::span<const Digit> digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster);
  //
  /// return the list of reconstructed clusters

  const std::vector<Cluster>& getClusters() const { return mClusters; }
  /// return the list of digits used in reconstructed clusters
  const std::vector<Digit>& getUsedDigits() const { return mUsedDigits; }
  void dumpPreCluster(ClusterDump* dumpFile, gsl::span<const Digit> digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster);
  void dumpClusterResults(ClusterDump* dumpFile, const std::vector<Cluster>& clusters, size_t startIdx, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster);

 private:
  // GG Original commented
  // Invalid static constexpr double SDistancePrecision = 1.e-3;                   ///< precision used to check overlaps and so on (cm)
  // static constexpr double SLowestPadCharge = 4.f * 0.22875f;            ///< minimum charge of a pad
  // static constexpr double SLowestPixelCharge = SLowestPadCharge / 12.;  ///< minimum charge of a pixel
  // static constexpr double SLowestClusterCharge = 2. * SLowestPadCharge; ///< minimum charge of a cluster

  static constexpr int SNFitClustersMax = 3;                     ///< maximum number of clusters fitted at the same time
  static constexpr int SNFitParamMax = 3 * SNFitClustersMax - 1; ///< maximum number of fit parameters
  static constexpr double SLowestCoupling = 1.e-2;               ///< minimum coupling between clusters of pixels and pads

  // Invalid ???
  // static constexpr char statFileName[] = "statistics.csv";
  // std::fstream statStream;

  // GG Unused
  // void resetPreCluster(gsl::span<const Digit>& digits);
  // void simplifyPreCluster(std::vector<int>& removedDigits);
  // void processPreCluster();

  // void buildPixArray();
  // void ProjectPadOverPixels(const PadOriginal& pad, TH2D& hCharges, TH2I& hEntries) const;

  // void findLocalMaxima(std::unique_ptr<TH2D>& histAnode, std::multimap<double, std::pair<int, int>, std::greater<>>& localMaxima);
  // void flagLocalMaxima(const TH2D& histAnode, int i0, int j0, std::vector<std::vector<int>>& isLocalMax) const;
  // void restrictPreCluster(const TH2D& histAnode, int i0, int j0);

  // void processSimple();
  // void process();
  // void addVirtualPad();
  // void computeCoefficients(std::vector<double>& coef, std::vector<double>& prob) const;
  // double mlem(const std::vector<double>& coef, const std::vector<double>& prob, int nIter);
  // void findCOG(const TH2D& histMLEM, double xy[2]) const;
  // void refinePixelArray(const double xyCOG[2], size_t nPixMax, double& xMin, double& xMax, double& yMin, double& yMax);
  // void shiftPixelsToKeep(double charge);
  // void cleanPixelArray(double threshold, std::vector<double>& prob);

  // int fit(const std::vector<const std::vector<int>*>& clustersOfPixels, const double fitRange[2][2], double fitParam[SNFitParamMax + 1]);
  // double fit(double currentParam[SNFitParamMax + 2], const double parmin[SNFitParamMax], const double parmax[SNFitParamMax],
  //            int nParamUsed, int& nTrials) const;
  // double computeChi2(const double param[SNFitParamMax + 2], int nParamUsed) const;
  // void param2ChargeFraction(const double param[SNFitParamMax], int nParamUsed, double fraction[SNFitClustersMax]) const;
  //
  // float chargeIntegration(double x, double y, const PadOriginal& pad) const;
  //
  // void split(const TH2D& histMLEM, const std::vector<double>& coef);
  // void addPixel(const TH2D& histMLEM, int i0, int j0, std::vector<int>& pixels, std::vector<std::vector<bool>>& isUsed);
  // void addCluster(int iCluster, std::vector<int>& coupledClusters, std::vector<bool>& isClUsed,
  //                 const std::vector<std::vector<double>>& couplingClCl) const;
  //  void extractLeastCoupledClusters(std::vector<int>& coupledClusters, std::vector<int>& clustersForFit,
  //                                  const std::vector<std::vector<double>>& couplingClCl) const;
  //  int selectPads(const std::vector<int>& coupledClusters, const std::vector<int>& clustersForFit,
  //                const std::vector<std::vector<double>>& couplingClPad);
  //  void merge(const std::vector<int>& clustersForFit, const std::vector<int>& coupledClusters, std::vector<std::vector<int>>& clustersOfPixels,
  //            std::vector<std::vector<double>>& couplingClCl, std::vector<std::vector<double>>& couplingClPad) const;
  //  void updatePads(const double fitParam[SNFitParamMax + 1], int nParamUsed);
  void setClusterResolution(Cluster& cluster) const;
  std::unique_ptr<MathiesonOriginal[]> mMathiesons; ///< Mathieson functions for station 1 and the others
  // GG MathiesonOriginal* mMathieson = nullptr;          ///< pointer to the Mathieson function currently used
  //
  // GG Introduced for run3
  // function to reinterpret digit ADC as charge
  std::function<double(uint32_t)> mADCToCharge = [](uint32_t adc) { return static_cast<double>(adc); };
  //
  std::unique_ptr<ClusterOriginal> mPreCluster; ///< precluster currently processed
  // GG  std::vector<PadOriginal> mPixels;   ///< list of pixels for the current precluster

  const mapping::Segmentation* mSegmentation = nullptr; ///< pointer to the DE segmentation for the current precluster
  std::vector<Cluster> mClusters{};                     ///< list of reconstructed clusters
  std::vector<Digit> mUsedDigits{};                     ///< list of digits used in reconstructed clusters

  PreClusterFinder mPreClusterFinder{}; ///< preclusterizer

  //
  // GG Added to process GEM and use Dump Files
  void initPreCluster(gsl::span<const Digit>& digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster);

  int mode;
  int nPads;
  double* xyDxy;
  Mask_t* cathode;
  Mask_t* saturated;
  double* padCharge;
  int DEId;
  // PreCluster Identification
  uint32_t currentBC;
  uint32_t currentOrbit;
  uint32_t currentPreClusterID;

  // Dump Files
  // Invalid
  // ClusterDump* pOriginalClusterDump;
  // ClusterDump* pGEMClusterDump;
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_CLUSTERFINDERGEM_H_
