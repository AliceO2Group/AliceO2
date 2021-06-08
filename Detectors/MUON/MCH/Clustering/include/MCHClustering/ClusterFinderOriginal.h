// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderOriginal.h
/// \brief Definition of a class to reconstruct clusters with the original MLEM algorithm
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_CLUSTERFINDERORIGINAL_H_
#define ALICEO2_MCH_CLUSTERFINDERORIGINAL_H_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gsl/span>

#include <TH2D.h>

#include "DataFormatsMCH/Digit.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHPreClustering/PreClusterFinder.h"

namespace o2
{
namespace mch
{

class PadOriginal;
class ClusterOriginal;
class MathiesonOriginal;

class ClusterFinderOriginal
{
 public:
  ClusterFinderOriginal();
  ~ClusterFinderOriginal();

  ClusterFinderOriginal(const ClusterFinderOriginal&) = delete;
  ClusterFinderOriginal& operator=(const ClusterFinderOriginal&) = delete;
  ClusterFinderOriginal(ClusterFinderOriginal&&) = delete;
  ClusterFinderOriginal& operator=(ClusterFinderOriginal&&) = delete;

  void init(bool run2Config);
  void deinit();
  void reset();

  void findClusters(gsl::span<const Digit> digits);

  /// return the list of reconstructed clusters
  const std::vector<ClusterStruct>& getClusters() const { return mClusters; }
  /// return the list of digits used in reconstructed clusters
  const std::vector<Digit>& getUsedDigits() const { return mUsedDigits; }

 private:
  static constexpr double SDistancePrecision = 1.e-3;            ///< precision used to check overlaps and so on (cm)
  static constexpr int SNFitClustersMax = 3;                     ///< maximum number of clusters fitted at the same time
  static constexpr int SNFitParamMax = 3 * SNFitClustersMax - 1; ///< maximum number of fit parameters
  static constexpr double SLowestCoupling = 1.e-2;               ///< minimum coupling between clusters of pixels and pads

  void resetPreCluster(gsl::span<const Digit>& digits);
  void simplifyPreCluster(std::vector<int>& removedDigits);
  void processPreCluster();

  void buildPixArray();
  void ProjectPadOverPixels(const PadOriginal& pad, TH2D& hCharges, TH2I& hEntries) const;

  void findLocalMaxima(std::unique_ptr<TH2D>& histAnode, std::multimap<double, std::pair<int, int>, std::greater<>>& localMaxima);
  void flagLocalMaxima(const TH2D& histAnode, int i0, int j0, std::vector<std::vector<int>>& isLocalMax) const;
  void restrictPreCluster(const TH2D& histAnode, int i0, int j0);

  void processSimple();
  void process();
  void addVirtualPad();
  void computeCoefficients(std::vector<double>& coef, std::vector<double>& prob) const;
  double mlem(const std::vector<double>& coef, const std::vector<double>& prob, int nIter);
  void findCOG(const TH2D& histMLEM, double xy[2]) const;
  void refinePixelArray(const double xyCOG[2], size_t nPixMax, double& xMin, double& xMax, double& yMin, double& yMax);
  void cleanPixelArray(double threshold, std::vector<double>& prob);

  int fit(const std::vector<const std::vector<int>*>& clustersOfPixels, const double fitRange[2][2], double fitParam[SNFitParamMax + 1]);
  double fit(double currentParam[SNFitParamMax + 2], const double parmin[SNFitParamMax], const double parmax[SNFitParamMax],
             int nParamUsed, int& nTrials) const;
  double computeChi2(const double param[SNFitParamMax + 2], int nParamUsed) const;
  void param2ChargeFraction(const double param[SNFitParamMax], int nParamUsed, double fraction[SNFitClustersMax]) const;
  float chargeIntegration(double x, double y, const PadOriginal& pad) const;

  void split(const TH2D& histMLEM, const std::vector<double>& coef);
  void addPixel(const TH2D& histMLEM, int i0, int j0, std::vector<int>& pixels, std::vector<std::vector<bool>>& isUsed);
  void addCluster(int iCluster, std::vector<int>& coupledClusters, std::vector<bool>& isClUsed,
                  const std::vector<std::vector<double>>& couplingClCl) const;
  void extractLeastCoupledClusters(std::vector<int>& coupledClusters, std::vector<int>& clustersForFit,
                                   const std::vector<std::vector<double>>& couplingClCl) const;
  int selectPads(const std::vector<int>& coupledClusters, const std::vector<int>& clustersForFit,
                 const std::vector<std::vector<double>>& couplingClPad);
  void merge(const std::vector<int>& clustersForFit, const std::vector<int>& coupledClusters, std::vector<std::vector<int>>& clustersOfPixels,
             std::vector<std::vector<double>>& couplingClCl, std::vector<std::vector<double>>& couplingClPad) const;
  void updatePads(const double fitParam[SNFitParamMax + 1], int nParamUsed);

  void setClusterResolution(ClusterStruct& cluster) const;

  /// function to reinterpret digit ADC as charge
  std::function<double(uint32_t)> mADCToCharge = [](uint32_t adc) { return static_cast<double>(adc); };

  double mLowestPadCharge = 0.;     ///< minimum charge of a pad
  double mLowestPixelCharge = 0.;   ///< minimum charge of a pixel
  double mLowestClusterCharge = 0.; ///< minimum charge of a cluster

  std::unique_ptr<MathiesonOriginal[]> mMathiesons; ///< Mathieson functions for station 1 and the others
  MathiesonOriginal* mMathieson = nullptr;          ///< pointer to the Mathieson function currently used

  std::unique_ptr<ClusterOriginal> mPreCluster; ///< precluster currently processed
  std::vector<PadOriginal> mPixels;             ///< list of pixels for the current precluster

  const mapping::Segmentation* mSegmentation = nullptr; ///< pointer to the DE segmentation for the current precluster

  std::vector<ClusterStruct> mClusters{}; ///< list of reconstructed clusters
  std::vector<Digit> mUsedDigits{};       ///< list of digits used in reconstructed clusters

  PreClusterFinder mPreClusterFinder{}; ///< preclusterizer
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_CLUSTERFINDERORIGINAL_H_
