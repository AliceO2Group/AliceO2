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
#ifndef ALICEO2_EMCAL_CLUSTERFACTORY_H_
#define ALICEO2_EMCAL_CLUSTERFACTORY_H_
#include <array>
#include <gsl/span>
#include "Rtypes.h"
#include "fmt/format.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/AnalysisCluster.h"
#include "EMCALBase/Geometry.h"
#include "MathUtils/Cartesian.h"

namespace o2
{

namespace emcal
{

/// \class ClusterFactory
/// \ingroup EMCALbase
/// \brief EMCal clusters factory
///  Ported from  class AliEMCALcluster
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since March 05, 2020
///
template <class InputType>
class ClusterFactory
{

 public:
  class ClusterRangeException final : public std::exception
  {
   public:
    /// \brief Constructor defining the error
    /// \param clusterIndex Cluster ID responsible for the exception
    /// \param maxClusters Maximum number of clusters handled by the cluster factory
    ClusterRangeException(int clusterIndex, int maxClusters) : std::exception(),
                                                               mClusterID(clusterIndex),
                                                               mMaxClusters(maxClusters),
                                                               mErrorMessage()
    {
      mErrorMessage = fmt::format("Cluster out of range: %d, max %d", mClusterID, mMaxClusters);
    }

    /// \brief Destructor
    ~ClusterRangeException() noexcept final = default;

    /// \brief Provide error message
    /// \return Error message connected to this exception
    const char* what() const noexcept final { return mErrorMessage.data(); }

    /// \brief Get the ID of the event raising the exception
    /// \return Event ID
    int getClusterID() const { return mClusterID; }

    /// \brief Get the maximum number of events handled by the event handler
    /// \return Max. number of event
    int getMaxNumberOfClusters() const { return mMaxClusters; }

   private:
    int mClusterID = 0;        ///< Cluster ID raising the exception
    int mMaxClusters = 0;      ///< Max. number of clusters handled by this cluster factory
    std::string mErrorMessage; ///< Error message
  };

  class CellIndexRangeException final : public std::exception
  {
   public:
    /// \brief Constructor defining the error
    /// \param cellIndex Cell Index responsible for the exception
    /// \param maxCellIndex Maximum number of Cell Indices handled by the ClusterFactor
    CellIndexRangeException(int cellIndex, int maxCellIndex) : std::exception(),
                                                               mCellIndex(cellIndex),
                                                               mMaxCellIndex(maxCellIndex),
                                                               mErrorMessage()
    {
      mErrorMessage = Form("Cell Index out of range: %d, max %d", mCellIndex, mMaxCellIndex);
    }

    /// \brief Destructor
    ~CellIndexRangeException() noexcept final = default;

    /// \brief Provide error message
    /// \return Error message connected to this exception
    const char* what() const noexcept final { return mErrorMessage.data(); }

    /// \brief Get the index of the cell raising the exception
    /// \return Cell index
    int getCellIndex() const { return mCellIndex; }

    /// \brief Get the maximum number of cell indices handled by the cluster factory
    /// \return Max. number of cell indices
    int getMaxNumberOfCellIndexs() const { return mMaxCellIndex; }

   private:
    int mCellIndex = 0;        ///< CellIndex ID raising the exception
    int mMaxCellIndex = 0;     ///< Max. number of CellIndexs handled by this CellIndex factory
    std::string mErrorMessage; ///< Error message
  };

  class ClusterIterator
  {
   public:
    /// \brief Constructor, initializing the iterator
    /// \param factory cluster factory to iterate over
    /// \param clusterIndex cluster ID from which to start the iteration
    /// \param forward Direction of the iteration (true = forward)
    ClusterIterator(const ClusterFactory& factory, int clusterIndex, bool forward);

    /// \brief Destructor
    ~ClusterIterator() = default;

    /// \brief Check for equalness
    /// \param rhs Iterator to compare to
    /// \return True if iterators are the same, false otherwise
    ///
    /// Check is done on same event factory, event ID and direction
    bool operator==(const ClusterIterator& rhs) const;

    /// \brief Check for not equalness
    /// \param rhs Iterator to compare to
    /// \return True if iterators are different, false otherwise
    ///
    /// Check is done on same event factory, event ID and direction
    bool operator!=(const ClusterIterator& rhs) const { return !(*this == rhs); }

    /// \brief Prefix incrementation operator
    /// \return Iterator after incrementation
    ClusterIterator& operator++();

    /// \brief Postfix incrementation operator
    /// \return Iterator before incrementation
    ClusterIterator operator++(int);

    /// \brief Prefix decrementation operator
    /// \return Iterator after decrementation
    ClusterIterator& operator--();

    /// \brief Postfix decrementation operator
    /// \return Iterator before decrementation
    ClusterIterator operator--(int);

    /// \brief Get pointer to the current cluster
    /// \return Pointer to the current event
    AnalysisCluster* operator*() { return &mCurrentCluster; }

    /// \brief Get reference to the current cluster
    /// \return Reference to the current event of the iterator
    AnalysisCluster& operator&() { return mCurrentCluster; }

    /// \brief Get the index of the current event
    /// \return Index of the current event
    int current_index() const { return mClusterID; }

   private:
    const ClusterFactory& mClusterFactory; ///< Event factory connected to the iterator
    AnalysisCluster mCurrentCluster;       ///< Cache for current cluster
    int mClusterID = 0;                    ///< Current cluster ID within the cluster factory
    bool mForward = true;                  ///< Iterator direction (forward or backward)
  };

  ///
  /// Dummy constructor
  ClusterFactory() = default;

  ///
  /// \brief Constructor initializing the ClusterFactory
  /// \param clustersContainer cluster container
  /// \param inputsContainer cells/digits container
  /// \param cellsIndices for cells/digits indices
  ClusterFactory(gsl::span<const o2::emcal::Cluster> clustersContainer, gsl::span<const InputType> inputsContainer, gsl::span<const int> cellsIndices);

  ///
  /// Copy constructor
  ClusterFactory(const ClusterFactory& rp) = default;

  ///
  /// Assignment operator
  ClusterFactory& operator=(const ClusterFactory& cf) = default;

  ///
  /// Destructor
  ~ClusterFactory() = default;

  /// \brief Get forward start iterator
  /// \return Start iterator
  ClusterIterator begin() const { return ClusterIterator(*this, 0, true); }

  /// \brief Get forward end iteration marker
  /// \return Iteration end marker
  ClusterIterator end() const { return ClusterIterator(*this, getNumberOfClusters(), true); }

  /// \brief Get backward start iterator
  /// \return Start iterator
  ClusterIterator rbegin() const { return ClusterIterator(*this, getNumberOfClusters() - 1, false); };

  /// \brief Get backward end iteration marker
  /// \return Iteration end marker
  ClusterIterator rend() const { return ClusterIterator(*this, -1, false); };

  /// \brief Reset containers
  void reset();

  ///
  /// evaluates cluster parameters: position, shower shape, primaries ...
  AnalysisCluster buildCluster(int index) const;

  void SetECALogWeight(Float_t w) { mLogWeight = w; }
  float GetECALogWeight() const { return mLogWeight; }

  void doEvalLocal2tracking(bool justCluster)
  {
    mJustCluster = justCluster;
  }

  ///
  /// Calculates the center of gravity in the local EMCAL-module coordinates
  void evalLocalPosition(gsl::span<const int> inputsIndices, AnalysisCluster& cluster) const;

  ///
  /// Calculates the center of gravity in the global ALICE coordinates
  void evalGlobalPosition(gsl::span<const int> inputsIndices, AnalysisCluster& cluster) const;

  void evalLocal2TrackingCSTransform() const;

  ///
  /// evaluates local position of clusters in SM
  void evalLocalPositionFit(Double_t deff, Double_t w0, Double_t phiSlope, gsl::span<const int> inputsIndices, AnalysisCluster& cluster) const;

  ///
  /// Applied for simulation data with threshold 3 adc
  /// Calculate efective distance (deff) and weigh parameter (w0)
  /// for coordinate calculation; 0.5 GeV < esum <100 GeV.
  /// Look to:  http://rhic.physics.wayne.edu/~pavlinov/ALICE/SHISHKEBAB/RES/CALIB/GEOMCORR/deffandW0VaEgamma_2.gif
  static void getDeffW0(const Double_t esum, Double_t& deff, Double_t& w0);

  ///
  /// Finds the maximum energy in the cluster
  /// \param  inputsIndices array for the clusters indices contributing to cluster
  /// \return the index of the cells with max enegry
  /// \return the maximum energy
  /// \return the total energy of the cluster
  std::tuple<int, float, float> getMaximalEnergyIndex(gsl::span<const int> inputsIndices) const;

  ///
  /// Calculates the multiplicity of digits/cells with energy larger than level*energy
  int getMultiplicityAtLevel(float level, gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const;

  int getSuperModuleNumber() const { return mSuperModuleNumber; }

  // searches for the local maxima
  // energy above relative level
  //int getNumberOfLocalMax(int nInputMult,
  //                        float locMaxCut, gsl::span<InputType> inputs) const;

  //int getNumberOfLocalMax(std::vector<InputType>& maxAt, std::vector<float>& maxAtEnergy,
  //                        float locMaxCut, gsl::span<InputType> inputs) const;

  bool sharedCluster() const { return mSharedCluster; }
  void setSharedCluster(bool s) { mSharedCluster = s; }

  ///
  /// \param  e: energy in GeV)
  /// \param  key: = 0(gamma, default); !=  0(electron)
  Double_t tMaxInCm(const Double_t e = 0.0, const int key = 0) const;

  bool getCoreRadius() const { return mCoreRadius; }
  void setCoreRadius(float radius) { mCoreRadius = radius; }

  void setClustersContainer(gsl::span<const o2::emcal::Cluster> clusterContainer)
  {
    mClustersContainer = clusterContainer;
  }

  void setCellsContainer(gsl::span<const InputType> cellContainer)
  {
    mInputsContainer = cellContainer;
  }

  void setCellsIndicesContainer(gsl::span<const int> indicesContainer)
  {
    mCellsIndices = indicesContainer;
  }

  int getNumberOfClusters() const
  {
    return mClustersContainer.size();
  }

 protected:
  ///
  /// This function calculates energy in the core,
  /// i.e. within a radius rad = mCoreRadius around the center. Beyond this radius
  /// in accordance with shower profile the energy deposition
  /// should be less than 2%
  /// Unfinished - Nov 15,2006
  /// Distance is calculate in (phi,eta) units
  void evalCoreEnergy(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const;

  ///
  /// Calculates the dispersion of the shower at the origin of the cluster
  /// in cell units
  void evalDispersion(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const;

  ///
  /// Calculates the axis of the shower ellipsoid in eta and phi
  /// in cell units
  void evalElipsAxis(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const;

  ///
  /// Time is set to the time of the digit with the maximum energy
  void evalTime(gsl::span<const int> inputsIndices, AnalysisCluster& clusterAnalysis) const;

  ///
  /// Converts Theta (Radians) to Eta (Radians)
  float thetaToEta(float arg) const;

  ///
  /// Converts Eta (Radians) to Theta (Radians)
  float etaToTheta(float arg) const;

 private:
  o2::emcal::Geometry* mGeomPtr = Geometry::GetInstanceFromRunNumber(223409); //!<! Pointer to geometry for utilities

  float mCoreRadius = 10; ///<  The radius in which the core energy is evaluated

  float mLogWeight = 4.5; ///<  logarithmic weight for the cluster center of gravity calculation

  bool mJustCluster = kFALSE; ///< Flag to evaluates local to "tracking" c.s. transformation (B.P.).

  mutable int mSuperModuleNumber = 0; ///<  number identifying supermodule containing cluster, reference is cell with maximum energy.
  float mDistToBadTower = -1;         ///<  Distance to nearest bad tower
  bool mSharedCluster = false;        ///<  States if cluster is shared by 2 SuperModules in same phi rack (0,1), (2,3) ... (10,11).

  gsl::span<const o2::emcal::Cluster> mClustersContainer; ///< Container for all the clusters in the event
  gsl::span<const InputType> mInputsContainer;            ///< Container for all the cells/digits in the event
  gsl::span<const int> mCellsIndices;                     ///< Container for cells indices in the event

  ClassDefNV(ClusterFactory, 2);
};

} // namespace emcal
} // namespace o2
#endif //ALICEO2_EMCAL_CLUSTERFACTORY_H_