// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_ANALYSISCLUSTER_H_
#define ALICEO2_EMCAL_ANALYSISCLUSTER_H_

#include "FairLogger.h"
#include <gsl/span>
#include <array>
#include "Rtypes.h"
#include "MathUtils/Cartesian3D.h"
#include "TLorentzVector.h"

namespace o2
{

namespace emcal
{

/// \class AnalysisCluster
/// \brief Cluster class for kinematic cluster parameters
/// \ingroup EMCALDataFormat
/// ported from AliVCluster in AliRoot
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since March 05, 2020
///

class AnalysisCluster
{

 public:
  /// \class CellOutOfRangeException
  /// \brief Exception handling non-existing cell indices
  /// \ingroup EMCALbase
  class CellOutOfRangeException : public std::exception
  {
   public:
    /// \brief Constructor, setting cell wrong cell index raising the exception
    /// \param cellIndex Cell index raising the exception
    CellOutOfRangeException(Int_t cellIndex) : std::exception(),
                                               mCellIndex(cellIndex),
                                               mMessage("Cell index " + std::to_string(mCellIndex) + " out of range.")
    {
    }

    /// \brief Destructor
    ~CellOutOfRangeException() noexcept final = default;

    /// \brief Access to cell ID raising the exception
    /// \return Cell ID
    Int_t getCellIndex() const noexcept { return mCellIndex; }

    /// \brief Access to error message of the exception
    /// \return Error message
    const char* what() const noexcept final { return mMessage.data(); }

   private:
    Int_t mCellIndex;     ///< Cell index raising the exception
    std::string mMessage; ///< error Message
  };

  AnalysisCluster() = default;
  ~AnalysisCluster() = default;
  AnalysisCluster(const AnalysisCluster& clus) = default;
  AnalysisCluster& operator=(const AnalysisCluster& source) = default;
  void clear();

  // Common EMCAL/PHOS/FMD/PMD

  void setID(int id) { mID = id; }
  int getID() const { return mID; }

  void setE(double ene) { mEnergy = ene; }
  double E() const { return mEnergy; }

  void setChi2(double chi2) { mChi2 = chi2; }
  double Chi2() const { return mChi2; }

  ///
  /// Set the cluster global position.
  void setGlobalPosition(Point3D<double> x);
  Point3D<double> getGlobalPosition() const
  {
    return mGlobalPos;
  }

  void setLocalPosition(Point3D<double> x);
  Point3D<double> getLocalPosition() const
  {
    return mLocalPos;
  }

  void setDispersion(double disp) { mDispersion = disp; }
  double getDispersion() const { return mDispersion; }

  void setM20(double m20) { mM20 = m20; }
  double getM20() const { return mM20; }

  void setM02(double m02) { mM02 = m02; }
  double getM02() const { return mM02; }

  void setNExMax(unsigned char nExMax) { mNExMax = nExMax; }
  unsigned char getNExMax() const { return mNExMax; }

  void setEmcCpvDistance(double dEmcCpv) { mEmcCpvDistance = dEmcCpv; }
  double getEmcCpvDistance() const { return mEmcCpvDistance; }
  void setTrackDistance(double dx, double dz)
  {
    mTrackDx = dx;
    mTrackDz = dz;
  }
  double getTrackDx() const { return mTrackDx; }
  double getTrackDz() const { return mTrackDz; }

  void setDistanceToBadChannel(double dist) { mDistToBadChannel = dist; }
  double getDistanceToBadChannel() const { return mDistToBadChannel; }

  void setNCells(int n) { mNCells = n; }
  int getNCells() const { return mNCells; }

  ///
  ///  Set the array of cell indices.
  void setCellsIndices(const std::vector<unsigned short>& array)
  {
    mCellsIndices = array;
  }

  const std::vector<unsigned short>& getCellsIndices() const { return mCellsIndices; }

  ///
  ///  Set the array of cell amplitude fractions.
  ///  Cell can be shared between 2 clusters, here the fraction of energy
  ///  assigned to each cluster is stored. Only in unfolded clusters.
  void setCellsAmplitudeFraction(const std::vector<Double32_t>& array)
  {
    mCellsAmpFraction = array;
  }
  const std::vector<Double32_t>& getCellsAmplitudeFraction() const { return mCellsAmpFraction; }

  int getCellIndex(int i) const
  {
    if (i >= 0 && i < mNCells)
      return mCellsIndices[i];
    else
      throw CellOutOfRangeException(i);
  }

  double getCellAmplitudeFraction(int i) const
  {
    if (i >= 0 && i < mNCells)
      return mCellsAmpFraction[i];
    else
      throw CellOutOfRangeException(i);
  }

  bool getIsExotic() const { return mIsExotic; }
  void setIsExotic(bool b) { mIsExotic = b; }

  void setClusterTime(float time)
  {
    mTime = time;
  }

  float getClusterTime() const
  {
    return mTime;
  }

  int getIndMaxInput() const { return mInputIndMax; }
  void setIndMaxInput(const int ind) { mInputIndMax = ind; }

  float getCoreEnergy() const { return mCoreEnergy; }
  void setCoreEnergy(float energy) { mCoreEnergy = energy; }

  ///
  /// Returns TLorentzVector with momentum of the cluster. Only valid for clusters
  /// identified as photons or pi0 (overlapped gamma) produced on the vertex
  /// Vertex can be recovered with esd pointer doing:
  TLorentzVector getMomentum(std::array<const double, 3> vertexPosition) const;

 protected:
  /// TODO to replace later by o2::MCLabel when implementing the MC handling
  std::vector<int> mLabels; ///< List of MC particles that generated the cluster, ordered in deposited energy.

  int mNCells = 0; ///< Number of cells in cluster.

  /// Array of cell indices contributing to this cluster.
  std::vector<unsigned short> mCellsIndices; //[mNCells]

  /// Array with cell amplitudes fraction. Only usable for unfolded clusters, where cell can be shared.
  /// here we store what fraction of the cell energy is assigned to a given cluster.
  std::vector<Double32_t> mCellsAmpFraction; //[mNCells][0.,1.,16]

  Point3D<double> mGlobalPos; ///< Position in global coordinate system (cm).
  Point3D<double> mLocalPos;  ///< Local  position in the sub-detector coordinate
  Double32_t mEnergy = 0;     ///< Energy measured by calorimeter in GeV.
  float mCoreEnergy = 0.;     ///<  Energy in a shower core
  Double32_t mDispersion = 0; ///< Cluster shape dispersion.
  Double32_t mChi2 = 0;       ///< Chi2 of cluster fit (unfolded clusters)
  Double32_t mM20 = 0;        ///< 2-nd moment along the second eigen axis.
  Double32_t mM02 = 0;        ///< 2-nd moment along the main eigen axis.

  Double32_t mEmcCpvDistance = 1024; ///< the distance from PHOS EMC rec.point to the closest CPV rec.point.

  Double32_t mTrackDx = 1024; ///< Distance to closest track in phi.
  Double32_t mTrackDz = 1024; ///< Distance to closest track in z.

  Double32_t mDistToBadChannel = 1024; ///< Distance to nearest bad channel.

  int mID = 0;               ///< Unique Id of the cluster.
  unsigned char mNExMax = 0; ///< Number of Local (Ex-)maxima before unfolding.

  /// Cluster time-of-flight
  double mTOF = 0.0; //[0,0,12]

  float mTime = 0.; ///<  Time of the digit/cell with maximal energy deposition

  bool mIsExotic = false; //!<! Cluster marked as "exotic" (high energy deposition concentrated in a single cell)

  int mInputIndMax = -1; ///<  index of digit/cell with max energy

  ClassDefNV(AnalysisCluster, 1);
};

} // namespace emcal
} // namespace o2
#endif //ANALYSISCLUSTER_H