// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HwCluster.h
/// \brief Class to have some more info about the HwClusterer clusters
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_HWCLUSTER_H
#define ALICEO2_TPC_HWCLUSTER_H

#include "TPCReconstruction/Cluster.h"
#include <vector>

namespace boost { namespace serialization { class access; } }

namespace o2 {
namespace TPC{

/// \class HwCluster
/// \brief Class to store HW clusters with cluster data
class HwCluster : public Cluster {
  public:

    /// Default Constructors
    HwCluster();

    /// Constructor
    /// \param sizeP Cluster size in pad direction
    /// \param sizeT Cluster size in time direction
    HwCluster(short sizeP, short sizeT);

    /// Constructor
    /// \param cru CRU
    /// \param row Row
    /// \param sizeP Cluster size in pad direction
    /// \param sizeT Cluster size in time direction
    /// \param clusterData 2D array of size sizeT x sizeP with cluster data
    /// \param maxPad Pad with max charge value
    /// \param maxTime Timebin with max charge value
    HwCluster(short cru, short row, short sizeP, short sizeT, 
        float** clusterData, short maxPad, short maxTime);

    /// Destructor
    ~HwCluster() = default;

    /// Copy Constructor
    /// \param other HwCluster to be copied
    HwCluster(const HwCluster& other);

//    short getPad() const { return mPad; }
//    short getTime() const { return mTime; }
//    short getSizeP() const { return mSizeP; }
//    short getSizeT() const { return mSizeT; }

    /// Set all cluster data
    /// \param cru CRU
    /// \param row Row
    /// \param sizeP Cluster size in pad direction
    /// \param sizeT Cluster size in time direction
    /// \param clusterData 2D array of size sizeT x sizeP with cluster data
    /// \param maxPad Pad with max charge value
    /// \param maxTime Timebin with max charge value
    void setClusterData(short cru, short row, short sizeP, short sizeT, 
                        float** clusterData, short maxPad, short maxTime);

    /// Print function
    /// \param output Stream to put the HwCluster on
    /// \return The output stream
    friend std::ostream& operator<< (std::ostream& out, const HwCluster &c) { return c.print(out); }
    std::ostream& print(std::ostream &output) const;
    std::ostream& PrintDetails(std::ostream &output) const;

  private:
    /// Calculates the cluster properties according to locally stored data
    void calculateClusterProperties();

    short   mPad;       ///< Pad with max charge
    short   mTime;      ///< Timebin with max charge
    short   mSizeP;     ///< Cluster size in pad direction
    short   mSizeT;     ///< Cluster size in time direction
    short   mSize;      ///< Actual size of cluster

    std::vector<std::vector<float>> mClusterData;  ///< CLuster data

    ClassDefNV(HwCluster, 1);
  };
}
}

#endif
