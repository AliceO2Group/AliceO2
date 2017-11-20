// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HwClusterFinder.h
/// \brief Class for TPC HW cluster finder
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_HWClusterFinder_H_
#define ALICEO2_TPC_HWClusterFinder_H_

#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <cstring>

namespace o2{
namespace TPC {

class Cluster;

/// \class HwClusterFinder
/// \brief Class for TPC HW cluster finder
class HwClusterFinder {
  public:
    /// Mini digit struct, consisting of charge, event number and digit index.
    /// The last two are needed to retrieve MC truth information.
    struct MiniDigit {
      float charge;
      int event;
      int index;

      MiniDigit() : charge(0), event(-1), index(-1) {};
      MiniDigit(const MiniDigit& other) : charge(other.charge), event(other.event), index(other.index) {};
      void clear() { charge = 0; event = -1; index = -1; };
    };

    /// Default Constructor
    /// \param cru CRU of this cluster finder
    /// \param row Row of this cluster finder
    /// \param padOffset Offset in pad direction of the cluster finder
    /// \param pad Number of pads
    /// \param timebins Number of time bins
    /// \param diffThreshold Minimum charge difference at neighboring pads
    /// \param chargeThreshold Minimum charge of cluster peak
    /// \param requirePositiveCharge Charge >0 required
    HwClusterFinder(unsigned short cru, unsigned short row,
        short padOffset, unsigned short pads=8, unsigned short timebins=8,
        float diffThreshold=0, float chargeThreshold=5, bool requirePositiveCharge=true);

    /// Destructor
    ~HwClusterFinder() = default;

    /// Add a new time bin to cluster finder
    /// \param timebin Iterator to a vector with data
    /// \param globalTime Global time of this time bin
    /// \param length Number of pads to be used after iterator starts
    /// \param zeroBin Switch to fill timebin with zero's instead
    void addTimebin(std::vector<MiniDigit>::iterator timebin, unsigned globalTime, int length = 8, bool zeroBin = false);

    /// Add a timebin with charges of 0
    /// \param globalTime Global time of this timebin
    /// \param length Size of array "timebin"
    void addZeroTimebin(unsigned globalTime = 0, int lengt = 8);

    /// Print the local storage of charges
    void printLocalStorage();

    /// Resets the local storage to zeros
    /// \param globalTimeAfterReset Global time of the first time bin after reset
    void reset(unsigned globalTimeAfterReset);


    /// Getter function
    /// \return Time of last inserted time bin
    unsigned getGlobalTimeOfLast() const { return mGlobalTimeOfLast; }

    /// Getter function
    /// \return Number of inserted time bins since last processing
    unsigned getTimebinsAfterLastProcessing() const { return mTimebinsAfterLastProcessing; };

    /// Getter function
    /// \return Configured CRU number
    unsigned short getCRU() const { return mCRU; }

    /// Getter function
    /// \return Configured row number
    unsigned short getRow() const { return mRow; }

    /// Getter function
    /// \return Configured pad offset of this CF
    short getPadOffset() const { return mPadOffset; }

    /// Getter function
    /// \return Width of CF in pad direction
    unsigned short getNpads() const { return mPads; }

    /// Getter function
    /// \return Width of CF in time direction
    unsigned short getNtimebins() const { return mTimebins; }

    /// Getter function
    /// \return Pointer to cluster container with all found clusters
    std::vector<Cluster>* getClusterContainer() { return &mClusterContainer; }

    /// Getter function
    /// \return Pointer to container with digits indices used for the clusters
    std::vector<std::vector<std::pair<int,int>>>* getClusterDigitIndices() { return &mClusterDigitIndices; }


    /// Setter function
    /// \param val Number of time bins since last processing
    void setTimebinsAfterLastProcessing(unsigned val) { mTimebinsAfterLastProcessing = val; };

    /// Setter function
    /// \param val Switch whether charge should be used only for one cluster ("charge splitting", TODO: not yet properly implemented)
    void setAssignChargeUnique(bool val) {  mAssignChargeUnique = val; }

    /// Setter function
    /// \param nextCF Pointer to neighboring CF instance (on the "left" side)
    void setNextCF(std::shared_ptr<HwClusterFinder> nextCF) { mNextCF = nextCF; };


    /// Clears the local cluster storage
    void clearClusterContainer() { mClusterContainer.clear(); mClusterDigitIndices.clear(); }

    /// Process the cluster finding
    bool findCluster();

    /// Neighboring cluster finder can inform about already used charges
    /// \param time Time bin of found cluster peak
    /// \param pad Pad of found cluster peak
    void clusterAlreadyUsed(short time, short pad);

  private:

    /// Comparator helper function for two MiniDigits. Checks if outerCharge is positiv (if required) and if innerCharge is above threshold
    /// \param outerCharge Charge of the "outer" pad
    /// \param innerCharge Charge of the "inner" pad
    /// \return MiniDigit to be used for cluster. Will be outerCharge, if requirements fulfilled, otherwise "0-MiniDigit"
    bool chargeForCluster(float outerCharge, float innerCharge);

    /// Prints 5x5 matrix of internal storage around given parameters
    /// \param time relative time bin of center
    /// \param pad relative pad number of center
    void printCluster(short time, short pad);

    /*
     * Class members
     */
    bool mRequirePositiveCharge;            ///< Switch if positive charge is required for individual pad
    bool mRequireNeighbouringPad;           ///< Switch if at least one neighboring pad needs charge > 0
    bool mRequireNeighbouringTimebin;       ///< Switch if at least one neighboring time bin needs charge > 0
    bool mAssignChargeUnique;               ///< Switch for "charge splitting", TODO: not yet properly implemented
    short mPadOffset;                       ///< Pad number in row of leftmost pad (can be negative for leftmost CF)
    unsigned short mCRU;                    ///< CRU number
    unsigned short mRow;                    ///< Row number
    unsigned short mPads;                   ///< Size of CF in pad direction
    unsigned short mTimebins;               ///< Size of CF in time direction
    unsigned short mClusterSizePads;        ///< Size of cluster in pad direction
    unsigned short mClusterSizeTime;        ///< Size of cluster in time direction
    float mDiffThreshold;                   ///< Charge difference threshold, not yet used
    float mChargeThreshold;                 ///< Charge threshold
    unsigned mGlobalTimeOfLast;             ///< Global time of last added time bin
    unsigned mTimebinsAfterLastProcessing;  ///< Number of time bins added after last processing
    std::weak_ptr<HwClusterFinder> mNextCF; ///< Not owning pointer to neighboring cluster finder (on the "left" side)

    std::vector<std::unique_ptr<std::vector<MiniDigit>>> mData;         ///< local data storage
    std::vector<std::vector<MiniDigit>> mTmpCluster;                    ///< local temporary cluster data storage
    std::vector<o2::TPC::Cluster> mClusterContainer;                    ///< Container for found clusters
    std::vector<std::vector<std::pair<int,int>>> mClusterDigitIndices;  ///< Container for digit indices associated with found clusters

};

//________________________________________________________________________
inline void HwClusterFinder::addTimebin(std::vector<MiniDigit>::iterator timebin, unsigned globalTime, int length, bool zeroBin)
{
  mGlobalTimeOfLast = globalTime;
  ++mTimebinsAfterLastProcessing;

  //
  // reordering of the local array
  //
  std::rotate(mData.begin(), mData.begin() + 1, mData.end());

  //
  // fillin with data
  // 
  if (zeroBin) {
    for (auto &digi : *mData.back()) digi.clear();
  } else {
    std::copy(timebin,timebin+length,mData.back()->begin());
    mData.back()->resize(mPads,MiniDigit());
  }
}

}
}


#endif
