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

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#include <utility>
#include <cstring>

namespace o2{
namespace TPC {

class Cluster;

/// \class HwClusterFinder
/// \brief Class for TPC HW cluster finder
class HwClusterFinder {
  public:
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
    /// \param id ID for the cluster finder
    /// \param padOffset Offset in pad direction of the cluster finder
    /// \param pad Number of pads
    /// \param timebins Number of timebins
    /// \param diffThreshold Minimum charge difference at neighboring pads
    /// \param chargeThreshold Minimum charge of cluster peak
    /// \param requirePositiveCharge Charge >0 required
    HwClusterFinder(short cru, short row, short id,
        short padOffset, short pads=8, short timebins=8,
        float diffThreshold=0, float chargeThreshold=5, bool requirePositiveCharge=true);

    /// Destructor
    ~HwClusterFinder();

    /// Copy constructor
    HwClusterFinder(const HwClusterFinder& other);

    /// Add a new timebin to cluster finder
    /// \param timebin Array of size "length" with new charges
    /// \param globalTime Global time of this timebin
    /// \param length Size of array "timebin"
    bool AddTimebin(MiniDigit* timebin, unsigned globalTime, int length = 8);

//    /// Add multiple timebins at once
//    /// \param nBins Number of timebins
//    /// \param timebins 2D array with new charges
//    /// \param globalTime Global time of this timebin
//    /// \param length Size of array "timebin"
//    bool AddTimebins(int nBins, float** timebins, unsigned globalTimeOfLast, int length = 8);

    /// Add a timebin with charges of 0
    /// \param globalTime Global time of this timebin
    /// \param length Size of array "timebin"
    void AddZeroTimebin(unsigned globalTime = 0, int lengt = 8);

    /// Print the local storagae of charges
    void PrintLocalStorage();

    /// Resets the local storage to zeros
    /// \param globalTimeAfterReset Global time of the first timebin after reset
    void reset(unsigned globalTimeAfterReset);


    // Getter functions
    int   getGlobalTimeOfLast() const             { return mGlobalTimeOfLast; }
    int   getTimebinsAfterLastProcessing() const  { return mTimebinsAfterLastProcessing; };
    short getCRU() const                          { return mCRU; }
    short getRow() const                          { return mRow; }
    short getId() const                           { return mId; }
    short getPadOffset() const                    { return mPadOffset; }
    short getNpads() const                        { return mPads; }
    short getNtimebins() const                    { return mTimebins; }
    short getClusterSizeP() const                 { return mClusterSizePads; }
    short getClusterSizeT() const                 { return mClusterSizeTime; }
    float getDiffThreshold() const                { return mDiffThreshold; }
    float getChargeThreshold() const              { return mChargeThreshold; }
    bool  getRequirePositiveCharge() const        { return mRequirePositiveCharge; }
    bool  getRequireNeighbouringPad() const       { return mRequireNeighbouringPad; }
    bool  getRequireNeighbouringTimebin() const   { return mRequireNeighbouringTimebin; }
    bool  getAutoProcessing() const               { return mAutoProcessing; }
    bool  getmAssignChargeUnique() const          { return mAssignChargeUnique; }
    HwClusterFinder* getNextCF() const            { return mNextCF; }
    std::vector<Cluster>* getClusterContainer()   { return &clusterContainer; }
    std::vector<std::vector<std::pair<int,int>>>* getClusterDigitIndices()    { return &clusterDigitIndices; }

    // Setter functions
    void  setTimebinsAfterLastProcessing(int val)       { mTimebinsAfterLastProcessing = val; };
    void  setCRU(short val)                             { mCRU = val; }
    void  setRow(short val)                             { mRow = val; }
    void  setId(short val)                              { mId = val; }
    void  setPadOffset(short val)                       { mPadOffset = val; }
    void  setNpads(short val)                           { mPads = val; }
    void  setNtimebins(short val)                       { mTimebins = val; }
    void  setClusterSizeP(short val)                    { mClusterSizePads = val; }
    void  setClusterSizeT(short val)                    { mClusterSizeTime = val; }
    void  setDiffThreshold(float val)                   { mDiffThreshold = val; }
    void  setChargeThreshold(float val)                 { mChargeThreshold = val; }
    void  setRequirePositiveCharge(bool val)            { mRequirePositiveCharge = val; }
    void  setRequireNeighbouringPad(bool val)           { mRequireNeighbouringPad = val; }
    void  setRequireNeighbouringTimebin(bool val)       { mRequireNeighbouringTimebin = val; }
    void  setAutoProcessing(bool val)                   { mAutoProcessing = val; }
    void  setAssignChargeUnique(bool val)               { mAssignChargeUnique = val; }
    void  setNextCF(HwClusterFinder* nextCF);


    /// Clears the local cluster storage
    void clearClusterContainer() { clusterContainer.clear(); clusterDigitIndices.clear(); }

    /// Process the cluster finding
    bool findCluster();

    /// Neighboring cluster finder can inform about already used charges
    /// \param time Time bin of found cluster peak
    /// \param pad Pad of found cluster peak
    /// \param cluster Cluster charges
    void clusterAlreadyUsed(short time, short pad, MiniDigit** cluster);

  private:

    MiniDigit chargeForCluster(MiniDigit* charge, MiniDigit* toCompare);
    void printCluster(short time, short pad);

    // local variables
    std::vector<o2::TPC::Cluster> clusterContainer;
    std::vector<std::vector<std::pair<int,int>>> clusterDigitIndices;
    int mTimebinsAfterLastProcessing;
    MiniDigit** mData;
    MiniDigit** tmpCluster;
    MiniDigit*  mZeroTimebin;


    // configuration
    int mGlobalTimeOfLast;
    short mCRU;
    short mRow;
    short mId;
    short mPadOffset;
    short mPads;
    short mTimebins;
    short mClusterSizePads;
    short mClusterSizeTime;
    float mDiffThreshold;
    float mChargeThreshold;
    bool mRequirePositiveCharge;
    bool mRequireNeighbouringPad;
    bool mRequireNeighbouringTimebin;
    bool mAutoProcessing;
    bool mAssignChargeUnique;

    HwClusterFinder* mNextCF;

};

//________________________________________________________________________
inline bool HwClusterFinder::AddTimebin(MiniDigit* timebin, unsigned globalTime, int length)
{
  mGlobalTimeOfLast = globalTime;
  ++mTimebinsAfterLastProcessing;

  //
  // reordering of the local arrays
  //
  MiniDigit* data0 = mData[0];
  std::memmove(mData,mData+1,(mTimebins-1)*sizeof(mData[0]));
  mData[mTimebins-1] = data0;
  if (length < mPads) {
    std::memset(*(mData+mTimebins-1)+length,0 ,(mPads-length)*sizeof(mData[mTimebins-1][0]));
    std::memcpy(*(mData+mTimebins-1),timebin,length*sizeof(timebin[0]));
  } else {
    std::memcpy(*(mData+mTimebins-1),timebin,mPads*sizeof(timebin[0]));
  }

  if (mAutoProcessing & (mTimebinsAfterLastProcessing >= (mTimebins -2 -2))) findCluster();
  return true;
}

}
}


#endif
