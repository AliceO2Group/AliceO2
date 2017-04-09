/// \file HwClusterFinder.h
/// \brief Class for TPC HW cluster finder
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_HWClusterFinder_H_
#define ALICEO2_TPC_HWClusterFinder_H_

#include <vector>

namespace o2{
namespace TPC {

class HwCluster;
    
/// \class HwClusterFinder
/// \brief Class for TPC HW cluster finder
class HwClusterFinder {
  public:
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
    bool AddTimebin(float* timebin, unsigned globalTime, int length = 8);

    /// Add multiple timebins at once
    /// \param nBins Number of timebins
    /// \param timebins 2D array with new charges
    /// \param globalTime Global time of this timebin
    /// \param length Size of array "timebin"
    bool AddTimebins(int nBins, float** timebins, unsigned globalTimeOfLast, int length = 8);

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
    bool  getAutProcessing() const                { return mAutoProcessing; } 
    bool  getmAssignChargeUnique() const          { return mAssignChargeUnique; }
    HwClusterFinder*          getNextCF() const     { return mNextCF; }
    std::vector<HwCluster>*   getClusterContainer() { return &clusterContainer; }

    // Setter functions
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
    void clearClusterContainer()        { clusterContainer.clear(); }

    /// Process the cluster finding
    bool findCluster();

    /// Neighboring cluster finder can inform about already used charges
    /// \param time Time bin of found cluster peak
    /// \param pad Pad of found cluster peak
    /// \param cluster Cluster charges
    void clusterAlreadyUsed(short time, short pad, float** cluster);

  private:

    float chargeForCluster(float* charge, float* toCompare);
    void printCluster(short time, short pad);

    // local variables
    std::vector<HwCluster> clusterContainer;
    int mTimebinsAfterLastProcessing;
    float** mData;
    float** tmpCluster;
    float*  mZeroTimebin;


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
}
}


#endif 
