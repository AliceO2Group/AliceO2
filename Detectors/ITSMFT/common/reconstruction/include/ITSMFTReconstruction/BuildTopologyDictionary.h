#ifndef ALICEO2_ITSMFT_BUILDTOPOLOGYDICTIONARY_H
#define ALICEO2_ITSMFT_BUILDTOPOLOGYDICTIONARY_H
#include "ITSMFTReconstruction/ClusterTopology.h"
#include "ITSMFTReconstruction/TopologyDictionary.h"
#include <array>
#include <algorithm>

#define _STUDY_
//#define _HISTO_ //in order to have a histogram with the ditribution of groupIDs

#ifdef _HISTO_
  #include "TH1F.h"
#endif

namespace o2
{
namespace ITSMFT
{
#ifdef _STUDY_
  struct TopologyInfo{
    int sizeX;
    int sizeZ;
    float xCOG;
    float zCOG;
    float xMean;
    float xSigma2;
    float zMean;
    float zSigma2;
    int nPixels;
  };
#endif

struct groupTmp{
  unsigned long GroupCounts;
  int tempGroupID;
  vector<unsigned long> GrHashes;
  int groupID;
};

class BuildTopologyDictionary {

  public:

    #ifdef _HISTO_
      TH1F mHdist; //Distribution of groupIDs
    #endif

    #ifndef _STUDY_
      void accountTopology(const std::string &cluster);
    #else
      void accountTopology(const std::string &cluster, float dX, float dZ);
    #endif

    BuildTopologyDictionary();

    void setNGroups(unsigned int ngr); //set number of groups
    void setThreshold(double thr);
    void setThresholdCumulative(double cumulative); //Considering the integral
    void groupRareTopologies();
    friend std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& BD);
    void printDictionary(string fname);
    void printDictionaryBinary(string fname);
    unsigned long checkHash(const std::string& clust);

    int getTotClusters() const {return mTotClusters;}
    int getNotInGroups() const {return mNotInGroups;}
    int getNGroups() const {return mNumberOfGroups;}

  private:
    TopologyDictionary mDictionary;
    unordered_map<unsigned long,std::pair<ClusterTopology,unsigned long>> mTopologyMap; //<hash,<topology,counts>>,
    vector <std::pair<unsigned long,unsigned long>> mTopologyFrequency; //<freq,hash>, needed to define threshold
    int mTotClusters;
    int mNumberOfGroups;
    int mNotInGroups;
    double mFrequencyThreshold;
    #ifdef _STUDY_
      unordered_map<long unsigned,TopologyInfo> fMapInfo;
    #endif
    ClusterTopology mTopology;
};
}
}
#endif
