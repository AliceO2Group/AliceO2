#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include <cmath>

using namespace std;

namespace o2
{
namespace ITSMFT
{
  BuildTopologyDictionary::BuildTopologyDictionary():mTotClusters(0){}

#ifndef _STUDY_
  void BuildTopologyDictionary::accountTopology(const std::string &cluster){
    mTotClusters++;
    fTtop.setPattern(cluster);
    //pair<unordered_map<unsigned long, pair<ClusterTopology,unsigned long>>::iterator,bool> ret;
    auto ret = mTopologyMap.insert(make_pair(mTopology.getHash(),make_pair(mTopology,1)));
    if(ret.second==false) ret.first->second.second++;
  }
#else
  void BuildTopologyDictionary::accountTopology(const std::string &cluster, float dX, float dZ){
    mTotClusters++;
    mTopology.setPattern(cluster);
    //pair<unordered_map<unsigned long, pair<ClusterTopology,unsigned long>>::iterator,bool> ret;
    auto ret = mTopologyMap.insert(make_pair(mTopology.getHash(),make_pair(mTopology,1)));
    if(ret.second==true){
      //___________________DEFINING_TOPOLOGY_CHARACTERISTICS__________________
      TopologyInfo topInf;
      int &rs = topInf.sizeX = mTopology.getRowSpan();
      int &cs = topInf.sizeZ = mTopology.getColumnSpan();
      //__________________COG_Deterrmination_____________
      int tempyCOG = 0;
      int tempzCOG = 0;
      int tempFiredPixels = 0;
      unsigned char tempChar = 0;
      int s = 0;
      int ic = 0;
      int ir = 0;
      for(unsigned int i=2; i<mTopology.getPattern().length(); i++){
        tempChar = mTopology.getPattern()[i];
        s=128;//0b10000000
        while(s>0){
          if((tempChar&s)!=0){
            tempFiredPixels++;
            tempyCOG+=ir;
            tempzCOG+=ic;
          }
          ic++;
          s/=2;
          if((ir+1)*ic==(rs*cs)) break;
          if(ic==cs){
            ic=0;
            ir++;
          }
        }
        if((ir+1)*ic==(rs*cs)) break;
      }
      topInf.xCOG = 0.5 + (float)tempyCOG/(float)tempFiredPixels;
      topInf.zCOG = 0.5 + (float)tempzCOG/(float)tempFiredPixels;
      topInf.nPixels = tempFiredPixels;
      topInf.xMean = dX;
      topInf.xSigma2 = 0;
      topInf.zMean = dZ;
      topInf.zSigma2 = 0;
      fMapInfo.insert(make_pair(mTopology.getHash(),topInf));
    }
    else{
      int num = (ret.first->second.second++);
      auto ind = fMapInfo.find(mTopology.getHash());
      float tmpxMean = ind->second.xMean;
      float newxMean = ind->second.xMean = ( (tmpxMean)*num + dX ) / (num+1);
      float tmpxSigma2 = ind->second.xSigma2;
      ind->second.xSigma2 = ( num*tmpxSigma2 + (dX - tmpxMean)*(dX - newxMean) ) / (num+1); //online variance algorithm
      float tmpzMean = ind->second.zMean;
      float newzMean = ind->second.zMean = ( (tmpzMean)*num + dZ ) / (num+1);
      float tmpzSigma2 = ind->second.zSigma2;
      ind->second.zSigma2 = ( num*tmpzSigma2 + (dZ - tmpzMean)*(dZ - newzMean) ) / (num+1); //online variance algorithm
    }
  }
#endif

unsigned long BuildTopologyDictionary::checkHash(const std::string &clust){
  mTopology.setPattern(clust);
  return mTopology.getHash();
}

void BuildTopologyDictionary::setThreshold(double thr){
  for(auto &&p : mTopologyMap){
    mTopologyFrequency.push_back(make_pair(p.second.second,p.first));
  }
  std::sort(mTopologyFrequency.begin(),mTopologyFrequency.end(), [] (const pair<unsigned long, unsigned long> &couple1, const pair<unsigned long, unsigned long> &couple2){return (couple1.first > couple2.first);});
  mNotInGroups = 0;
  mNumberOfGroups = 0;
  mDictionary.mFinalMap.clear();
  mFrequencyThreshold=thr;
  for(auto &q : mTopologyFrequency){
    if( ((double)q.first)/mTotClusters > thr ) mNotInGroups++;
    else break;
  }
  mNumberOfGroups=mNotInGroups;
}

void BuildTopologyDictionary::setNGroups(unsigned int ngr){
  for(auto &&p : mTopologyMap){
    mTopologyFrequency.push_back(make_pair(p.second.second,p.first));
  }
  std::sort(mTopologyFrequency.begin(),mTopologyFrequency.end(), [] (const pair<unsigned long, unsigned long> &couple1, const pair<unsigned long, unsigned long> &couple2){return (couple1.first > couple2.first);});
  if(ngr<10 || ngr > (mTopologyFrequency.size()-49)){
    cout << "BuildTopologyDictionary::setNGroups : Invalid number of groups" << endl;
    exit(1);
  }
  mNumberOfGroups = mNotInGroups = ngr-49;
  mDictionary.mFinalMap.clear();
  mFrequencyThreshold=((double)mTopologyFrequency[mNotInGroups-1].first)/mTotClusters;
}

void BuildTopologyDictionary::setThresholdCumulative(double cumulative){
  cout<<"setThresholdCumulative: mTotClusters: " << mTotClusters << endl;
  if(cumulative<=0. || cumulative >=1.) cumulative = 0.99;
  double totFreq = 0.;
  for(auto &&p : mTopologyMap){
    mTopologyFrequency.push_back(make_pair(p.second.second,p.first));
  }
  std::sort(mTopologyFrequency.begin(),mTopologyFrequency.end(), [] (const pair<unsigned long, unsigned long> &couple1, const pair<unsigned long, unsigned long> &couple2){return (couple1.first > couple2.first);});
  mNotInGroups = 0;
  mNumberOfGroups = 0;
  mDictionary.mFinalMap.clear();
  for(auto &q : mTopologyFrequency){
    totFreq += ((double)(q.first))/mTotClusters;
    if(totFreq<cumulative){
      mNotInGroups++;
    }
    else break;
  }
  mFrequencyThreshold=((double)(mTopologyFrequency[--mNotInGroups].first))/mTotClusters;
  while(((double)mTopologyFrequency[mNotInGroups].first)/mTotClusters == mFrequencyThreshold) mNotInGroups--;
  mFrequencyThreshold=((double)mTopologyFrequency[mNotInGroups++].first)/mTotClusters;
  mNumberOfGroups=mNotInGroups;
}

void BuildTopologyDictionary::groupRareTopologies(){

  cout<<"groupRareTopologies: mTotClusters: " << mTotClusters << endl;
  #ifdef _HISTO_
    mHdist = TH1F("mHdist", "Groups distribution", mNumberOfGroups+49, -0.5, mNumberOfGroups+48.5);
    mHdist.getXaxis()->setTitle("GroupID");
    mHdist.setFillColor(kRed);
    mHdist.setFillStyle(3005);
  #endif

  double totFreq=0.;
  for(int j=0; j<mNotInGroups; j++){
    #ifdef _HISTO_
      mHdist.Fill(j,mTopologyFrequency[j].first);
    #endif
    totFreq+=((double)(mTopologyFrequency[j].first))/mTotClusters;
    GroupStruct gr;
    gr.hash=mTopologyFrequency[j].second;
    gr.frequency=totFreq;
    //rough estimation for the error considering a uniform distribution
    gr.errX = std::sqrt(fMapInfo.find(gr.hash)->second.xSigma2);
    gr.errZ = std::sqrt(fMapInfo.find(gr.hash)->second.zSigma2);
    mDictionary.mVectorOfGroupIDs.push_back(gr);
    mDictionary.mFinalMap.insert(make_pair(gr.hash,j));
  }
  //groupRareTopologies based on binning over number of rows and columns (7*7)
  mNumberOfGroups+=49; //(7*7)
  //array of groups
  std::array<GroupStruct,49> GroupArray;
  std::array<unsigned long,49> groupCounts{0};
  auto func = [&GroupArray] (int rowBinEdge, int colBinEdge, int &index) {
    unsigned long provvHash = 0;
    provvHash = ( ((unsigned long)(index+1)) << 32 ) & 0xffffffff00000000;
    GroupArray[index].hash = provvHash;
    GroupArray[index].errX = (rowBinEdge)*2e-3/std::sqrt(12); // 2e-3 is the pitch
    GroupArray[index].errZ = (colBinEdge)*2e-3/std::sqrt(12); // 2e-3 is the pitch
    index++;
    return;
  };
  int grNum=0;
  for(int ir=0; ir<6; ir++){ //row bins: {[0;4],[5;9],[10;14],[15;19],[20;24],[25,29]} (+ [30;32] later)
    for(int ic=0; ic<6; ic++){ //col bins: {[0;4],[5;9],[10;14],[15;19],[20;24],[25,29]} (+ [30;32] later)
      func((ir+1)*5-1, (ic+1)*5-1, grNum);
    }
    // col bin [30;32]
    func((ir+1)*5-1, 32, grNum);
  }
  // row bin [30;32]
  for(int ic=0; ic<6; ic++){ //col bins: {[0;4],[5;9],[10;14],[15;19],[20;24],[25,29]} (+ [30;32] later)
    func(32, (ic+1)*5-1, grNum);
    unsigned long provvHash = 0;
  }
  func(32, 32, grNum);
  if(grNum!=49){
    cout << "Wrong number of groups" << endl;
    exit(1);
  }
  int rs;
  int cs;
  int index;

  for(unsigned int j = (unsigned int)mNotInGroups; j<mTopologyFrequency.size(); j++){
    unsigned long
    hash1 = mTopologyFrequency[j].second;
    rs = mTopologyMap.find(hash1)->second.first.getRowSpan();
    cs = mTopologyMap.find(hash1)->second.first.getColumnSpan();
    index = (rs/5)*7 + cs/5;
    if(index >48) index = 48;
    groupCounts[index]+=mTopologyFrequency[j].first;
  }

  for(int i=0; i<49; i++){
    totFreq+=((double)groupCounts[i])/mTotClusters;
    GroupArray[i].frequency = totFreq;
    #ifdef _HISTO_
      mHdist.Fill(mNotInGroups+i,groupCounts[i]);
    #endif
    mDictionary.mVectorOfGroupIDs.push_back(GroupArray[i]);
  }
  #ifdef _HISTO_
    mHdist.Scale(1./mHdist.Integral());
  #endif
}


std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& DB){
  for(int i=0; i<DB.mNotInGroups; i++){
    const unsigned long &hash = DB.mTopologyFrequency[i].second;
    os << "Hash: " << hash << endl;
    os << "counts: " << DB.mTopologyMap.find(hash)->second.second << endl;
    os << "sigmaX: " << std::sqrt(DB.fMapInfo.find(hash)->second.xSigma2) << endl;
    os << "sigmaZ: " << std::sqrt(DB.fMapInfo.find(hash)->second.zSigma2) << endl;
    os << DB.mTopologyMap.find(hash)->second.first;
  }
  return os;
}

void BuildTopologyDictionary::printDictionary(string fname){
  ofstream out(fname);
  out << mDictionary;
  out.close();
}

void BuildTopologyDictionary::printDictionaryBinary(string fname){
  ofstream out(fname);
  mDictionary.WriteBinaryFile(fname);
  out.close();
}
}
}
