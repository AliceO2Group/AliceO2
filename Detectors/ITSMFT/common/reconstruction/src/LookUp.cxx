#include "ITSMFTReconstruction/LookUp.h"

namespace o2
{
namespace ITSMFT
{
LookUp::LookUp(std::string fileName){
  mDictionary.ReadBinaryFile(fileName);
  mTopologiesOverThreshold = mDictionary.mFinalMap.size();
}

int LookUp::findGroupID(const std::string& cluster){
  mTopology.setPattern(cluster);
  auto ret = mDictionary.mFinalMap.find(mTopology.getHash());
  if(ret!=mDictionary.mFinalMap.end()) return ret->second;
  else{
    int rs = mTopology.getRowSpan();
    int cs = mTopology.getColumnSpan();
    int index = (rs/5)*7 + cs/5;
    if(index >48) index = 48;
    return (mTopologiesOverThreshold+index);
  }
}
}
}
