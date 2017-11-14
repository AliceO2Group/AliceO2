#ifndef ALICEO2_ITSMFT_LOOKUP_H
#define ALICEO2_ITSMFT_LOOKUP_H
#include "ITSMFTReconstruction/TopologyDictionary.h"
#include "ITSMFTReconstruction/ClusterTopology.h"

using std::vector;
using std::unordered_map;

namespace o2
{
namespace ITSMFT
{
class LookUp{
  public:
    LookUp(std::string fileName);
    int findGroupID(const std::string& cluster);
    int getTopologiesOverThreshold() {return mTopologiesOverThreshold;}

  private:
    TopologyDictionary mDictionary;
    int mTopologiesOverThreshold;
    ClusterTopology mTopology;
};
}
}

#endif
