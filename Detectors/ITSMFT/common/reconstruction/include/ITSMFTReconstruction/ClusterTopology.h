#ifndef ALICEO2_ITSMFT_CLUSTERTOPOLOGY_H
#define ALICEO2_ITSMFT_CLUSTERTOPOLOGY_H
#include "ITSMFTReconstruction/Cluster.h"
#include <iostream>
#include <string>

namespace o2
{
namespace ITSMFT
{

void convertClusterToString(const Cluster &cluster, std::string &str);

class ClusterTopology {

  public:
    ClusterTopology();
    ClusterTopology(const std::string &str);

    std::string& getPattern() { return mPattern; }
    int getRowSpan() const { return static_cast<int>(mPattern[0]); }
    int getColumnSpan() const { return static_cast<int>(mPattern[1]); }
    unsigned long getHash() const { return mHash; }
    friend std::ostream& operator<<(std::ostream& os, const ClusterTopology& top);
    static unsigned int hashFunction(const void * key, int len);
    void setPattern(const std::string &str);

  private:
    std::string mPattern;
    unsigned long mHash;

};
}
}

#endif /* ALICEO2_ITS_CLUSTERTOPOLOGY_H */
