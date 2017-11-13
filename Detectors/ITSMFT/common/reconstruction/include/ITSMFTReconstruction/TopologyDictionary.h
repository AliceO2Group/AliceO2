#ifndef ALICEO2_ITSMFT_TOPOLOGYDICTIONARY_H
#define ALICEO2_ITSMFT_TOPOLOGYDICTIONARY_H
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>

using std::vector;
using std::unordered_map;
using std::string;

namespace o2
{
namespace ITSMFT
{

struct GroupStruct{
  unsigned long hash;
  float errX;
  float errZ;
  double frequency;
};

class TopologyDictionary{
  public:
    friend std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dictionary);
    void WriteBinaryFile(string outputFile);
    void ReadFile(string fileName);
    void ReadBinaryFile(string fileName);
  private:
    unordered_map<unsigned long, int> mFinalMap; //<hash,groupID> just for topologies over threshold
    vector<GroupStruct> mVectorOfGroupIDs;
};
}
}

#endif
