#include "ITSMFTReconstruction/TopologyDictionary.h"

using std::cout;
using std::endl;

namespace o2
{
namespace ITSMFT
{
std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dict)
{
  for(auto &p : dict.mVectorOfGroupIDs){
    os << p.hash << " " << p.errX << " " << p.errZ << " " << p.frequency << std::endl;
  }
  return os;
}

void TopologyDictionary::WriteBinaryFile(string outputfile){
  std::ofstream file_output(outputfile, std::ios::out | std::ios::binary);
  for(auto &p : mVectorOfGroupIDs){
    file_output.write(reinterpret_cast<char *>(&p.hash),sizeof(unsigned long));
    file_output.write(reinterpret_cast<char *>(&p.errX),sizeof(float));
    file_output.write(reinterpret_cast<char *>(&p.errZ),sizeof(float));
    file_output.write(reinterpret_cast<char *>(&p.frequency),sizeof(double));
  }
  file_output.close();
}

void TopologyDictionary::ReadFile(string fname){
  mVectorOfGroupIDs.clear();
  mFinalMap.clear();
  std::ifstream in(fname);
  GroupStruct gr;
  int groupID=0;
  if(!in.is_open()){
    cout << "The file could not be opened" << endl;
    exit(1);
  }
  else{
    while(in >> gr.hash >> gr.errX >> gr.errZ >> gr.frequency){
      mVectorOfGroupIDs.push_back(gr);
      if(((gr.hash)&0xffffffff) != 0) mFinalMap.insert(std::make_pair(gr.hash,groupID));
      groupID++;
    }
  }
  in.close();
}

void TopologyDictionary::ReadBinaryFile(string fname){
  mVectorOfGroupIDs.clear();
  mFinalMap.clear();
  std::ifstream in(fname.data(),std::ios::in | std::ios::binary);
  GroupStruct gr;
  int groupID=0;
  if(!in.is_open()){
    cout << "The file could not be opened" << endl;
    exit(1);
  }
  else{
    while(in.read(reinterpret_cast<char*>(&gr.hash), sizeof(unsigned long))){
      in.read(reinterpret_cast<char*>(&gr.errX), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.errZ), sizeof(float));
      in.read(reinterpret_cast<char*>(&gr.frequency), sizeof(double));
      mVectorOfGroupIDs.push_back(gr);
      if(((gr.hash)&0xffffffff) != 0) mFinalMap.insert(std::make_pair(gr.hash,groupID));
      groupID++;
    }
  }
  in.close();
}
}
}
