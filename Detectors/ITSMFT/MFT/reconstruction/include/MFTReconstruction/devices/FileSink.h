#ifndef FILESINK_H_
#define FILESINK_H_

#include "FairMQDevice.h"

class TFile;
class TTree;
class TObject;
class TFolder;

namespace AliceO2 {

namespace MFT {

class FileSink : public FairMQDevice
{

 public:

  FileSink();
  virtual ~FileSink();
  
  void SetOutputFileName(std::string tempString) { mFileName = tempString; }
  void AddOutputBranch  (std::string classString, std::string branchString) 
  { 
    mClassNames .push_back(classString); 
    mBranchNames.push_back(branchString); 
    LOG(INFO) << "AddOutput class " << classString.c_str() << " branch " << branchString.c_str() << ""; 
  }
  std::string GetOutputFileName () { return mFileName;}

  void SetInputChannelName (std::string tstr) {mInputChannelName = tstr;}
  void SetAckChannelName(std::string tstr) {mAckChannelName = tstr;}

 protected:

  bool StoreData(FairMQParts&, int);
  virtual void Init();

 private:

  std::string     mInputChannelName;
  std::string     mAckChannelName;
  
  std::string mFileName;
  std::string mTreeName;
  
  std::vector<std::string> mBranchNames;
  std::vector<std::string> mClassNames;
  
  std::string mFileOption;
  bool mFlowMode;
  bool mWrite;
  
  TFile* mOutFile;
  TTree* mTree;
  unsigned int    mNObjects;
  TObject**       mOutputObjects;
  /* FairEventHeader* fEventHeader; */
  /* TClonesArray*    fOutput; */
  TFolder* mFolder;
  
  FileSink(const FileSink&);
  FileSink& operator=(const FileSink&);

};
 
}
}

#endif
