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

  enum
  {
    OutputFileName = FairMQDevice::Last,
    Last
  };
  
  FileSink();
  virtual ~FileSink();
  
  void SetProperty(const int key, const std::string& value);
  void SetOutputFileName(std::string tempString) { fFileName = tempString; }
  void AddOutputBranch  (std::string classString, std::string branchString) 
  { 
    fClassNames .push_back(classString); 
    fBranchNames.push_back(branchString); 
    LOG(INFO) << "AddOutput class " << classString.c_str() << " branch " << branchString.c_str() << ""; 
  }
  std::string GetOutputFileName () { return fFileName;}
  void SetInputChannelName (std::string tstr) {fInputChannelName = tstr;}
  void SetAckChannelName(std::string tstr) {fAckChannelName = tstr;}

 protected:

  virtual void Init();
  virtual void Run();

 private:

  std::string     fInputChannelName;
  std::string     fAckChannelName;
  
  std::string fFileName;
  std::string fTreeName;
  
  std::vector<std::string> fBranchNames;
  std::vector<std::string> fClassNames;
  
  std::string fFileOption;
  bool fFlowMode;
  bool fWrite;
  
  TFile* fOutFile;
  TTree* fTree;
  unsigned int    fNObjects;
  TObject**       fOutputObjects;
  /* FairEventHeader* fEventHeader; */
  /* TClonesArray*    fOutput; */
  TFolder* fFolder;
  
  FileSink(const FileSink&);
  FileSink& operator=(const FileSink&);

};
 
}
}

#endif
