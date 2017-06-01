// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FILESINK_H_
#define FILESINK_H_

#include <FairMQDevice.h>

class TFile;
class TTree;
class TObject;
class TFolder;

namespace o2 {

namespace MFT {

class FileSink : public FairMQDevice
{

 public:

  FileSink();
  ~FileSink() override;
  
  void setOutputFileName(std::string tempString) { mFileName = tempString; }
  void addOutputBranch  (std::string classString, std::string branchString) 
  { 
    mClassNames .push_back(classString); 
    mBranchNames.push_back(branchString); 
    LOG(INFO) << "AddOutput class " << classString.c_str() << " branch " << branchString.c_str() << ""; 
  }
  std::string getOutputFileName () { return mFileName;}

  void setInputChannelName (std::string tstr) {mInputChannelName = tstr;}
  void setAckChannelName(std::string tstr) {mAckChannelName = tstr;}

 protected:

  bool storeData(FairMQParts&, int);
  void Init() override;

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
  /* FairEventHeader* mEventHeader; */
  /* TClonesArray*    mOutput; */
  TFolder* mFolder;
  
  FileSink(const FileSink&);
  FileSink& operator=(const FileSink&);

};
 
}
}

#endif
