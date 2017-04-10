#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <string>

#include <boost/thread.hpp>

#include "FairFileSource.h"
#include "FairRunAna.h"
#include <FairMQDevice.h>

namespace o2 {

namespace MFT {

class Sampler : public FairMQDevice
{

 public:

  Sampler();
  ~Sampler() override;
  
  void AddInputFileName(std::string s) { mFileNames.push_back(s); }
  void AddInputBranchName(std::string s) { mBranchNames.push_back(s); }

  void SetMaxIndex(int64_t tempInt) {mMaxIndex=tempInt;}
  
  void SetSource(FairSource* tempSource) {mSource = tempSource;}
  
  void ListenForAcks();
  
  void SetOutputChannelName(std::string tstr) {mOutputChannelName = tstr;}
  void SetAckChannelName(std::string tstr) {mAckChannelName = tstr;}

 protected:

  bool ConditionalRun() override;
  void PreRun() override;
  void PostRun() override;
  void InitTask() override;
 
 private:

  Sampler(const Sampler&);
  Sampler& operator=(const Sampler&);

  std::string     mOutputChannelName;
  std::string     mAckChannelName;
  
  FairRunAna*     mRunAna;
  FairSource*     mSource;
  TObject*        mInputObjects[100];
  int             mNObjects;
  int64_t         mMaxIndex;
  int             mEventCounter;
  std::vector<std::string>     mBranchNames;
  std::vector<std::string>     mFileNames;

  boost::thread* mAckListener;

};

}
}

#endif
