#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <string>

#include <boost/thread.hpp>

#include "FairFileSource.h"
#include "FairRunAna.h"
#include "FairMQDevice.h"

namespace AliceO2 {

namespace MFT {

class Sampler : public FairMQDevice
{

 public:

  Sampler();
  virtual ~Sampler();
  
  void AddInputFileName(std::string s) { fFileNames.push_back(s); }
  void AddInputBranchName(std::string s) { fBranchNames.push_back(s); }

  void SetMaxIndex(int64_t tempInt) {fMaxIndex=tempInt;}
  
  void SetSource(FairSource* tempSource) {fSource = tempSource;}
  
  void ListenForAcks();
  
  void SetOutputChannelName(std::string tstr) {fOutputChannelName = tstr;}
  void SetAckChannelName(std::string tstr) {fAckChannelName = tstr;}

 protected:

  virtual bool ConditionalRun();
  virtual void PreRun();
  virtual void PostRun();
  virtual void InitTask();
 
 private:

  Sampler(const Sampler&);
  Sampler& operator=(const Sampler&);

  std::string     fOutputChannelName;
  std::string     fAckChannelName;
  
  FairRunAna*     fRunAna;
  FairSource*     fSource;
  TObject*        fInputObjects[100];
  int             fNObjects;
  int64_t         fMaxIndex;
  int             fEventCounter;
  std::vector<std::string>     fBranchNames;
  std::vector<std::string>     fFileNames;

  boost::thread* fAckListener;

};

}
}

#endif
