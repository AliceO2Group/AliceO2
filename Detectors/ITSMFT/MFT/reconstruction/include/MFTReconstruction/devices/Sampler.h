#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <string>

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

 protected:

  virtual void Run();
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
  std::vector<std::string>     fBranchNames;
  std::vector<std::string>     fFileNames;

};

}
}

#endif
