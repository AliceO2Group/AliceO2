#ifndef TASKPROCESSOR_H_
#define TASKPROCESSOR_H_

#include "FairMQDevice.h"
#include "FairMQParts.h"

#include "MFTBase/EventHeader.h"

#include "TMessage.h"

class TList;

namespace AliceO2 {

namespace MFT {

template<typename T>
class TaskProcessor : public FairMQDevice
{

 public:

  TaskProcessor();
  virtual ~TaskProcessor();

  void SetDataToKeep(std::string tStr) { fDataToKeep = tStr;}

  void SetInputChannelName (std::string tstr) {fInputChannelName = tstr;}
  void SetOutputChannelName(std::string tstr) {fOutputChannelName = tstr;}
  void SetParamChannelName (std::string tstr) {fParamChannelName  = tstr;}

 protected:

  bool ProcessData(FairMQParts&, int);
  virtual void Init();
  virtual void PostRun();

 private:
  
  std::string     fInputChannelName;
  std::string     fOutputChannelName;
  std::string     fParamChannelName;
  
  EventHeader*     fEventHeader;
  TList*           fInput;
  TList*           fOutput;
  
  int fNewRunId;
  int fCurrentRunId;
  
  std::string fDataToKeep;
  
  int fReceivedMsgs = 0;
  int fSentMsgs = 0;

  T* fFairTask;

  TaskProcessor(const TaskProcessor&);    
  TaskProcessor& operator=(const TaskProcessor&);    

};

#include "TaskProcessor.tpl"

}
}

#endif
