#ifndef TASKPROCESSOR_H_
#define TASKPROCESSOR_H_

#include <string>

#include "FairMQDevice.h"

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

 protected:

  virtual void Run();
  virtual void Init();

 private:
  
  std::string     fInputChannelName;
  std::string     fOutputChannelName;
  
  EventHeader*     fEventHeader;
  TList*           fInput;
  TList*           fOutput;
  
  int fNewRunId;
  int fCurrentRunId;
  
  std::string fDataToKeep;
  
  T* fFairTask;

  TaskProcessor(const TaskProcessor&);    
  TaskProcessor& operator=(const TaskProcessor&);    

};

#include "TaskProcessor.tpl"

}
}

#endif
