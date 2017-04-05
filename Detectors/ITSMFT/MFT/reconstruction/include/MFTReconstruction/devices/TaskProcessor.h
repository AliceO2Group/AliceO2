#ifndef TASKPROCESSOR_H_
#define TASKPROCESSOR_H_

#include <FairMQDevice.h>
#include <FairMQParts.h>

#include "TMessage.h"

#include "MFTSimulation/EventHeader.h"

class TList;

namespace AliceO2 {

namespace MFT {

template<typename T>
class TaskProcessor : public FairMQDevice
{

 public:

  TaskProcessor();
  virtual ~TaskProcessor();

  void SetDataToKeep(std::string tStr) { mDataToKeep = tStr;}

  void SetInputChannelName (std::string tstr) {mInputChannelName = tstr;}
  void SetOutputChannelName(std::string tstr) {mOutputChannelName = tstr;}
  void SetParamChannelName (std::string tstr) {mParamChannelName  = tstr;}

 protected:

  bool ProcessData(FairMQParts&, int);
  virtual void Init();
  virtual void PostRun();

 private:
  
  std::string     mInputChannelName;
  std::string     mOutputChannelName;
  std::string     mParamChannelName;
  
  EventHeader*     mEventHeader;
  TList*           mInput;
  TList*           mOutput;
  
  int mNewRunId;
  int mCurrentRunId;
  
  std::string mDataToKeep;
  
  int mReceivedMsgs = 0;
  int mSentMsgs = 0;

  T* mFairTask;

  TaskProcessor(const TaskProcessor&);    
  TaskProcessor& operator=(const TaskProcessor&);    

};

#include "TaskProcessor.tpl"

}
}

#endif
