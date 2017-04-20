#ifndef TASKPROCESSOR_H_
#define TASKPROCESSOR_H_

#include <FairMQDevice.h>
#include <FairMQParts.h>

#include "TMessage.h"

#include "MFTSimulation/EventHeader.h"

class TList;

namespace o2 {

namespace MFT {

template<typename T>
class TaskProcessor : public FairMQDevice
{

 public:

  TaskProcessor();
  ~TaskProcessor() override;

  void setDataToKeep(std::string tStr) { mDataToKeep = tStr;}

  void setInputChannelName (std::string tstr) {mInputChannelName = tstr;}
  void setOutputChannelName(std::string tstr) {mOutputChannelName = tstr;}
  void setParamChannelName (std::string tstr) {mParamChannelName  = tstr;}

 protected:

  bool processData(FairMQParts&, int);
  void Init() override;
  void PostRun() override;

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
