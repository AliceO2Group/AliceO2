// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
