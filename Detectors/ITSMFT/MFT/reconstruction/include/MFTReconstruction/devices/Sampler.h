// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
  
  void addInputFileName(std::string s) { mFileNames.push_back(s); }
  void addInputBranchName(std::string s) { mBranchNames.push_back(s); }

  void setMaxIndex(int64_t tempInt) {mMaxIndex=tempInt;}
  
  void setSource(FairSource* tempSource) {mSource = tempSource;}
  
  void listenForAcks();
  
  void setOutputChannelName(std::string tstr) {mOutputChannelName = tstr;}
  void setAckChannelName(std::string tstr) {mAckChannelName = tstr;}

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
