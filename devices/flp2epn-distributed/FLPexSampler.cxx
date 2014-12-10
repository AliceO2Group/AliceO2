/**
 * FLPexSampler.cpp
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#include <vector>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FLPexSampler.h"

using namespace std;
using boost::posix_time::ptime;

using namespace AliceO2::Devices;

FLPexSampler::FLPexSampler()
  : fEventRate(1)
  , fEventCounter(0)
{
}

FLPexSampler::~FLPexSampler()
{
}

void FLPexSampler::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";
  boost::this_thread::sleep(boost::posix_time::milliseconds(10000));

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));
  boost::thread resetEventCounter(boost::bind(&FLPexSampler::ResetEventCounter, this));

  uint64_t timeframeId = 0;

  while (fState == RUNNING) {
    FairMQMessage* msg = fTransportFactory->CreateMessage(sizeof(uint64_t));
    memcpy(msg->GetData(), &timeframeId, sizeof(uint64_t));

    if (fPayloadOutputs->at(0)->Send(msg, "no-block") == 0) {
      LOG(ERROR) << "Could not send signal without blocking";
    }

    ++timeframeId;

    --fEventCounter;

    while (fEventCounter == 0) {
      boost::this_thread::sleep(boost::posix_time::milliseconds(1));
    }

    delete msg;
  }

  try {
    rateLogger.interrupt();
    rateLogger.join();
    resetEventCounter.interrupt();
    resetEventCounter.join();
  } catch(boost::thread_resource_error& e) {
    LOG(ERROR) << e.what();
  }

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

void FLPexSampler::ResetEventCounter()
{
  while (true) {
    try {
      fEventCounter = fEventRate / 100;
      boost::this_thread::sleep(boost::posix_time::milliseconds(10));
    } catch (boost::thread_interrupted&) {
      break;
    }
  }
}

void FLPexSampler::SetProperty(const int key, const string& value, const int slot /*= 0*/)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

string FLPexSampler::GetProperty(const int key, const string& default_ /*= ""*/, const int slot /*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}

void FLPexSampler::SetProperty(const int key, const int value, const int slot /*= 0*/)
{
  switch (key) {
    case EventRate:
      fEventRate = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

int FLPexSampler::GetProperty(const int key, const int default_ /*= 0*/, const int slot /*= 0*/)
{
  switch (key) {
    case EventRate:
      return fEventRate;
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}
