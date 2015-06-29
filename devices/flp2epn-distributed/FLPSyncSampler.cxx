/**
 * FLPSyncSampler.cpp
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#include <vector>
#include <fstream>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FLPSyncSampler.h"

using namespace std;
using boost::posix_time::ptime;

using namespace AliceO2::Devices;

FLPSyncSampler::FLPSyncSampler()
  : fEventRate(1)
  , fEventCounter(0)
  , fTimeframeRTT()
{
}

FLPSyncSampler::~FLPSyncSampler()
{
}

void FLPSyncSampler::InitTask()
{
  // LOG(INFO) << "Waiting 10 seconds...";
  // boost::this_thread::sleep(boost::posix_time::milliseconds(10000));
  // LOG(INFO) << "Done!";
}

void FLPSyncSampler::Run()
{
  boost::thread resetEventCounter(boost::bind(&FLPSyncSampler::ResetEventCounter, this));
  boost::thread ackListener(boost::bind(&FLPSyncSampler::ListenForAcks, this));

  int NOBLOCK = fChannels.at("data-out").at(0).fSocket->NOBLOCK;

  uint16_t timeFrameId = 1;

  FairMQChannel& dataOutputChannel = fChannels.at("data-out").at(0);

  while (CheckCurrentState(RUNNING)) {
    FairMQMessage* msg = fTransportFactory->CreateMessage(sizeof(uint16_t));
    memcpy(msg->GetData(), &timeFrameId, sizeof(uint16_t));

    if (dataOutputChannel.Send(msg, NOBLOCK) > 0) {
      fTimeframeRTT[timeFrameId].start = boost::posix_time::microsec_clock::local_time();

      if (++timeFrameId == UINT16_MAX - 1) {
        timeFrameId = 1;
      }
    }

    --fEventCounter;

    while (fEventCounter == 0) {
      boost::this_thread::sleep(boost::posix_time::milliseconds(1));
    }

    delete msg;
  }

  try {
    resetEventCounter.interrupt();
    resetEventCounter.join();
    ackListener.interrupt();
    ackListener.join();
  } catch(boost::thread_resource_error& e) {
    LOG(ERROR) << e.what();
  }
}

void FLPSyncSampler::ListenForAcks()
{
  uint16_t id = 0;

  string name = to_iso_string(boost::posix_time::microsec_clock::local_time()).substr(0, 20);
  ofstream ofsFrames(name + "-frames.log");
  ofstream ofsTimes(name + "-times.log");

  while (CheckCurrentState(RUNNING)) {
    try {
      FairMQMessage* idMsg = fTransportFactory->CreateMessage();

      if (fChannels.at("ack-in").at(0).Receive(idMsg) > 0) {
        id = *(static_cast<uint16_t*>(idMsg->GetData()));
        fTimeframeRTT.at(id).end = boost::posix_time::microsec_clock::local_time();
        // store values in a file
        ofsFrames << id << "\n";
        ofsTimes  << (fTimeframeRTT.at(id).end - fTimeframeRTT.at(id).start).total_microseconds() << "\n";

        LOG(INFO) << "Timeframe #" << id << " acknowledged after "
                  << (fTimeframeRTT.at(id).end - fTimeframeRTT.at(id).start).total_microseconds() << " μs.";
      }
    } catch (boost::thread_interrupted&) {
      LOG(DEBUG) << "Acknowledgement listener thread interrupted";
      break;
    }
  }

  ofsFrames.close();
  ofsTimes.close();
}

void FLPSyncSampler::ResetEventCounter()
{
  while (true) {
    try {
      fEventCounter = fEventRate / 100;
      boost::this_thread::sleep(boost::posix_time::milliseconds(10));
    } catch (boost::thread_interrupted&) {
      LOG(DEBUG) << "Event rate limiter thread interrupted";
      break;
    }
  }
}

void FLPSyncSampler::SetProperty(const int key, const string& value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string FLPSyncSampler::GetProperty(const int key, const string& default_ /*= ""*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void FLPSyncSampler::SetProperty(const int key, const int value)
{
  switch (key) {
    case EventRate:
      fEventRate = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int FLPSyncSampler::GetProperty(const int key, const int default_ /*= 0*/)
{
  switch (key) {
    case EventRate:
      return fEventRate;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}
