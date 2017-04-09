/**
 * FLPSyncSampler.cpp
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#include <fstream>
#include <ctime>

#include <FairMQLogger.h>
#include <options/FairMQProgOptions.h>

#include "FLP2EPNex_distributed/FLPSyncSampler.h"

using namespace std;
using namespace std::chrono;
using namespace o2::Devices;

FLPSyncSampler::FLPSyncSampler()
  : mTimeframeRTT()
  , mEventRate(1)
  , mMaxEvents(0)
  , mStoreRTTinFile(0)
  , mEventCounter(0)
  , mTimeFrameId(0)
  , mAckListener()
  , mResetEventCounter()
  , mLeaving(false)
  , mAckChannelName()
  , mOutChannelName()
{
}

FLPSyncSampler::~FLPSyncSampler()
= default;

void FLPSyncSampler::InitTask()
{
  // LOG(INFO) << "Waiting 10 seconds...";
  // this_thread::sleep_for(seconds(10));
  // LOG(INFO) << "Done!";
  mEventRate = GetConfig()->GetValue<int>("event-rate");
  mMaxEvents = GetConfig()->GetValue<int>("max-events");
  mStoreRTTinFile = GetConfig()->GetValue<int>("store-rtt-in-file");
  mAckChannelName = GetConfig()->GetValue<string>("ack-chan-name");
  mOutChannelName = GetConfig()->GetValue<string>("out-chan-name");
}

void FLPSyncSampler::PreRun()
{
  mLeaving = false;
  mAckListener = thread(&FLPSyncSampler::ListenForAcks, this);
  mResetEventCounter = thread(&FLPSyncSampler::ResetEventCounter, this);
}

bool FLPSyncSampler::ConditionalRun()
{
  FairMQMessagePtr msg(NewSimpleMessage(mTimeFrameId));

  if (fChannels.at(mOutChannelName).at(0).Send(msg) >= 0) {
    mTimeframeRTT[mTimeFrameId].start = steady_clock::now();

    if (++mTimeFrameId == UINT16_MAX - 1) {
      mTimeFrameId = 0;
    }
  }

  // rate limiting
  --mEventCounter;
  while (mEventCounter == 0) {
    this_thread::sleep_for(milliseconds(1));
  }

  if (mMaxEvents > 0 && mTimeFrameId >= mMaxEvents) {
    LOG(INFO) << "Reached configured maximum number of events (" << mMaxEvents << "). Exiting Run().";
    return false;
  }

  return true;
}

void FLPSyncSampler::PostRun()
{
    mLeaving = true;
    mResetEventCounter.join();
    mAckListener.join();
}

void FLPSyncSampler::ListenForAcks()
{
  uint16_t id = 0;

  ofstream ofsFrames;
  ofstream ofsTimes;

  // store round trip time measurements in a file
  if (mStoreRTTinFile > 0) {
    std::time_t t = system_clock::to_time_t(system_clock::now());
    tm utc = *gmtime(&t);
    std::stringstream s;
    s << utc.tm_year + 1900 << "-" << utc.tm_mon + 1 << "-" << utc.tm_mday << "-" << utc.tm_hour << "-" << utc.tm_min << "-" << utc.tm_sec;
    string name = s.str();
    ofsFrames.open(name + "-frames.log");
    ofsTimes.open(name + "-times.log");
  }

  while (!mLeaving) {
    FairMQMessagePtr idMsg(NewMessage());

    if (Receive(idMsg, mAckChannelName, 0, 1000) >= 0) {
      id = *(static_cast<uint16_t*>(idMsg->GetData()));
      mTimeframeRTT.at(id).end = steady_clock::now();
      // store values in a file
      auto elapsed = duration_cast<microseconds>(mTimeframeRTT.at(id).end - mTimeframeRTT.at(id).start);

      if (mStoreRTTinFile > 0) {
        ofsFrames << id << "\n";
        ofsTimes  << elapsed.count() << "\n";
      }

      LOG(INFO) << "Timeframe #" << id << " acknowledged after " << elapsed.count() << " μs.";
    }
  }

  // store round trip time measurements in a file
  if (mStoreRTTinFile > 0) {
    ofsFrames.close();
    ofsTimes.close();
  }
  LOG(INFO) << "Exiting Ack listener";
}

void FLPSyncSampler::ResetEventCounter()
{
  while (!mLeaving) {
    mEventCounter = mEventRate / 100;
    this_thread::sleep_for(milliseconds(10));
  }
  LOG(INFO) << "Exiting ResetEventCounter";
}
