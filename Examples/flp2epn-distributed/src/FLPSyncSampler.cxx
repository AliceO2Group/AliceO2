/**
 * FLPSyncSampler.cpp
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#include <fstream>
#include <ctime>

#include "FairMQLogger.h"
#include "FairMQProgOptions.h"

#include "FLP2EPNex_distributed/FLPSyncSampler.h"

using namespace std;
using namespace std::chrono;
using namespace AliceO2::Devices;

FLPSyncSampler::FLPSyncSampler()
  : fTimeframeRTT()
  , fEventRate(1)
  , fMaxEvents(0)
  , fStoreRTTinFile(0)
  , fEventCounter(0)
  , fTimeFrameId(0)
  , fAckListener()
  , fResetEventCounter()
  , fLeaving(false)
{
}

FLPSyncSampler::~FLPSyncSampler()
{
}

void FLPSyncSampler::InitTask()
{
  // LOG(INFO) << "Waiting 10 seconds...";
  // this_thread::sleep_for(seconds(10));
  // LOG(INFO) << "Done!";
  fEventRate = fConfig->GetValue<int>("event-rate");
  fMaxEvents = fConfig->GetValue<int>("max-events");
  fStoreRTTinFile = fConfig->GetValue<int>("store-rtt-in-file");
}

void FLPSyncSampler::PreRun()
{
  fLeaving = false;
  fAckListener = thread(&FLPSyncSampler::ListenForAcks, this);
  fResetEventCounter = thread(&FLPSyncSampler::ResetEventCounter, this);
}

bool FLPSyncSampler::ConditionalRun()
{
  FairMQMessagePtr msg(NewSimpleMessage(fTimeFrameId));

  if (fChannels.at("data").at(0).Send(msg) >= 0) {
    fTimeframeRTT[fTimeFrameId].start = steady_clock::now();

    if (++fTimeFrameId == UINT16_MAX - 1) {
      fTimeFrameId = 0;
    }
  }

  // rate limiting
  --fEventCounter;
  while (fEventCounter == 0) {
    this_thread::sleep_for(milliseconds(1));
  }

  if (fMaxEvents > 0 && fTimeFrameId >= fMaxEvents) {
    LOG(INFO) << "Reached configured maximum number of events (" << fMaxEvents << "). Exiting Run().";
    return false;
  }

  return true;
}

void FLPSyncSampler::PostRun()
{
    fLeaving = true;
    fResetEventCounter.join();
    fAckListener.join();
}

void FLPSyncSampler::ListenForAcks()
{
  uint16_t id = 0;

  ofstream ofsFrames;
  ofstream ofsTimes;

  // store round trip time measurements in a file
  if (fStoreRTTinFile > 0) {
    std::time_t t = system_clock::to_time_t(system_clock::now());
    tm utc = *gmtime(&t);
    std::stringstream s;
    s << utc.tm_year + 1900 << "-" << utc.tm_mon + 1 << "-" << utc.tm_mday << "-" << utc.tm_hour << "-" << utc.tm_min << "-" << utc.tm_sec;
    string name = s.str();
    ofsFrames.open(name + "-frames.log");
    ofsTimes.open(name + "-times.log");
  }

  while (!fLeaving) {
    FairMQMessagePtr idMsg(NewMessage());

    if (fChannels.at("ack").at(0).Receive(idMsg) >= 0) {
      id = *(static_cast<uint16_t*>(idMsg->GetData()));
      fTimeframeRTT.at(id).end = steady_clock::now();
      // store values in a file
      auto elapsed = duration_cast<microseconds>(fTimeframeRTT.at(id).end - fTimeframeRTT.at(id).start);

      if (fStoreRTTinFile > 0) {
        ofsFrames << id << "\n";
        ofsTimes  << elapsed.count() << "\n";
      }

      LOG(INFO) << "Timeframe #" << id << " acknowledged after " << elapsed.count() << " μs.";
    } else {
      break;
    }
  }

  // store round trip time measurements in a file
  if (fStoreRTTinFile > 0) {
    // ofsFrames.close();
    // ofsTimes.close();
  }
  LOG(INFO) << "Exiting Ack listener";
}

void FLPSyncSampler::ResetEventCounter()
{
  while (!fLeaving) {
    fEventCounter = fEventRate / 100;
    this_thread::sleep_for(milliseconds(10));
  }
  LOG(INFO) << "Exiting ResetEventCounter";
}
