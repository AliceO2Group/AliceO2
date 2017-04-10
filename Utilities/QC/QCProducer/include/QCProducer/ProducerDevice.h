#pragma once

#include <memory>
#include <string>

#include <FairMQDevice.h>

#include <dds_intercom.h>

#include "QCProducer/Producer.h"

namespace o2
{
namespace qc
{
class ProducerDevice : public FairMQDevice
{
 public:
  ProducerDevice(const char* producerId, const int numIoThreads, std::shared_ptr<Producer>& producer);
  ~ProducerDevice() override = default;

  static void deleteTMessage(void* data, void* hint);
  void executeRunLoop();
  void establishChannel(std::string type, std::string method, std::string address, std::string channelName,
                        const int bufferSize);

 protected:
  ProducerDevice() = default;
  void Run() override;

 private:
  std::shared_ptr<Producer> mProducer;
  dds::intercom_api::CIntercomService mService;
  std::unique_ptr<dds::intercom_api::CCustomCmd> ddsCustomCmd;
  int mNumberOfEntries;

  volatile bool mBufferOverloaded{ false };
  clock_t mLastBufferOverloadTime{ 0 };

  void subscribeDdsCommands();
  void sendDataToMerger(std::unique_ptr<FairMQMessage> request);
  bool outputLimitReached();
  int getCurrentSecond() const;
  void waitForLimitUnlock();

  unsigned int mInternalStateMessageId{ 0 };
  const int OUTPUT_LIMIT_PER_SECOND{ 100 };
  int sentObjectsInCurrentSecond{ 0 };
  int lastCheckedSecond{ 0 };
};
}
}