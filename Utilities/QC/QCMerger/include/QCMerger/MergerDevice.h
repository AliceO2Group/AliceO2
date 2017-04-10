#pragma once

#include <chrono>
#include <ctime>
#include <deque>
#include <memory>
#include <string>

#include <boost/property_tree/ptree.hpp>

#include <FairMQDevice.h>
#include <TApplication.h>
#include <TMessage.h>
#include <dds_intercom.h>

#include "Merger.h"

namespace o2
{
namespace qc
{
class MergerDevice : public FairMQDevice
{
 public:
  MergerDevice(std::unique_ptr<Merger> merger, std::string producerId, int numIoThreads);
  ~MergerDevice() override;

  static void deleteTMessage(void* data, void* hint);
  void establishChannel(std::string type, std::string method, std::string address, std::string channelName,
                        int receiveBuffer, int sendBuffer);
  void executeRunLoop();

 protected:
  void Run() override;

 private:
  void subscribeCustomCommands();
  void subscribeDdsCommands();
  void subscribeOnError();
  boost::property_tree::ptree createCheckStateResponse(const boost::property_tree::ptree& request);
  boost::property_tree::ptree createGetMetricsResponse(const boost::property_tree::ptree& request);
  void handleReceivedDataObject();
  TObject* receiveDataObjectFromProducer();
  TMessage* createTMessageForViewer(const TObject* objectToSend) const;
  size_t sendMergedObjectToViewer(TObject* dataObject);
  void sendControlResponse(const boost::property_tree::ptree& response, std::string senderId);
  std::string getVmRSSUsage();
  double calculateAvgMegreTime();
  std::string calculateCpuUsage();
  double calculateNumberOfMergedObjectsPerSecond();
  void updateMetrics();
  inline bool isObjectNotEmpty(const TObject* object) const;

  std::unique_ptr<Merger> mMerger;
  dds::intercom_api::CIntercomService mService;
  std::unique_ptr<dds::intercom_api::CCustomCmd> ddsCustomCmd;
  std::deque<double> mMergeTimes;

  std::chrono::high_resolution_clock::time_point lastCpuMeasuredTime;
  std::chrono::high_resolution_clock::time_point mLastNumberOfMergeObjectsTime;
  unsigned int mNumberOfMergedObjects{ 0 };
  clock_t lastCpuMeasuredValue{ 0 };
  std::ifstream procSelfStatus;

  const unsigned LOGGED_MESSAGES{ 10 };
  const size_t MESSAGE_MAXIMUM_SIZE = 1000000 * 1000; // Mb

  volatile bool mReceiveBufferOverloaded{ false };
  clock_t mLastReceiveBufferOverloadTime{ 0 };

  volatile bool mSendBufferOverloaded{ false };
  clock_t mLastSendBufferOverloadTime{ 0 };

  unsigned int mInternalMetricMessageId{ 0 };
  unsigned int mInternalStateMessageId{ 0 };
};
}
}