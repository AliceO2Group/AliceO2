#include <cstdlib>
#include <mutex>
#include <string>

#include <FairMQLogger.h>

#include <dds_intercom.h>

#include "QCProducer/ProducerDevice.h"
#include "QCProducer/TH1Producer.h"
#include "QCProducer/TH2Producer.h"
#include "QCProducer/TH3Producer.h"
#include "QCProducer/THnProducer.h"
#include "QCProducer/TreeProducer.h"

using namespace dds;
using namespace dds::intercom_api;
using namespace std;
using namespace o2::qc;

namespace
{
shared_ptr<Producer> producer;
const int NUMBER_OF_IO_THREADS = 1;
const char* outputAddress;
mutex keyMutex;
}

int main(int argc, char** argv)
{
  if (argc < 7 || argc > 8) {
    LOG(ERROR) << "Provided wrong number of arguments: " << argc - 1;
    exit(-1);
  }

  string outputTopologyProperty = argv[1];
  const char* producerId = argv[2];
  const string producerType = argv[3];
  const char* name = argv[4];
  const char* title = argv[5];
  const int bufferSize = atoi(argv[6]);

  if (producerType.compare("TH1F") == 0) {
    LOG(INFO) << "Producing TH1F objects";
    const int binsNumber = atoi(argv[7]);
    producer = make_shared<TH1Producer>(name, title, binsNumber);
  } else if (producerType.compare("TH2F") == 0) {
    LOG(INFO) << "Producing TH2F objects";
    const int binsNumber = atoi(argv[7]);
    producer = make_shared<TH2Producer>(name, title, binsNumber);
  } else if (producerType.compare("TH3F") == 0) {
    LOG(INFO) << "Producing TH3F objects";
    const int binsNumber = atoi(argv[7]);
    producer = make_shared<TH3Producer>(name, title, binsNumber);
  } else if (producerType.compare("THnF") == 0) {
    LOG(INFO) << "Producing THnF objects";
    const int binsNumber = atoi(argv[7]);
    producer = make_shared<THnProducer>(name, title, binsNumber);
  } else if (producerType.compare("TTree") == 0) {
    LOG(INFO) << "Producing TTree objects";
    const int numberOfBranches = atoi(argv[7]);
    const int numberOfEntriesInEachBranch = atoi(argv[8]);
    producer = make_shared<TreeProducer>(name, title, numberOfBranches, numberOfEntriesInEachBranch);
  } else {
    LOG(ERROR) << R"(Unknown type of producer: ")" << producerType << R"(", number of provided arguments: )"
               << (argc - 1);
    return -1;
  }

  ProducerDevice producerDevice(producerId, NUMBER_OF_IO_THREADS, producer);
  producerDevice.CatchSignals();

  LOG(INFO) << "PID: " << getpid();
  LOG(INFO) << "Producer id: " << producerDevice.GetProperty(ProducerDevice::Id, "default_id");
  LOG(INFO) << "Hostname: " << getenv("HOSTNAME");

  condition_variable keyCondition;
  CIntercomService service;
  CKeyValue keyValue(service);

  service.subscribeOnError([](EErrorCode _errorCode, const string& _msg) {
    cerr << "DDS key-value error code: " << _errorCode << ", message: " << _msg;
  });

  keyValue.subscribe(
    [&keyCondition, &outputTopologyProperty](const string& _propertyID, const string& _key, const string& _value) {
      LOG(INFO) << "Received key-value update: propertyID=" << _propertyID << " key=" << _key << " value=" << _value
                << std::endl;

      if (outputTopologyProperty.compare(_propertyID) == 0) {
        outputAddress = _value.c_str();
        keyCondition.notify_all();
      }
    });

  service.start();
  unique_lock<mutex> lock(keyMutex);
  keyCondition.wait(lock);

  LOG(INFO) << "Output address: " << outputAddress;

  producerDevice.establishChannel("push", "connect", outputAddress, "data-out", bufferSize);
  producerDevice.executeRunLoop();
}
