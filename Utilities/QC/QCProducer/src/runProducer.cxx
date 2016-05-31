#include <csignal>
#include <FairMQLogger.h>
#include <cstdlib>

#include "QCProducer/ProducerDevice.h"
#include "QCProducer/HistogramProducer.h"
#include "QCProducer/TreeProducer.h"


int main(int argc, char** argv)
{
  const int requiredNumberOfParametersForTrees = 6;
  const int requiredNumberOfParametersForHistograms = 6;

  int numberOfIoThreads = 1;
  std::string producerType = argv[1];

  std::string namePrefix = argv[2];
  std::string title = argv[3];

  std::shared_ptr<Producer> producer;

  if (producerType == "-histogram") {
    float xLow = atof(argv[4]);
    float xUp = atof(argv[5]);
    producer = std::make_shared<HistogramProducer>(namePrefix, title, xLow, xUp);
  }
  else if (producerType == "-tree" && argc == requiredNumberOfParametersForTrees) {
    float numberOfBranches = atof(argv[4]);
    float numberOfEntriesInEachBranch = atof(argv[5]);
    producer = std::make_shared<TreeProducer>(namePrefix, title, numberOfBranches, numberOfEntriesInEachBranch);
  }
  else {
    LOG(ERROR) << "Unknown type of producer: " << producerType;
    return -1;
  }

  ProducerDevice producerDevice("Producer", numberOfIoThreads, producer);

  LOG(INFO) << "PID: " << getpid() << "Producer id: " << producerDevice.GetProperty(ProducerDevice::Id, "default_id");

  producerDevice.establishChannel("req", "connect", "tcp://localhost:5005", "data");
  producerDevice.executeRunLoop();

  LOG(INFO) << "END OF runProducerDevice";
}
