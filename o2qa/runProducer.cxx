#include <csignal>
#include <FairMQLogger.h>
#include <cstdlib>
#include <vector>

#include "ProducerDevice.h"

namespace
{
std::vector<ProducerDevice*> producerDevices;
}

int main(int argc, char** argv)
{
    constexpr int requiredNumberOfProgramParameters{5};

    if (argc != requiredNumberOfProgramParameters) {
        LOG(ERROR) << "Wrong number of program parameters, required four parameters: xLow, xUp, name prefix and title";
        return -1;
    }

    ProducerDevice producerDevice("Producer", argv[3], argv[4], atof(argv[1]), atof(argv[2]), 1);
    producerDevices.push_back(&producerDevice);

    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "Producer id: " 
              << producerDevices[0]->GetProperty(ProducerDevice::Id, "default_id");

    producerDevices[0]->establishChannel("req", "connect", "tcp://localhost:5005", "data");

    producerDevices[0]->executeRunLoop();

    LOG(INFO) << "END OF runProducerDevice";

    return 0;
}
