#include <TApplication.h>
#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>
#include <memory>

#include "QCMerger/MergerDevice.h"
#include "QCCommon/HistogramTMessage.h"
#include "QCMerger/Merger.h"

using namespace std;

int main(int argc, char** argv)
{
  MergerDevice mergerDevice(unique_ptr<Merger>(new Merger()), "Merger_1", 1);

  LOG(INFO) << "PID: " << getpid();
  LOG(INFO) << "Merger id: "
            << mergerDevice.GetProperty(MergerDevice::Id, "default_id");

  mergerDevice.establishChannel("rep", "bind", "tcp://*:5005", "data");
  mergerDevice.establishChannel("req", "connect", "tcp://localhost:5004", "data");
  mergerDevice.executeRunLoop();

  LOG(INFO) << "END OF runMergerDevice";
}
