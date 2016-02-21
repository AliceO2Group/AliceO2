#include <TApplication.h>
#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>
#include <memory>

#include "MergerDevice.h"
#include "HistogramTMessage.h"
#include "Merger.h"

using namespace std;

MergerDevice mergerDevice(unique_ptr<Merger>(new Merger()), "Merger_1", 1);

int main(int argc, char** argv)
{    
    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "Merger id: " 
              << mergerDevice.GetProperty(MergerDevice::Id, "default_id");

    mergerDevice.establishChannel("rep", "bind", "tcp://*:5005", "data");
    mergerDevice.establishChannel("req", "connect", "tcp://localhost:5004", "data");
    // mergerDevice.establishChannel("rep", "bind", "tcp://*:5001", "data"); // controller
    mergerDevice.executeRunLoop();

    LOG(INFO) << "END OF runHistogramMerger";

    return 0;
}
