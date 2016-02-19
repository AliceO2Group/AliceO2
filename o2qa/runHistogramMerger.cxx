/**
 * runHistogramMerger.cxx
 *
 * @since 2013-04-23
 * @author Patryk Lesiak
 */

#include <csignal>
#include <TApplication.h>
#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>

#include "HistogramMerger.h"
#include "HistogramTMessage.h"

using namespace std;

HistogramMerger histogramMerger("Merger_1", 1);

int main(int argc, char** argv)
{    
    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "Merger id: " 
              << histogramMerger.GetProperty(HistogramMerger::Id, "default_id");

    histogramMerger.establishChannel("rep", "bind", "tcp://*:5005", "data");
    histogramMerger.establishChannel("req", "connect", "tcp://localhost:5004", "data");
    // histogramMerger.establishChannel("rep", "bind", "tcp://*:5001", "data"); // controller
    histogramMerger.executeRunLoop();

    LOG(INFO) << "END OF runHistogramMerger";

    return 0;
}
