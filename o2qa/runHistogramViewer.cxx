/**
 * runHistogramViewer.cxx
 *
 * @since 2013-04-23
 * @author Patryk Lesiak
 */

#include <csignal>
#include <TApplication.h>
#include <FairMQLogger.h>

#include "HistogramViewer.h"

using namespace std;

HistogramViewer histogramViewer("Viewer_1", 1);

namespace
{

void signalHandler(int signal)
{
    LOG(INFO) << "Caught signal " << signal;
    histogramViewer.ChangeState(HistogramViewer::END);
    LOG(INFO) << "Caught signal " << signal;
}

}

int main(int argc, char** argv)
{   
    TApplication *app; 
    app = new TApplication("app1", &argc, argv);

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "Viewer id: " 
              << histogramViewer.GetProperty(HistogramViewer::Id, "default_id");

    histogramViewer.establishChannel("rep", "bind", "tcp://*:5004", "data");

    histogramViewer.executeRunLoop();

    LOG(INFO) << "END OF runHistogramViewer";

    return 0;
}
