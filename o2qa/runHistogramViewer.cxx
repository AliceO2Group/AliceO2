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

int main(int argc, char** argv)
{   
    TApplication *app; 
    app = new TApplication("app1", &argc, argv);

    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "Viewer id: " 
              << histogramViewer.GetProperty(HistogramViewer::Id, "default_id");

    histogramViewer.establishChannel("rep", "bind", "tcp://*:5004", "data");

    histogramViewer.executeRunLoop();

    LOG(INFO) << "END OF runHistogramViewer";

    return 0;
}
