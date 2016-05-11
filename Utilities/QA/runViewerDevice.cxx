#include <csignal>
#include <TApplication.h>
#include <FairMQLogger.h>

#include "ViewerDevice.h"

using namespace std;

int main(int argc, char** argv)
{
  string drawingOptions = "";
  if (argc == 2) {
    drawingOptions = argv[1];
  }
  ViewerDevice viewerDevice("Viewer_1", 1, drawingOptions);
  TApplication *app = new TApplication("app1", &argc, argv);

  LOG(INFO) << "PID: " << getpid();
  LOG(INFO) << "Viewer id: "
            << viewerDevice.GetProperty(ViewerDevice::Id, "default_id");

  viewerDevice.establishChannel("rep", "bind", "tcp://*:5004", "data");
  viewerDevice.executeRunLoop();

  LOG(INFO) << "END OF runHistogramViewer";
}
