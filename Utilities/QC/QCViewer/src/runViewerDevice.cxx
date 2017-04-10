#include <csignal>

#include <FairMQLogger.h>
#include <TApplication.h>

#include "QCViewer/ViewerDevice.h"

using namespace o2::qc;
using namespace std;

int main(int argc, char** argv)
{
  string drawingOptions = "";
  if (argc == 2) {
    drawingOptions = argv[1];
  }
  ViewerDevice viewerDevice("Viewer_1", 1, drawingOptions);
  viewerDevice.CatchSignals();
  auto* app = new TApplication("app1", &argc, argv);

  LOG(INFO) << "PID: " << getpid();
  LOG(INFO) << "Viewer id: " << viewerDevice.GetProperty(ViewerDevice::Id, "default_id");

  viewerDevice.establishChannel("pull", "bind", "tcp://*:5004", "data-in");
  viewerDevice.executeRunLoop();

  LOG(INFO) << "END OF runHistogramViewer";
}
