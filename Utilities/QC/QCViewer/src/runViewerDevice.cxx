// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
  ViewerDevice viewerDevice("Viewer_1", drawingOptions);
  viewerDevice.CatchSignals();
  auto* app = new TApplication("app1", &argc, argv);

  LOG(INFO) << "PID: " << getpid();
  LOG(INFO) << "Viewer id: " << viewerDevice.GetId();

  viewerDevice.establishChannel("pull", "bind", "tcp://*:5004", "data-in");
  viewerDevice.executeRunLoop();

  LOG(INFO) << "END OF runHistogramViewer";
}
