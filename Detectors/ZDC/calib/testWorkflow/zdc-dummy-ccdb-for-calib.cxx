// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataAllocator.h"
#include "Framework/runDataProcessing.h"
#include <TClass.h>
#include "Framework/ControlService.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  return {
    DataProcessorSpec{
      "simple",
      Inputs{{"input", "ZDC", "CALIBDATA"}},
      Outputs{OutputSpec{{"config"}, "ZDC", "ZDCConfig"},
              OutputSpec{{"hv"}, "ZDC", "HVSetting"},
              OutputSpec{{"position"}, "ZDC", "Position"}},
      /*adaptStateless([](DataAllocator& outputs, ControlService& control) {
        auto& calibTimeSlewing = outputs.make<o2::dataformats::CalibTimeSlewingParamZDC>(OutputRef{"timeSlewing", 0});
        for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamZDC::NCHANNELS; ich++) {
          calibTimeSlewing.addTimeSlewingInfo(ich, 0, 0);
          int sector = ich / o2::dataformats::CalibTimeSlewingParamZDC::NCHANNELXSECTOR;
          int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamZDC::NCHANNELXSECTOR;
          calibTimeSlewing.setFractionUnderPeak(sector, channelInSector, 1);
        }*/

      // you should uncomment the lines below if you need that this workflow
      // triggers the end of the subsequent ones (e.g. in the ZDC reconstruction workflow, but not fot the ZDC-calib-workflow)
      // control.endOfStream();
      // control.readyToQuit(QuitRequest::Me);
    }};
}
