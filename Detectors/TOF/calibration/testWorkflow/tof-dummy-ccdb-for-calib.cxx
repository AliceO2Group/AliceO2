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
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  return {
    DataProcessorSpec{
      "simple",
      Inputs{{"input", "TOF", "CALIBDATA"}},
      Outputs{OutputSpec{{"phase"}, "TOF", "LHCphase"},
              OutputSpec{{"timeSlewing"}, "TOF", "ChannelCalib"},
              OutputSpec{{"startLHCphase"}, "TOF", "StartLHCphase"},
              OutputSpec{{"startTimeChCal"}, "TOF", "StartChCalib"}},
      adaptStateless([](DataAllocator& outputs, ControlService& control) {
        // Create and fill a dummy LHCphase object
        auto& lhcPhase = outputs.make<o2::dataformats::CalibLHCphaseTOF>(OutputRef{"phase", 0});
        lhcPhase.addLHCphase(0, 1234);          // should be in ps
        lhcPhase.addLHCphase(2000000000, 2345); // should be in ps
        auto& calibTimeSlewing = outputs.make<o2::dataformats::CalibTimeSlewingParamTOF>(OutputRef{"timeSlewing", 0});
        for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELS; ich++) {
          calibTimeSlewing.addTimeSlewingInfo(ich, 0, 0);
          int sector = ich / o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
          int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
          calibTimeSlewing.setFractionUnderPeak(sector, channelInSector, 1);
        }
        auto& startTimeLHCphase = outputs.make<long>(OutputRef{"startLHCphase", 0}); // we send also the start validity of the LHC phase
        auto& startTimeChCalib = outputs.make<long>(OutputRef{"startTimeChCal", 0}); // we send also the start validity of the channel calibration
                                                                                     // you should uncomment the lines below if you need that this workflow triggers the end of the subsequent ones (e.g. in the tof reconstruction workflow, but not fot the tof-calib-workflow
        //control.endOfStream();
        //control.readyToQuit(QuitRequest::Me);
      })}};
}
