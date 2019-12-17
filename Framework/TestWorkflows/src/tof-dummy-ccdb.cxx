// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
      Inputs{},
      Outputs{OutputSpec{{"phase"}, "TOF", "LHCphase"},
              OutputSpec{{"timeSlewing"}, "TOF", "ChannelCalib"}},
      adaptStateless([](DataAllocator& outputs, ControlService& control) {
        // Create and fill a dummy LHCphase object
        auto& lhcPhase = outputs.make<o2::dataformats::CalibLHCphaseTOF>(OutputRef{"phase", 0});
        lhcPhase.addLHCphase(0, 123450);
        lhcPhase.addLHCphase(2000000000, 234567);
        auto& calibTimeSlewing = outputs.make<o2::dataformats::CalibTimeSlewingParamTOF>(OutputRef{"timeSlewing", 0});
        for (int ich = 0; ich < o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELS; ich++) {
          calibTimeSlewing.addTimeSlewingInfo(ich, 0, 0);
          int sector = ich / o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
          int channelInSector = ich % o2::dataformats::CalibTimeSlewingParamTOF::NCHANNELXSECTOR;
          calibTimeSlewing.setFractionUnderPeak(sector, channelInSector, 1);
        }
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
      })}};
}
