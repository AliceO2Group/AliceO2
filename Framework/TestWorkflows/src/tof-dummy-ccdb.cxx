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
#include "DataFormatsTOF/CalibLHCphaseTOF.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  return {
    DataProcessorSpec{
      "simple",
      Inputs{},
      {OutputSpec{{"phase"}, "TOF", "LHCPhase"}},
      adaptStateless([](DataAllocator& outputs) {
        // Create and fill a dummy LHCphase object
        auto& lhcPhase = outputs.make<o2::dataformats::CalibLHCphaseTOF>(OutputRef{"phase", 0});
        lhcPhase.addLHCphase(0, 0);
      })}};
}
