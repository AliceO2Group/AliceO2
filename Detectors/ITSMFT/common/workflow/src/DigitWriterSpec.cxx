// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for ITSMFT digits

#include "ITSMFTWorkflow/DigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>
#include <string>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace itsmft
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MCCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

/// create the processor spec
/// describing a processor receiving digits for ITS/MFT and writing them to file
DataProcessorSpec getDigitWriterSpec(bool mctruth, o2::header::DataOrigin detOrig, o2::detectors::DetID detId)
{
  std::string detStr = o2::detectors::DetID::getName(detId);
  std::string detStrL = detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto logger = [](std::vector<o2::itsmft::Digit> const& inDigits) {
    LOG(INFO) << "RECEIVED DIGITS SIZE " << inDigits.size();
  };
  return MakeRootTreeWriterSpec((detStr + "DigitWriter").c_str(),
                                (detStrL + "digits.root").c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Digits tree"},
                                BranchDefinition<MCCont>{InputSpec{"digitsMCTR", detOrig, "DIGITSMCTR", 0},
                                                         (detStr + "DigitMCTruth").c_str(),
                                                         (mctruth ? 1 : 0)},
                                BranchDefinition<std::vector<itsmft::MC2ROFRecord>>{InputSpec{"digitsMC2ROF", detOrig, "DIGITSMC2ROF", 0},
                                                                                    (detStr + "DigitMC2ROF").c_str(),
                                                                                    (mctruth ? 1 : 0)},
                                BranchDefinition<std::vector<itsmft::Digit>>{InputSpec{"digits", detOrig, "DIGITS", 0},
                                                                             (detStr + "Digit").c_str(),
                                                                             logger},
                                BranchDefinition<std::vector<itsmft::ROFRecord>>{InputSpec{"digitsROF", detOrig, "DIGITSROF", 0},
                                                                                 (detStr + "DigitROF").c_str()})();
}

DataProcessorSpec getITSDigitWriterSpec(bool mctruth)
{
  return getDigitWriterSpec(mctruth, o2::header::gDataOriginITS, o2::detectors::DetID::ITS);
}

DataProcessorSpec getMFTDigitWriterSpec(bool mctruth)
{
  return getDigitWriterSpec(mctruth, o2::header::gDataOriginMFT, o2::detectors::DetID::MFT);
}

} // end namespace itsmft
} // end namespace o2
