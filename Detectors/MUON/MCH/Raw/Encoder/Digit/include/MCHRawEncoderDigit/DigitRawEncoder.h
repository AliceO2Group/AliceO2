// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_DIGIT_RAW_ENCODER_H
#define O2_MCH_RAW_ENCODER_DIGIT_RAW_ENCODER_H

#include "DataFormatsMCH/Digit.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MCHRawEncoderDigit/DigitPayloadEncoder.h"
#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include <cstdint>
#include <functional>
#include <gsl/span>
#include <set>
#include <string>

/** DigitRawEncoder converts MCH digits into raw data. */

namespace o2::mch::raw
{

struct DigitRawEncoderOptions {
  std::string outputDir = ".";    // directory where the raw data files will be written
  bool chargeSumMode = true;      // whether to encode only charge sum or all samples (usually true)
  bool filePerLink = true;        // whether or not to output one file per link (usually true)
  bool userLogic = true;          // whether or not to use user logic (usually true)
  int userLogicVersion = 1;       // UL version (only relevant if userLogic=true)
  bool noEmptyHBF = false;        // disable writing of empty HBFs (for debug)
  bool noGRP = false;             // do not try to read GRP information (for debug or unit tests)
  bool dummyElecMap = false;      // use dummy electronic mapping (for debug only, temporary)
  bool writeHB = true;            // write Heatbeat headers at start of time frame
  int rawFileWriterVerbosity = 0; // verbosity of the RawFileWriter
  int rdhVersion = 6;             // RDH version to use
};

class DigitRawEncoder
{
 public:
  DigitRawEncoder(DigitRawEncoderOptions opt = {});

  void addHeartbeats(std::set<DsElecId> dsElecIds, uint32_t orbit);

  void encodeDigits(gsl::span<o2::mch::Digit> digits, uint32_t orbit, uint16_t bc);

  void writeConfig();

 private:
  DigitRawEncoderOptions mOptions;
  o2::raw::RawFileWriter mRawFileWriter;
  Solar2LinkInfo mSolar2LinkInfo;
  std::set<LinkInfo> mLinks;
  std::unique_ptr<PayloadEncoder> mPayloadEncoder;
  DigitPayloadEncoder mDigitPayloadEncoder;
};

} // namespace o2::mch::raw
#endif
