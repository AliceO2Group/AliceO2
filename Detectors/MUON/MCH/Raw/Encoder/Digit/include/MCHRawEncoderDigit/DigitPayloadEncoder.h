// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_DIGIT_PAYLOAD_ENCODER_H
#define O2_MCH_RAW_ENCODER_DIGIT_PAYLOAD_ENCODER_H

#include "DataFormatsMCH/Digit.h"
#include "MCHRawElecMap/Mapper.h"
#include <cstdint>
#include <functional>
#include <gsl/span>
#include <optional>
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "MCHRawEncoderDigit/Digit2ElecMapper.h"

/** DigitPayloadEncoder encodes MCH digits into memory buffers that
 * contain the MCH RawData Payload and a very slim information
 * about the {orbit,bc} for each payload. That {orbit,bc} is then
 * used by the PayloadPaginator to create Raw Data in the actual
 * Alice format, i.e. using RDH (RawDataHeader) + payload blocks.
 */

namespace o2::mch::raw
{

class DigitPayloadEncoder
{
 public:
  DigitPayloadEncoder(Digit2ElecMapper digit2elec, PayloadEncoder& encoder);

  void encodeDigits(gsl::span<o2::mch::Digit> digits,
                    uint32_t orbit,
                    uint16_t bc,
                    std::vector<std::byte>& buffer);

 private:
  Digit2ElecMapper mDigit2ElecMapper;
  PayloadEncoder& mEncoder;
};

} // namespace o2::mch::raw
#endif
