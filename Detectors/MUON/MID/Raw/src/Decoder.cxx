// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/Decoder.cxx
/// \brief  MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Decoder.h"

#include "Headers/RDHAny.h"
#include "DPLUtils/RawParser.h"

namespace o2
{
namespace mid
{

// namespace impl
// {
class FEEIDGetterImpl
{
 public:
  FEEIDGetterImpl(const FEEIdConfig& feeIdConfig) : mFeeIdConfig(feeIdConfig) {}
  uint16_t operator()(const o2::header::RDHAny& rdh) { return mFeeIdConfig.getFeeId(o2::raw::RDHUtils::getLinkID(rdh), o2::raw::RDHUtils::getEndPointID(rdh), o2::raw::RDHUtils::getCRUID(rdh)); }

 private:
  FEEIdConfig mFeeIdConfig{};
};
// } // namespace impl

Decoder::Decoder(bool isDebugMode, bool isBare, const ElectronicsDelay& electronicsDelay, const CrateMasks& crateMasks, const FEEIdConfig& feeIdConfig) : mData(), mROFRecords(), mGBTDecoders()
{
  /// Constructor
  for (uint16_t igbt = 0; igbt < crateparams::sNGBTs; ++igbt) {
    mGBTDecoders[igbt] = createGBTDecoder(igbt, isBare, isDebugMode, crateMasks.getMask(igbt), electronicsDelay);
  }
  if (isBare) {
    mGetFEEID = FEEIDGetterImpl(feeIdConfig);
  }
}

void Decoder::clear()
{
  /// Clears the decoded data
  mData.clear();
  mROFRecords.clear();
}

void Decoder::process(gsl::span<const uint8_t> bytes)
{
  /// Decodes the buffer
  clear();
  o2::framework::RawParser parser(bytes.data(), bytes.size());
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    if (it.size() == 0) {
      continue;
    }
    gsl::span<const uint8_t> payload(it.data(), it.size());
    auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
    process(payload, *rdhPtr);
  }
}

std::unique_ptr<Decoder> createDecoder(const o2::header::RDHAny& rdh, bool isDebugMode, ElectronicsDelay& electronicsDelay, const CrateMasks& crateMasks, const FEEIdConfig& feeIdConfig)
{
  /// Creates the decoder from the RDH info
  bool isBare = (o2::raw::RDHUtils::getLinkID(rdh) != raw::sUserLogicLinkID);
  return std::make_unique<Decoder>(isDebugMode, isBare, electronicsDelay, crateMasks, feeIdConfig);
}
std::unique_ptr<Decoder> createDecoder(const o2::header::RDHAny& rdh, bool isDebugMode, const char* electronicsDelayFile, const char* crateMasksFile, const char* feeIdConfigFile)
{
  /// Creates the decoder from the RDH info
  o2::mid::ElectronicsDelay electronicsDelay;
  std::string filename = electronicsDelayFile;
  if (!filename.empty()) {
    electronicsDelay = o2::mid::readElectronicsDelay(filename.c_str());
  }

  o2::mid::CrateMasks crateMasks;
  filename = crateMasksFile;
  if (!filename.empty()) {
    crateMasks = o2::mid::CrateMasks(filename.c_str());
  }

  o2::mid::FEEIdConfig feeIdConfig;
  filename = feeIdConfigFile;
  if (!filename.empty()) {
    feeIdConfig = o2::mid::FEEIdConfig(filename.c_str());
  }

  return createDecoder(rdh, isDebugMode, electronicsDelay, crateMasks, feeIdConfig);
}

} // namespace mid
} // namespace o2
