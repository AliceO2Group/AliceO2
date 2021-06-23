// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawEncoderDigit/DigitRawEncoder.h"

#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/Logger.h"
#include "Headers/DAQID.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHRawEncoderDigit/Digit2ElecMapper.h"
#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <set>

namespace o2::mch::raw
{
/** Prepare the rawfilewriter for work.
 *
 * @tparam ELECMAP : a type describing the electronic mapping of MCH
 * @tparam FORMAT : the output raw data format (Bare or UserLogic)
 * @tparam CHARGESUM : the output raw data mode (Sample or ChargeSum Mode)
 *
 * @param fw : the rawFileWriter to configure
 * @param opts : processing options
 */
void setupRawFileWriter(o2::raw::RawFileWriter& fw, const std::set<LinkInfo>& links, DigitRawEncoderOptions opt)
{
  fw.useRDHVersion(opt.rdhVersion);
  fw.setVerbosity(opt.rawFileWriterVerbosity);

  bool continuous = true;
  if (!opt.noGRP) {
    std::string inputGRP = o2::base::NameConf::getGRPFileName();
    std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(inputGRP)};
    continuous = grp->isDetContinuousReadOut(o2::detectors::DetID::MCH);
  }
  fw.setContinuousReadout(continuous); // must be set explicitly

  if (opt.noEmptyHBF) {
    fw.setDontFillEmptyHBF(true);
  }

  if (!std::filesystem::exists(opt.outputDir)) {
    if (!std::filesystem::create_directories(opt.outputDir)) {
      LOGP(fatal, "could not create output directory {}", opt.outputDir);
    } else {
      LOGP(info, "created output directory {}", opt.outputDir);
    }
  }

  std::string output = fmt::format("{:s}/mch", opt.outputDir);

  // Register the corresponding links (might have several solars for 1 link)
  registerLinks(fw, output, links, opt.filePerLink);
}

Solar2LinkInfo getSolar2LinkInfo(bool userLogic, bool dummyElecMap, int userLogicVersion)
{
  // the following cascade is not pretty, but the non-negociable intent is
  // to avoid exposing templates to the DigitRawEncoder interface, hence
  // this bool to template argument dance.

  if (dummyElecMap) {
    if (userLogic) {
      if (userLogicVersion == 0) {
        return createSolar2LinkInfo<ElectronicMapperDummy, UserLogicFormat, ChargeSumMode, 0>();
      } else if (userLogicVersion == 1) {
        return createSolar2LinkInfo<ElectronicMapperDummy, UserLogicFormat, ChargeSumMode, 1>();
      } else {
        throw std::invalid_argument("Version can only be 0 or 1");
      }
    } else {
      return createSolar2LinkInfo<ElectronicMapperDummy, BareFormat, ChargeSumMode, 0>();
    }
  }
  if (userLogic) {
    if (userLogicVersion == 0) {
      return createSolar2LinkInfo<ElectronicMapperGenerated, UserLogicFormat, ChargeSumMode, 0>();
    } else if (userLogicVersion == 1) {
      return createSolar2LinkInfo<ElectronicMapperGenerated, UserLogicFormat, ChargeSumMode, 1>();
    } else {
      throw std::invalid_argument("Version can only be 0 or 1");
    }
  } else {
    return createSolar2LinkInfo<ElectronicMapperGenerated, BareFormat, ChargeSumMode, 0>();
  }
}

Solar2FeeLinkMapper getSolar2FeeLink(bool dummyElecMap)
{
  if (dummyElecMap) {
    return createSolar2FeeLinkMapper<ElectronicMapperDummy>();
  }
  return createSolar2FeeLinkMapper<ElectronicMapperGenerated>();
}

std::set<uint16_t> getSolarUIDs(bool dummyElecMap)
{
  if (dummyElecMap) {
    return getSolarUIDs<ElectronicMapperDummy>();
  }
  return getSolarUIDs<ElectronicMapperGenerated>();
}

std::set<o2::mch::raw::LinkInfo> getLinks(Solar2LinkInfo solar2LinkInfo, bool dummyElecMap)
{
  // Get the list of solarIds and convert it to a list of unique RDH(Any)s
  //
  // Note that there's not necessarily a one to one correspondence
  // between solarIds and FEEID (hence RDH) :
  // the most probable format is UserLogic and in that case several
  // solars end up in the same RDH/FEEID (readout is basically
  // gathering solars per CRU endpoint)
  auto solarIds = getSolarUIDs(dummyElecMap);
  std::set<LinkInfo> links;
  for (auto solarId : solarIds) {
    auto li = solar2LinkInfo(solarId);
    if (!li.has_value()) {
      LOGP(FATAL, "Could not find information about solarId {:d}", solarId);
    }
    links.insert(li.value());
  }

  LOGP(INFO, "MCH: registered {:d} links for {:d} solars",
       links.size(),
       solarIds.size());
  return links;
}

Digit2ElecMapper getDigit2Elec(bool dummyElecMap)
{
  if (dummyElecMap) {
    return createDigit2ElecMapper(createDet2ElecMapper<ElectronicMapperDummy>());
  }
  return createDigit2ElecMapper(createDet2ElecMapper<ElectronicMapperGenerated>());
}

std::ostream& operator<<(std::ostream& os, const DigitRawEncoderOptions& opt)
{
  os << fmt::format("output dir {} filePerLink {} userLogic {} dummyElecMap {} ulVersion {}\n",
                    opt.outputDir, opt.filePerLink, opt.userLogic, opt.dummyElecMap, opt.userLogicVersion);
  return os;
}

DigitRawEncoder::DigitRawEncoder(DigitRawEncoderOptions opts)
  : mOptions{opts},
    mRawFileWriter{o2::header::DAQID(o2::header::DAQID::MCH).getO2Origin()},
    mSolar2LinkInfo{getSolar2LinkInfo(opts.userLogic, opts.dummyElecMap, opts.userLogicVersion)},
    mLinks{getLinks(mSolar2LinkInfo, opts.dummyElecMap)},
    mPayloadEncoder{createPayloadEncoder(
      getSolar2FeeLink(opts.dummyElecMap),
      opts.userLogic,
      opts.userLogicVersion,
      opts.chargeSumMode)},
    mDigitPayloadEncoder{getDigit2Elec(opts.dummyElecMap), *(mPayloadEncoder.get())}
{
  setupRawFileWriter(mRawFileWriter, mLinks, opts);
}

void DigitRawEncoder::addHeartbeats(std::set<DsElecId> dsElecIds, uint32_t orbit)
{
  LOGP(info, "Adding heartbeats for orbit={}", orbit);
  mPayloadEncoder->startHeartbeatFrame(orbit, 0);
  mPayloadEncoder->addHeartbeatHeaders(dsElecIds);
}

void DigitRawEncoder::encodeDigits(gsl::span<o2::mch::Digit> digits, uint32_t orbit, uint16_t bc)
{
  LOGP(info, "Encoding {} MCH digits for orbit {} bc {}", digits.size(), orbit, bc);
  std::vector<std::byte> buffer;
  mDigitPayloadEncoder.encodeDigits(digits, orbit, bc, buffer);
  paginate(mRawFileWriter, buffer, mLinks, mSolar2LinkInfo);
}

void DigitRawEncoder::writeConfig()
{
  mRawFileWriter.writeConfFile("MCH", "RAWDATA", fmt::format("{:s}/MCHraw.cfg", mOptions.outputDir));
}

} // namespace o2::mch::raw
