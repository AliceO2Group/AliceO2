// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CTFCoder.h
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of ZDC data

#ifndef O2_ZDC_CTFCODER_H
#define O2_ZDC_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsZDC/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "ZDCReconstruction/CTFHelper.h"

class TTree;

namespace o2
{
namespace zdc
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::ZDC) {}
  ~CTFCoder() = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const BCData>& trgData, const gsl::span<const ChannelData>& chanData, const gsl::span<const PedestalData>& pedData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VCHAN, typename VPED>
  void decode(const CTF::base& ec, VTRG& trigVec, VCHAN& chanVec, VPED& pedVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<BCData>& trigVec, std::vector<ChannelData>& chanVec, std::vector<PedestalData>& pedVec);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const BCData>& trigData, const gsl::span<const ChannelData>& chanData, const gsl::span<const PedestalData>& pedData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // _bcIncTrig
    MD::EENCODE, // _orbitIncTrig,
    MD::EENCODE, // _moduleTrig,
    MD::EENCODE, // _channelsHL,
    MD::EENCODE, // _triggersHL,
    MD::EENCODE, // _extTriggers,
    MD::EENCODE, // _nchanTrig,
    //
    MD::EENCODE, // _chanID,
    MD::EENCODE, // _chanData,
    //
    MD::EENCODE, // _orbitIncPed,
    MD::EENCODE, // _pedData
  };

  CTFHelper helper(trigData, chanData, pedData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEZDC(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEZDC(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  ENCODEZDC(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  ENCODEZDC(helper.begin_moduleTrig(),   helper.end_moduleTrig(),    CTF::BLC_moduleTrig,   0);
  ENCODEZDC(helper.begin_channelsHL(),   helper.end_channelsHL(),    CTF::BLC_channelsHL,   0);
  ENCODEZDC(helper.begin_triggersHL(),   helper.end_triggersHL(),    CTF::BLC_triggersHL,   0);
  ENCODEZDC(helper.begin_extTriggers(),  helper.end_extTriggers(),   CTF::BLC_extTriggers,  0);
  ENCODEZDC(helper.begin_nchanTrig(),    helper.end_nchanTrig(),     CTF::BLC_nchanTrig,    0);
  //
  ENCODEZDC(helper.begin_chanID(),       helper.end_chanID(),        CTF::BLC_chanID,       0);
  ENCODEZDC(helper.begin_chanData(),     helper.end_chanData(),      CTF::BLC_chanData,     0);  
  //
  ENCODEZDC(helper.begin_orbitIncPed(),  helper.end_orbitIncPed(),   CTF::BLC_orbitIncPed,  0);
  ENCODEZDC(helper.begin_pedData(),      helper.end_pedData(),       CTF::BLC_pedData,      0);

  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VTRG, typename VCHAN, typename VPED>
void CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VCHAN& chanVec, VPED& pedVec)
{
  auto header = ec.getHeader();
  ec.print(getPrefix());
  std::vector<uint16_t> bcIncTrig, moduleTrig, nchanTrig, chanData, pedData, triggersHL, channelsHL;
  std::vector<uint32_t> orbitIncTrig, orbitIncPed;
  std::vector<uint8_t> extTriggers, chanID;

#define DECODEZDC(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEZDC(bcIncTrig,      CTF::BLC_bcIncTrig);
  DECODEZDC(orbitIncTrig,   CTF::BLC_orbitIncTrig);
  DECODEZDC(moduleTrig,     CTF::BLC_moduleTrig);
  DECODEZDC(channelsHL,     CTF::BLC_channelsHL);
  DECODEZDC(triggersHL,     CTF::BLC_triggersHL);
  DECODEZDC(extTriggers,    CTF::BLC_extTriggers);
  DECODEZDC(nchanTrig,      CTF::BLC_nchanTrig);
  //
  DECODEZDC(chanID,         CTF::BLC_chanID);
  DECODEZDC(chanData,       CTF::BLC_chanData);
  //
  DECODEZDC(orbitIncPed,    CTF::BLC_orbitIncPed);
  DECODEZDC(pedData,        CTF::BLC_pedData);
  // clang-format on
  //
  trigVec.clear();
  chanVec.clear();
  pedVec.clear();
  trigVec.reserve(header.nTriggers);
  chanVec.reserve(header.nChannels);
  pedVec.reserve(header.nPedestals);

  // triggers and channels
  uint32_t firstEntry = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);
  auto chanDataIt = chanData.begin();
  auto chanIdIt = chanID.begin();
  auto modTrigIt = moduleTrig.begin();
  auto pedValIt = pedData.begin();
  auto channelsHLIt = channelsHL.begin();
  auto triggersHLIt = triggersHL.begin();

  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitIncTrig[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcIncTrig[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitIncTrig[itrig];
    } else {
      ir.bc += bcIncTrig[itrig];
    }

    auto firstChanEntry = chanVec.size();
    for (uint16_t ic = 0; ic < nchanTrig[itrig]; ic++) {
      auto& chan = chanVec.emplace_back();
      chan.id = *chanIdIt++;
      std::copy_n(chanDataIt, NTimeBinsPerBC, chan.data.begin());
      chanDataIt += NTimeBinsPerBC;
    }
    uint32_t chHL = (uint32_t(*channelsHLIt++) << 16) + *channelsHLIt++;
    uint32_t trHL = (uint32_t(*triggersHLIt++) << 16) + *triggersHLIt++;

    auto& bcTrig = trigVec.emplace_back(firstChanEntry, chanVec.size() - firstChanEntry, ir, chHL, trHL, extTriggers[itrig]);
    std::copy_n(modTrigIt, NModules, bcTrig.moduleTriggers.begin());
    modTrigIt += NModules;
  }

  // pedestal data
  ir = {o2::constants::lhc::LHCMaxBunches - 1, header.firstOrbitPed};
  for (uint32_t ip = 0; ip < header.nPedestals; ip++) {
    ir.orbit += orbitIncPed[ip];
    auto& ped = pedVec.emplace_back(PedestalData{ir, {}});
    std::copy_n(pedValIt, NChannels, ped.data.begin());
    pedValIt += NChannels;
  }
  // make sure whole data was used
  assert(chanDataIt == chanData.end());
  assert(chanIdIt == chanID.end());
  assert(modTrigIt == moduleTrig.end());
  assert(pedValIt == pedData.end());
  assert(channelsHLIt == channelsHL.end());
  assert(triggersHLIt == triggersHL.end());
}

} // namespace zdc
} // namespace o2

#endif // O2_ZDC_CTFCODER_H
