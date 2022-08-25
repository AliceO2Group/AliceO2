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

/// \file   CTFCoder.h
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of EMCAL data

#ifndef O2_EMCAL_CTFCODER_H
#define O2_EMCAL_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsEMCAL/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "EMCALReconstruction/CTFHelper.h"

class TTree;

namespace o2
{
namespace emcal
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::EMC) {}
  ~CTFCoder() final = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cell>& cellData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VCELL>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VTRG& trigVec, VCELL& cellVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  template <typename VEC>
  o2::ctf::CTFIOSize encode_impl(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cell>& cellData);
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Cell>& cellVec);
  std::vector<TriggerRecord> mTrgDataFilt;
  std::vector<Cell> mCellDataFilt;
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cell>& cellData)
{
  if (mIRFrameSelector.isSet()) { // preselect data
    mTrgDataFilt.clear();
    mCellDataFilt.clear();
    for (const auto& trig : trigData) {
      if (mIRFrameSelector.check(trig.getBCData()) >= 0) {
        mTrgDataFilt.push_back(trig);
        auto cellIt = cellData.begin() + trig.getFirstEntry();
        auto& trigC = mTrgDataFilt.back();
        trigC.setDataRange((int)mCellDataFilt.size(), trig.getNumberOfObjects());
        std::copy(cellIt, cellIt + trig.getNumberOfObjects(), std::back_inserter(mCellDataFilt));
      }
    }
    return encode_impl(buff, mTrgDataFilt, mCellDataFilt);
  }
  return encode_impl(buff, trigData, cellData);
}

template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode_impl(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cell>& cellData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncTrig
    MD::EENCODE, // BLC_orbitIncTrig
    MD::EENCODE, // BLC_entriesTrig
    MD::EENCODE, // BLC_towerID
    MD::EENCODE, // BLC_time
    MD::EENCODE, // BLC_energy
    MD::EENCODE, // BLC_status
    // extra slot was added in the end
    MD::EENCODE // BLC_trigger
  };

  CTFHelper helper(trigData, cellData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODEEMC(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  iosize += ENCODEEMC(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  iosize += ENCODEEMC(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  iosize += ENCODEEMC(helper.begin_entriesTrig(),  helper.end_entriesTrig(),   CTF::BLC_entriesTrig,  0);

  iosize += ENCODEEMC(helper.begin_towerID(),     helper.end_towerID(),      CTF::BLC_towerID,     0);
  iosize += ENCODEEMC(helper.begin_time(),        helper.end_time(),         CTF::BLC_time,        0);
  iosize += ENCODEEMC(helper.begin_energy(),      helper.end_energy(),       CTF::BLC_energy,      0);
  iosize += ENCODEEMC(helper.begin_status(),      helper.end_status(),       CTF::BLC_status,      0);
  // extra slot was added in the end
  iosize += ENCODEEMC(helper.begin_trigger(),  helper.end_trigger(),         CTF::BLC_trigger,     0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = sizeof(TriggerRecord) * trigData.size() + sizeof(Cell) * cellData.size();
  return iosize;
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VTRG, typename VCELL>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VCELL& cellVec)
{
  const auto& header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);
  std::vector<uint16_t> bcInc, entries, energy, cellTime, tower, trigger;
  std::vector<uint32_t> orbitInc;
  std::vector<uint8_t> status;

  o2::ctf::CTFIOSize iosize;
#define DECODEEMCAL(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  iosize += DECODEEMCAL(bcInc,       CTF::BLC_bcIncTrig);
  iosize += DECODEEMCAL(orbitInc,    CTF::BLC_orbitIncTrig);
  iosize += DECODEEMCAL(entries,     CTF::BLC_entriesTrig);
  iosize += DECODEEMCAL(tower,       CTF::BLC_towerID);

  iosize += DECODEEMCAL(cellTime,    CTF::BLC_time);
  iosize += DECODEEMCAL(energy,      CTF::BLC_energy);
  iosize += DECODEEMCAL(status,      CTF::BLC_status);
  // extra slot was added in the end
  iosize += DECODEEMCAL(trigger,     CTF::BLC_trigger);
  // triggers were added later, in old data they are absent:
  if (trigger.empty()) {
    trigger.resize(header.nTriggers);
  }
  //
  // clang-format on
  //
  trigVec.clear();
  cellVec.clear();
  trigVec.reserve(header.nTriggers);
  status.reserve(header.nCells);

  uint32_t firstEntry = 0, cellCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);

  Cell cell;
  TriggerRecord trg;
  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }

    firstEntry = cellVec.size();
    for (uint16_t ic = 0; ic < entries[itrig]; ic++) {
      cell.setPacked(tower[cellCount], cellTime[cellCount], energy[cellCount], status[cellCount]);
      cellVec.emplace_back(cell);
      cellCount++;
    }
    trg.setBCData(ir);
    trg.setDataRange(firstEntry, entries[itrig]);
    trg.setTriggerBitsCompressed(trigger[itrig]);
    trigVec.emplace_back(trg);
  }
  assert(cellCount == header.nCells);
  iosize.rawIn = sizeof(TriggerRecord) * trigVec.size() + sizeof(Cell) * cellVec.size();
  return iosize;
}

} // namespace emcal
} // namespace o2

#endif // O2_EMCAL_CTFCODER_H
