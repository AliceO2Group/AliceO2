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
/// \brief class for entropy encoding/decoding of CPV data

#ifndef O2_CPV_CTFCODER_H
#define O2_CPV_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsCPV/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "CPVReconstruction/CTFHelper.h"

class TTree;

namespace o2
{
namespace cpv
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::CPV) {}
  ~CTFCoder() final = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cluster>& cluData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VCLUSTER>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VTRG& trigVec, VCLUSTER& cluVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  template <typename VEC>
  o2::ctf::CTFIOSize encode_impl(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cluster>& cluData);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Cluster>& cluVec);
  std::vector<TriggerRecord> mTrgDataFilt;
  std::vector<Cluster> mClusDataFilt;
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cluster>& cluData)
{
  if (mIRFrameSelector.isSet()) { // preselect data
    mTrgDataFilt.clear();
    mClusDataFilt.clear();
    for (const auto& trig : trigData) {
      if (mIRFrameSelector.check(trig.getBCData()) >= 0) {
        mTrgDataFilt.push_back(trig);
        auto clusIt = cluData.begin() + trig.getFirstEntry();
        auto& trigC = mTrgDataFilt.back();
        trigC.setDataRange((int)mClusDataFilt.size(), trig.getNumberOfObjects());
        std::copy(clusIt, clusIt + trig.getNumberOfObjects(), std::back_inserter(mClusDataFilt));
      }
    }
    return encode_impl(buff, mTrgDataFilt, mClusDataFilt);
  }
  return encode_impl(buff, trigData, cluData);
}

template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode_impl(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cluster>& cluData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncTrig
    MD::EENCODE, // BLC_orbitIncTrig
    MD::EENCODE, // BLC_entriesTrig
    MD::EENCODE, // BLC_posX
    MD::EENCODE, // BLC_posZ
    MD::EENCODE, // BLC_energy
    MD::EENCODE  // BLC_status
  };

  CTFHelper helper(trigData, cluData);

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
#define ENCODECPV(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  iosize += ENCODECPV(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  iosize += ENCODECPV(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  iosize += ENCODECPV(helper.begin_entriesTrig(),  helper.end_entriesTrig(),   CTF::BLC_entriesTrig,  0);

  iosize += ENCODECPV(helper.begin_posX(),        helper.end_posX(),           CTF::BLC_posX,         0);
  iosize += ENCODECPV(helper.begin_posZ(),        helper.end_posZ(),           CTF::BLC_posZ,         0);
  iosize += ENCODECPV(helper.begin_energy(),      helper.end_energy(),         CTF::BLC_energy,       0);
  iosize += ENCODECPV(helper.begin_status(),      helper.end_status(),         CTF::BLC_status,       0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = trigData.size() * sizeof(TriggerRecord) + cluData.size() * sizeof(Cluster);
  return iosize;
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VTRG, typename VCLUSTER>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VCLUSTER& cluVec)
{
  auto header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);
  std::vector<uint16_t> bcInc, entries, posX, posZ;
  std::vector<uint32_t> orbitInc;
  std::vector<uint8_t> energy, status;

  o2::ctf::CTFIOSize iosize;
#define DECODECPV(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  iosize += DECODECPV(bcInc,       CTF::BLC_bcIncTrig);
  iosize += DECODECPV(orbitInc,    CTF::BLC_orbitIncTrig);
  iosize += DECODECPV(entries,     CTF::BLC_entriesTrig);
  iosize += DECODECPV(posX,        CTF::BLC_posX);
  iosize += DECODECPV(posZ,        CTF::BLC_posZ);
  iosize += DECODECPV(energy,      CTF::BLC_energy);
  iosize += DECODECPV(status,      CTF::BLC_status);
  // clang-format on
  //
  trigVec.clear();
  cluVec.clear();
  trigVec.reserve(header.nTriggers);
  status.reserve(header.nClusters);

  uint32_t firstEntry = 0, cluCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);
  bool checkIROK = (mBCShift == 0); // need to check if CTP offset correction does not make the local time negative ?
  Cluster clu;
  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }
    if (checkIROK || canApplyBCShift(ir)) { // correction will be ok
      checkIROK = true;
    } else { // correction would make IR prior to mFirstTFOrbit, skip
      cluCount += entries[itrig];
      continue;
    }
    firstEntry = cluVec.size();
    for (uint16_t ic = 0; ic < entries[itrig]; ic++) {
      clu.setPacked(posX[cluCount], posZ[cluCount], energy[cluCount], status[cluCount]);
      cluVec.emplace_back(clu);
      cluCount++;
    }
    trigVec.emplace_back(ir - mBCShift, firstEntry, entries[itrig]);
  }
  assert(cluCount == header.nClusters);
  iosize.rawIn = trigVec.size() * sizeof(TriggerRecord) + cluVec.size() * sizeof(Cluster);
  return iosize;
}

} // namespace cpv
} // namespace o2

#endif // O2_CPV_CTFCODER_H
