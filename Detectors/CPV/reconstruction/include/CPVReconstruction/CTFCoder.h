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
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::CPV) {}
  ~CTFCoder() = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cluster>& cluData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VCLUSTER>
  void decode(const CTF::base& ec, VTRG& trigVec, VCLUSTER& cluVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Cluster>& cluVec);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Cluster>& cluData)
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
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODECPV(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODECPV(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  ENCODECPV(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  ENCODECPV(helper.begin_entriesTrig(),  helper.end_entriesTrig(),   CTF::BLC_entriesTrig,  0);
  
  ENCODECPV(helper.begin_posX(),        helper.end_posX(),           CTF::BLC_posX,         0);
  ENCODECPV(helper.begin_posZ(),        helper.end_posZ(),           CTF::BLC_posZ,         0);
  ENCODECPV(helper.begin_energy(),      helper.end_energy(),         CTF::BLC_energy,       0);
  ENCODECPV(helper.begin_status(),      helper.end_status(),         CTF::BLC_status,       0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VTRG, typename VCLUSTER>
void CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VCLUSTER& cluVec)
{
  auto header = ec.getHeader();
  ec.print(getPrefix());
  std::vector<uint16_t> bcInc, entries, posX, posZ;
  std::vector<uint32_t> orbitInc;
  std::vector<uint8_t> energy, status;

#define DECODECPV(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODECPV(bcInc,       CTF::BLC_bcIncTrig);
  DECODECPV(orbitInc,    CTF::BLC_orbitIncTrig);
  DECODECPV(entries,     CTF::BLC_entriesTrig);
  DECODECPV(posX,        CTF::BLC_posX);
  DECODECPV(posZ,        CTF::BLC_posZ);
  DECODECPV(energy,      CTF::BLC_energy);
  DECODECPV(status,      CTF::BLC_status);
  // clang-format on
  //
  trigVec.clear();
  cluVec.clear();
  trigVec.reserve(header.nTriggers);
  status.reserve(header.nClusters);

  uint32_t firstEntry = 0, cluCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);

  Cluster clu;
  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }

    firstEntry = cluVec.size();
    for (uint16_t ic = 0; ic < entries[itrig]; ic++) {
      clu.setPacked(posX[cluCount], posZ[cluCount], energy[cluCount], status[cluCount]);
      cluVec.emplace_back(clu);
      cluCount++;
    }
    trigVec.emplace_back(ir, firstEntry, entries[itrig]);
  }
  assert(cluCount == header.nClusters);
}

} // namespace cpv
} // namespace o2

#endif // O2_CPV_CTFCODER_H
