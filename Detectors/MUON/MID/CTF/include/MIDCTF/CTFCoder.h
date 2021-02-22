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
/// \brief class for entropy encoding/decoding of MID column data

#ifndef O2_MID_CTFCODER_H
#define O2_MID_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsMID/CTF.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ColumnData.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "MIDCTF/CTFHelper.h"

class TTree;

namespace o2
{
namespace mid
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::MID) {}
  ~CTFCoder() = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const ColumnData>& colData);

  /// entropy decode data from buffer with CTF
  template <typename VROF, typename VCOL>
  void decode(const CTF::base& ec, VROF& rofVec, VCOL& colVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofVec, std::vector<ColumnData>& colVec);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const ColumnData>& colData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncROF
    MD::EENCODE, // BLC_orbitIncROF
    MD::EENCODE, // BLC_entriesROF
    MD::EENCODE, // BLC_evtypeROF
    MD::EENCODE, // BLC_pattern
    MD::EENCODE, // BLC_deId
    MD::EENCODE  // BLC_colId
  };
  CTFHelper helper(rofData, colData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEMID(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEMID(helper.begin_bcIncROF(),    helper.end_bcIncROF(),     CTF::BLC_bcIncROF,    0);
  ENCODEMID(helper.begin_orbitIncROF(), helper.end_orbitIncROF(),  CTF::BLC_orbitIncROF, 0);
  ENCODEMID(helper.begin_entriesROF(),  helper.end_entriesROF(),   CTF::BLC_entriesROF,  0);
  ENCODEMID(helper.begin_evtypeROF(),   helper.end_evtypeROF(),    CTF::BLC_evtypeROF,   0);

  ENCODEMID(helper.begin_pattern(),     helper.end_pattern(),      CTF::BLC_pattern,     0);
  ENCODEMID(helper.begin_deId(),        helper.end_deId(),         CTF::BLC_deId,        0);
  ENCODEMID(helper.begin_colId(),       helper.end_colId(),        CTF::BLC_colId,       0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VROF, typename VCOL>
void CTFCoder::decode(const CTF::base& ec, VROF& rofVec, VCOL& colVec)
{
  auto header = ec.getHeader();
  ec.print(getPrefix());
  std::vector<uint16_t> bcInc, entries, pattern;
  std::vector<uint32_t> orbitInc;
  std::vector<uint8_t> evType, deId, colId;

#define DECODEMID(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEMID(bcInc,       CTF::BLC_bcIncROF);
  DECODEMID(orbitInc,    CTF::BLC_orbitIncROF);
  DECODEMID(entries,     CTF::BLC_entriesROF);
  DECODEMID(evType,      CTF::BLC_evtypeROF);

  DECODEMID(pattern,     CTF::BLC_pattern);
  DECODEMID(deId,        CTF::BLC_deId);
  DECODEMID(colId,       CTF::BLC_colId);
  // clang-format on
  //
  rofVec.clear();
  colVec.clear();
  rofVec.reserve(header.nROFs);
  colVec.reserve(header.nColumns);

  uint32_t firstEntry = 0, rofCount = 0, colCount = 0, pCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);

  for (uint32_t irof = 0; irof < header.nROFs; irof++) {
    // restore ROFRecord
    if (orbitInc[irof]) {  // non-0 increment => new orbit
      ir.bc = bcInc[irof]; // bcInc has absolute meaning
      ir.orbit += orbitInc[irof];
    } else {
      ir.bc += bcInc[irof];
    }

    firstEntry = colVec.size();
    for (uint8_t ic = 0; ic < entries[irof]; ic++) {
      colVec.emplace_back(ColumnData{deId[colCount], colId[colCount], std::array{pattern[pCount], pattern[pCount + 1], pattern[pCount + 2], pattern[pCount + 3], pattern[pCount + 4]}});
      pCount += 5;
      colCount++;
    }
    rofVec.emplace_back(ROFRecord{ir, EventType(evType[irof]), firstEntry, entries[irof]});
  }
  assert(colCount == header.nColumns);
}

} // namespace mid
} // namespace o2

#endif // O2_MID_CTFCODER_H
