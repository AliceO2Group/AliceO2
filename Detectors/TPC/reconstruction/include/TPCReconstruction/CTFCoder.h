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
/// \brief class for entropy encoding/decoding of TPC compressed clusters data

#ifndef O2_TPC_CTFCODER_H
#define O2_TPC_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <cassert>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace tpc
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::TPC) {}
  ~CTFCoder() = default;

  /// entropy-encode compressed clusters to flat buffer
  template <typename VEC>
  void encode(VEC& buff, const CompressedClusters& ccl);

  template <typename VEC>
  void decode(const CTF::base& ec, VEC& buff);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);
  size_t estimateCompressedSize(const CompressedClusters& ccl);

  static size_t constexpr Alignment = 16;
  static size_t estimateSize(CompressedClusters& c);
  static void setCompClusAddresses(CompressedClusters& c, void*& buff);

  template <size_t ALG = Alignment, typename T>
  static size_t alignSize(T*& var, size_t n = 1)
  {
    auto sz = sizeof(T) * n;
    auto res = sz % ALG;
    return res ? sz + (ALG - res) : sz;
  }

  template <size_t ALG = Alignment, typename T>
  static void setAlignedPtr(void*& ptr, T*& var, size_t n = 1)
  {
    auto sz = sizeof(T) * n;
    auto res = sz % ALG;
    var = reinterpret_cast<T*>(ptr);
    auto& ptrR = reinterpret_cast<size_t&>(ptr);
    ptrR += res ? sz + (ALG - res) : sz;
  }

  bool getCombineColumns() const { return mCombineColumns; }
  void setCombineColumns(bool v) { mCombineColumns = v; }

 private:
  void checkDataDictionaryConsistency(const CTFHeader& h);

  template <int NU, int NL>
  static constexpr auto MVAR()
  {
    typename std::conditional<(NU + NL > 16), uint32_t, typename std::conditional<(NU + NL > 8), uint16_t, uint8_t>::type>::type tp = 0;
    return tp;
  }
  template <int NU, int NL>
  static constexpr auto MPTR()
  {
    typename std::conditional<(NU + NL > 16), uint32_t, typename std::conditional<(NU + NL > 8), uint16_t, uint8_t>::type>::type* tp = nullptr;
    return tp;
  }
#define MTYPE(A, B) decltype(CTFCoder::MVAR<A, B>())

  template <int NU, int NL, typename CU, typename CL>
  static void splitColumns(const std::vector<MTYPE(NU, NL)>& vm, CU*& vu, CL*& vl);

  template <int NU, int NL, typename CU, typename CL>
  static auto mergeColumns(const CU* vu, const CL* vl, size_t nelem);

  bool mCombineColumns = false; // combine correlated columns

  ClassDefNV(CTFCoder, 1);
};

/// split words of input vector to columns assigned to provided pointers (memory must be allocated in advance)
template <int NU, int NL, typename CU, typename CL>
void CTFCoder::splitColumns(const std::vector<MTYPE(NU, NL)>& vm, CU*& vu, CL*& vl)
{
  static_assert(NU <= sizeof(CU) * 8 && NL <= sizeof(CL) * 8, "output columns bit count is wrong");
  size_t n = vm.size();
  for (size_t i = 0; i < n; i++) {
    vu[i] = static_cast<CU>(vm[i] >> NL);
    vl[i] = static_cast<CL>(vm[i] & ((0x1 << NL) - 1));
  }
}

/// merge elements of 2 columns pointed by vu and vl to a single vector with wider field
template <int NU, int NL, typename CU, typename CL>
auto CTFCoder::mergeColumns(const CU* vu, const CL* vl, size_t nelem)
{
  // merge 2 columns to 1
  static_assert(NU <= sizeof(NU) * 8 && NL <= sizeof(NL) * 8, "input columns bit count is wrong");
  std::vector<MTYPE(NU, NL)> outv;
  outv.reserve(nelem);
  for (size_t i = 0; i < nelem; i++) {
    outv.push_back((static_cast<MTYPE(NU, NL)>(vu[i]) << NL) | static_cast<MTYPE(NU, NL)>(vl[i]));
  }
  return std::move(outv);
}

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const CompressedClusters& ccl)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctf::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, //qTotA
    MD::EENCODE, //qMaxA
    MD::EENCODE, //flagsA
    MD::EENCODE, //rowDiffA
    MD::EENCODE, //sliceLegDiffA
    MD::EENCODE, //padResA
    MD::EENCODE, //timeResA
    MD::EENCODE, //sigmaPadA
    MD::EENCODE, //sigmaTimeA
    MD::EENCODE, //qPtA
    MD::EENCODE, //rowA
    MD::EENCODE, //sliceA
    MD::EENCODE, //timeA
    MD::EENCODE, //padA
    MD::EENCODE, //qTotU
    MD::EENCODE, //qMaxU
    MD::EENCODE, //flagsU
    MD::EENCODE, //padDiffU
    MD::EENCODE, //timeDiffU
    MD::EENCODE, //sigmaPadU
    MD::EENCODE, //sigmaTimeU
    MD::EENCODE, //nTrackClusters
    MD::EENCODE  //nSliceRowClusters
  };

  // book output size with some margin
  auto szIni = estimateCompressedSize(ccl);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  uint32_t flags = 0;
  if (mCombineColumns) {
    flags |= CTFHeader::CombinedColumns;
  }
  ec->setHeader(CTFHeader{reinterpret_cast<const CompressedClustersCounters&>(ccl), flags});
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODETPC(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off

  if (mCombineColumns) {
    auto mrg = mergeColumns<CTF::NBitsQTot, CTF::NBitsQMax>(ccl.qTotA, ccl.qMaxA, ccl.nAttachedClusters);
    ENCODETPC(&mrg[0],             (&mrg[0]) + ccl.nAttachedClusters,                CTF::BLCqTotA,             0);
  }
  else {
    ENCODETPC(ccl.qTotA,           ccl.qTotA + ccl.nAttachedClusters,                CTF::BLCqTotA,             0);
  }
  ENCODETPC(ccl.qMaxA,             ccl.qMaxA + (mCombineColumns ? 0 : ccl.nAttachedClusters), CTF::BLCqMaxA,    0);
  
  ENCODETPC(ccl.flagsA,            ccl.flagsA + ccl.nAttachedClusters,               CTF::BLCflagsA,            0);
  
  if (mCombineColumns) {
    auto mrg = mergeColumns<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>(ccl.rowDiffA, ccl.sliceLegDiffA, ccl.nAttachedClustersReduced);    
    ENCODETPC(&mrg[0],             (&mrg[0]) + ccl.nAttachedClustersReduced,         CTF::BLCrowDiffA,          0);
  }
  else {
    ENCODETPC(ccl.rowDiffA,        ccl.rowDiffA + ccl.nAttachedClustersReduced,      CTF::BLCrowDiffA,          0);
  }
  ENCODETPC(ccl.sliceLegDiffA,     ccl.sliceLegDiffA + (mCombineColumns ? 0 : ccl.nAttachedClustersReduced), CTF::BLCsliceLegDiffA, 0);

  ENCODETPC(ccl.padResA,           ccl.padResA + ccl.nAttachedClustersReduced,       CTF::BLCpadResA,           0);
  ENCODETPC(ccl.timeResA,          ccl.timeResA + ccl.nAttachedClustersReduced,      CTF::BLCtimeResA,          0);

  if (mCombineColumns) {
    auto mrg = mergeColumns<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>(ccl.sigmaPadA, ccl.sigmaTimeA, ccl.nAttachedClusters);
    ENCODETPC(&mrg[0],             &mrg[0] + ccl.nAttachedClusters,                  CTF::BLCsigmaPadA,         0);
  }
  else {
    ENCODETPC(ccl.sigmaPadA,       ccl.sigmaPadA + ccl.nAttachedClusters,            CTF::BLCsigmaPadA,         0);
  }
  ENCODETPC(ccl.sigmaTimeA,        ccl.sigmaTimeA + (mCombineColumns ? 0 : ccl.nAttachedClusters), CTF::BLCsigmaTimeA, 0);  

  ENCODETPC(ccl.qPtA,              ccl.qPtA + ccl.nTracks,                           CTF::BLCqPtA,              0);
  ENCODETPC(ccl.rowA,              ccl.rowA + ccl.nTracks,                           CTF::BLCrowA,              0);
  ENCODETPC(ccl.sliceA,            ccl.sliceA + ccl.nTracks,                         CTF::BLCsliceA,            0);
  ENCODETPC(ccl.timeA,             ccl.timeA + ccl.nTracks,                          CTF::BLCtimeA,             0);
  ENCODETPC(ccl.padA,              ccl.padA + ccl.nTracks,                           CTF::BLCpadA,              0);

  if (mCombineColumns) {
    auto mrg = mergeColumns<CTF::NBitsQTot, CTF::NBitsQMax>(ccl.qTotU,ccl.qMaxU,ccl.nUnattachedClusters);
    ENCODETPC(&mrg[0],             &mrg[0] + ccl.nUnattachedClusters,                CTF::BLCqTotU,             0);
  }
  else {    
    ENCODETPC(ccl.qTotU,           ccl.qTotU + ccl.nUnattachedClusters,              CTF::BLCqTotU,             0);
  }
  ENCODETPC(ccl.qMaxU,             ccl.qMaxU + (mCombineColumns ? 0 : ccl.nUnattachedClusters), CTF::BLCqMaxU,  0);

  ENCODETPC(ccl.flagsU,            ccl.flagsU + ccl.nUnattachedClusters,             CTF::BLCflagsU,            0);
  ENCODETPC(ccl.padDiffU,          ccl.padDiffU + ccl.nUnattachedClusters,           CTF::BLCpadDiffU,          0);
  ENCODETPC(ccl.timeDiffU,         ccl.timeDiffU + ccl.nUnattachedClusters,          CTF::BLCtimeDiffU,         0);

  if (mCombineColumns) {
    auto mrg = mergeColumns<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>(ccl.sigmaPadU, ccl.sigmaTimeU, ccl.nUnattachedClusters);
    ENCODETPC(&mrg[0],             &mrg[0] + ccl.nUnattachedClusters,                CTF::BLCsigmaPadU,         0);
  }
  else {
    ENCODETPC(ccl.sigmaPadU,       ccl.sigmaPadU + ccl.nUnattachedClusters,          CTF::BLCsigmaPadU,         0);
  }
  ENCODETPC(ccl.sigmaTimeU,        ccl.sigmaTimeU + (mCombineColumns ? 0 : ccl.nUnattachedClusters), CTF::BLCsigmaTimeU, 0);

  ENCODETPC(ccl.nTrackClusters,    ccl.nTrackClusters + ccl.nTracks,                 CTF::BLCnTrackClusters,    0);
  ENCODETPC(ccl.nSliceRowClusters, ccl.nSliceRowClusters + ccl.nSliceRows,           CTF::BLCnSliceRowClusters, 0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded bloks to TPC CompressedClusters into the externally provided vector (e.g. PMR vector from DPL)
template <typename VEC>
void CTFCoder::decode(const CTF::base& ec, VEC& buffVec)
{
  CompressedClusters cc;
  CompressedClustersCounters& ccCount = cc;
  auto& header = ec.getHeader();
  checkDataDictionaryConsistency(header);
  ccCount = reinterpret_cast<const CompressedClustersCounters&>(header);
  CompressedClustersFlat* ccFlat = nullptr;
  size_t sizeCFlatBody = alignSize(ccFlat);
  size_t sz = sizeCFlatBody + estimateSize(cc);                                             // total size of the buffVec accounting for the alignment
  size_t vsz = sizeof(typename std::remove_reference<decltype(buffVec)>::type::value_type); // size of the element of the buffer
  buffVec.resize(sz / vsz);
  ccFlat = reinterpret_cast<CompressedClustersFlat*>(buffVec.data());                           // RS? do we need to align this pointer, or PMR vector will be already aligned?
  auto buff = reinterpret_cast<void*>(reinterpret_cast<char*>(buffVec.data()) + sizeCFlatBody); // will be the start of the CompressedClustersFlat payload

  setCompClusAddresses(cc, buff);
  ccFlat->set(sz, cc); // set offsets
  ec.print(getPrefix());

  // decode encoded data directly to destination buff
#define DECODETPC(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  if (mCombineColumns) {
    std::vector<MTYPE(CTF::NBitsQTot, CTF::NBitsQMax)> mrg;
    DECODETPC(mrg,                  CTF::BLCqTotA);
    splitColumns<CTF::NBitsQTot, CTF::NBitsQMax>(mrg, cc.qTotA, cc.qMaxA);
  }
  else {
    DECODETPC(cc.qTotA,             CTF::BLCqTotA);
    DECODETPC(cc.qMaxA,             CTF::BLCqMaxA);
  }
  
  DECODETPC(cc.flagsA,              CTF::BLCflagsA);
  
  if (mCombineColumns) {
    std::vector<MTYPE(CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff)> mrg;
    DECODETPC(mrg,                  CTF::BLCrowDiffA);
    splitColumns<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>(mrg, cc.rowDiffA, cc.sliceLegDiffA);
  }
  else {
    DECODETPC(cc.rowDiffA,          CTF::BLCrowDiffA);
    DECODETPC(cc.sliceLegDiffA,     CTF::BLCsliceLegDiffA);
  }
  
  DECODETPC(cc.padResA,             CTF::BLCpadResA);
  DECODETPC(cc.timeResA,            CTF::BLCtimeResA);

  if (mCombineColumns) {
    std::vector<MTYPE(CTF::NBitsSigmaPad, CTF::NBitsSigmaTime)> mrg;
    DECODETPC(mrg,                  CTF::BLCsigmaPadA);
    splitColumns<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>(mrg, cc.sigmaPadA, cc.sigmaTimeA);
  }
  else { 
    DECODETPC(cc.sigmaPadA,         CTF::BLCsigmaPadA);
    DECODETPC(cc.sigmaTimeA,        CTF::BLCsigmaTimeA);
  }
  
  DECODETPC(cc.qPtA,                CTF::BLCqPtA);
  DECODETPC(cc.rowA,                CTF::BLCrowA);
  DECODETPC(cc.sliceA,              CTF::BLCsliceA);
  DECODETPC(cc.timeA,               CTF::BLCtimeA);
  DECODETPC(cc.padA,                CTF::BLCpadA);

  if (mCombineColumns) {
    std::vector<MTYPE(CTF::NBitsQTot, CTF::NBitsQMax)> mrg;
    DECODETPC(mrg,                  CTF::BLCqTotU);
    splitColumns<CTF::NBitsQTot, CTF::NBitsQMax>(mrg, cc.qTotU, cc.qMaxU);
  }
  else {
    DECODETPC(cc.qTotU,             CTF::BLCqTotU);
    DECODETPC(cc.qMaxU,             CTF::BLCqMaxU);
  }

  DECODETPC(cc.flagsU,              CTF::BLCflagsU);
  DECODETPC(cc.padDiffU,            CTF::BLCpadDiffU);
  DECODETPC(cc.timeDiffU,           CTF::BLCtimeDiffU);

  if (mCombineColumns) {
    std::vector<MTYPE(CTF::NBitsSigmaPad, CTF::NBitsSigmaTime)> mrg;
    DECODETPC(mrg,                  CTF::BLCsigmaPadU);
    splitColumns<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>(mrg, cc.sigmaPadU, cc.sigmaTimeU);
  }
  else {
    DECODETPC(cc.sigmaPadU,         CTF::BLCsigmaPadU);
    DECODETPC(cc.sigmaTimeU,        CTF::BLCsigmaTimeU);
  }
  
  DECODETPC(cc.nTrackClusters,      CTF::BLCnTrackClusters);
  DECODETPC(cc.nSliceRowClusters,   CTF::BLCnSliceRowClusters);
  // clang-format on
}
#undef MTYPE

} // namespace tpc
} // namespace o2

#endif // O2_TPC_CTFCODER_H
