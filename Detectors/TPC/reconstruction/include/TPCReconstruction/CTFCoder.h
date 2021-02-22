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
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "rANS/utils.h"

class TTree;

namespace o2
{
namespace tpc
{

namespace detail
{

template <int A, int B>
struct combinedType {
  using type = std::conditional_t<(A + B > 16), uint32_t, std::conditional_t<(A + B > 8), uint16_t, uint8_t>>;
};

template <int A, int B>
using combinedType_t = typename combinedType<A, B>::type;

template <typename value_T, size_t shift>
class ShiftFunctor
{
 public:
  template <typename iterA_T, typename iterB_T>
  inline value_T operator()(iterA_T iterA, iterB_T iterB) const
  {
    return *iterB + (static_cast<value_T>(*iterA) << shift);
  };

  template <typename iterA_T, typename iterB_T>
  inline void operator()(iterA_T iterA, iterB_T iterB, value_T value) const
  {
    *iterA = value >> shift;
    *iterB = value & ((0x1 << shift) - 0x1);
  };
};

template <typename iterA_T, typename iterB_T, typename F>
auto makeInputIterators(iterA_T iterA, iterB_T iterB, size_t nElements, F functor)
{
  using namespace o2::rans::utils;

  auto advanceIter = [](auto iter, size_t nElements) {
    auto tmp = iter;
    std::advance(tmp, nElements);
    return tmp;
  };

  return std::make_tuple(CombinedInputIterator{iterA, iterB, functor},
                         CombinedInputIterator{advanceIter(iterA, nElements), advanceIter(iterB, nElements), functor});
};

template <int bits_A, int bits_B>
struct MergedColumnsDecoder {

  using combined_t = combinedType_t<bits_A, bits_B>;

  template <typename iterA_T, typename iterB_T, typename F>
  static void decode(iterA_T iterA, iterB_T iterB, CTF::Slots slot, F decodingFunctor)
  {
    ShiftFunctor<combined_t, bits_B> f{};
    auto iter = rans::utils::CombinedOutputIteratorFactory<combined_t>::makeIter(iterA, iterB, f);

    decodingFunctor(iter, slot);
  }
};

} // namespace detail

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::TPC) {}

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

  template <int NU, int NL, typename CU, typename CL>
  static void splitColumns(const std::vector<detail::combinedType_t<NU, NL>>& vm, CU*& vu, CL*& vl);

  template <typename source_T>
  void buildCoder(ctf::CTFCoderBase::OpType coderType, const CTF::container_t& ctf, CTF::Slots slot);

  bool mCombineColumns = false; // combine correlated columns

  ClassDefNV(CTFCoder, 1);
};

template <typename source_T>
void CTFCoder::buildCoder(ctf::CTFCoderBase::OpType coderType, const CTF::container_t& ctf, CTF::Slots slot)
{
  auto buildFrequencyTable = [](const CTF::container_t& ctf, CTF::Slots slot) -> rans::FrequencyTable {
    rans::FrequencyTable frequencyTable;
    auto block = ctf.getBlock(slot);
    auto metaData = ctf.getMetadata(slot);
    frequencyTable.addFrequencies(block.getDict(), block.getDict() + block.getNDict(), metaData.min, metaData.max);
    return frequencyTable;
  };
  auto getProbabilityBits = [](const CTF::container_t& ctf, CTF::Slots slot) -> int {
    return ctf.getMetadata(slot).probabilityBits;
  };

  this->createCoder<source_T>(coderType, buildFrequencyTable(ctf, slot), getProbabilityBits(ctf, slot), static_cast<int>(slot));
}

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const CompressedClusters& ccl)
{
  using MD = o2::ctf::Metadata::OptStore;
  using namespace detail;
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

  auto encodeTPC = [&buff, &optField, &coders = mCoders](auto begin, auto end, CTF::Slots slot, size_t probabilityBits) {
    // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
    const auto slotVal = static_cast<int>(slot);
    CTF::get(buff.data())->encode(begin, end, slotVal, probabilityBits, optField[slotVal], &buff, coders[slotVal].get());
  };

  if (mCombineColumns) {
    const auto [begin, end] = makeInputIterators(ccl.qTotA, ccl.qMaxA, ccl.nAttachedClusters,
                                                 ShiftFunctor<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>, CTF::NBitsQMax>{});
    encodeTPC(begin, end, CTF::BLCqTotA, 0);
  } else {
    encodeTPC(ccl.qTotA, ccl.qTotA + ccl.nAttachedClusters, CTF::BLCqTotA, 0);
  }
  encodeTPC(ccl.qMaxA, ccl.qMaxA + (mCombineColumns ? 0 : ccl.nAttachedClusters), CTF::BLCqMaxA, 0);

  encodeTPC(ccl.flagsA, ccl.flagsA + ccl.nAttachedClusters, CTF::BLCflagsA, 0);

  if (mCombineColumns) {
    const auto [begin, end] = makeInputIterators(ccl.rowDiffA, ccl.sliceLegDiffA, ccl.nAttachedClustersReduced,
                                                 ShiftFunctor<combinedType_t<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>, CTF::NBitsSliceLegDiff>{});
    encodeTPC(begin, end, CTF::BLCrowDiffA, 0);
  } else {
    encodeTPC(ccl.rowDiffA, ccl.rowDiffA + ccl.nAttachedClustersReduced, CTF::BLCrowDiffA, 0);
  }
  encodeTPC(ccl.sliceLegDiffA, ccl.sliceLegDiffA + (mCombineColumns ? 0 : ccl.nAttachedClustersReduced), CTF::BLCsliceLegDiffA, 0);

  encodeTPC(ccl.padResA, ccl.padResA + ccl.nAttachedClustersReduced, CTF::BLCpadResA, 0);
  encodeTPC(ccl.timeResA, ccl.timeResA + ccl.nAttachedClustersReduced, CTF::BLCtimeResA, 0);

  if (mCombineColumns) {
    const auto [begin, end] = makeInputIterators(ccl.sigmaPadA, ccl.sigmaTimeA, ccl.nAttachedClusters,
                                                 ShiftFunctor<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>, CTF::NBitsSigmaTime>{});
    encodeTPC(begin, end, CTF::BLCsigmaPadA, 0);
  } else {
    encodeTPC(ccl.sigmaPadA, ccl.sigmaPadA + ccl.nAttachedClusters, CTF::BLCsigmaPadA, 0);
  }
  encodeTPC(ccl.sigmaTimeA, ccl.sigmaTimeA + (mCombineColumns ? 0 : ccl.nAttachedClusters), CTF::BLCsigmaTimeA, 0);

  encodeTPC(ccl.qPtA, ccl.qPtA + ccl.nTracks, CTF::BLCqPtA, 0);
  encodeTPC(ccl.rowA, ccl.rowA + ccl.nTracks, CTF::BLCrowA, 0);
  encodeTPC(ccl.sliceA, ccl.sliceA + ccl.nTracks, CTF::BLCsliceA, 0);
  encodeTPC(ccl.timeA, ccl.timeA + ccl.nTracks, CTF::BLCtimeA, 0);
  encodeTPC(ccl.padA, ccl.padA + ccl.nTracks, CTF::BLCpadA, 0);

  if (mCombineColumns) {
    const auto [begin, end] = makeInputIterators(ccl.qTotU, ccl.qMaxU, ccl.nUnattachedClusters,
                                                 ShiftFunctor<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>, CTF::NBitsQMax>{});
    encodeTPC(begin, end, CTF::BLCqTotU, 0);
  } else {
    encodeTPC(ccl.qTotU, ccl.qTotU + ccl.nUnattachedClusters, CTF::BLCqTotU, 0);
  }
  encodeTPC(ccl.qMaxU, ccl.qMaxU + (mCombineColumns ? 0 : ccl.nUnattachedClusters), CTF::BLCqMaxU, 0);

  encodeTPC(ccl.flagsU, ccl.flagsU + ccl.nUnattachedClusters, CTF::BLCflagsU, 0);
  encodeTPC(ccl.padDiffU, ccl.padDiffU + ccl.nUnattachedClusters, CTF::BLCpadDiffU, 0);
  encodeTPC(ccl.timeDiffU, ccl.timeDiffU + ccl.nUnattachedClusters, CTF::BLCtimeDiffU, 0);

  if (mCombineColumns) {
    const auto [begin, end] = makeInputIterators(ccl.sigmaPadU, ccl.sigmaTimeU, ccl.nUnattachedClusters,
                                                 ShiftFunctor<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>, CTF::NBitsSigmaTime>{});
    encodeTPC(begin, end, CTF::BLCsigmaPadU, 0);
  } else {
    encodeTPC(ccl.sigmaPadU, ccl.sigmaPadU + ccl.nUnattachedClusters, CTF::BLCsigmaPadU, 0);
  }
  encodeTPC(ccl.sigmaTimeU, ccl.sigmaTimeU + (mCombineColumns ? 0 : ccl.nUnattachedClusters), CTF::BLCsigmaTimeU, 0);

  encodeTPC(ccl.nTrackClusters, ccl.nTrackClusters + ccl.nTracks, CTF::BLCnTrackClusters, 0);
  encodeTPC(ccl.nSliceRowClusters, ccl.nSliceRowClusters + ccl.nSliceRows, CTF::BLCnSliceRowClusters, 0);
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded bloks to TPC CompressedClusters into the externally provided vector (e.g. PMR vector from DPL)
template <typename VEC>
void CTFCoder::decode(const CTF::base& ec, VEC& buffVec)
{
  using namespace detail;
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
  auto decodeTPC = [&ec, &coders = mCoders](auto begin, CTF::Slots slot) {
    const auto slotVal = static_cast<int>(slot);
    ec.decode(begin, slotVal, coders[slotVal].get());
  };

  if (mCombineColumns) {
    detail::MergedColumnsDecoder<CTF::NBitsQTot, CTF::NBitsQMax>::decode(cc.qTotA, cc.qMaxA, CTF::BLCqTotA, decodeTPC);
  } else {
    decodeTPC(cc.qTotA, CTF::BLCqTotA);
    decodeTPC(cc.qMaxA, CTF::BLCqMaxA);
  }

  decodeTPC(cc.flagsA, CTF::BLCflagsA);

  if (mCombineColumns) {
    detail::MergedColumnsDecoder<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>::decode(cc.rowDiffA, cc.sliceLegDiffA, CTF::BLCrowDiffA, decodeTPC);
  } else {
    decodeTPC(cc.rowDiffA, CTF::BLCrowDiffA);
    decodeTPC(cc.sliceLegDiffA, CTF::BLCsliceLegDiffA);
  }

  decodeTPC(cc.padResA, CTF::BLCpadResA);
  decodeTPC(cc.timeResA, CTF::BLCtimeResA);

  if (mCombineColumns) {
    detail::MergedColumnsDecoder<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>::decode(cc.sigmaPadA, cc.sigmaTimeA, CTF::BLCsigmaPadA, decodeTPC);
  } else {
    decodeTPC(cc.sigmaPadA, CTF::BLCsigmaPadA);
    decodeTPC(cc.sigmaTimeA, CTF::BLCsigmaTimeA);
  }

  decodeTPC(cc.qPtA, CTF::BLCqPtA);
  decodeTPC(cc.rowA, CTF::BLCrowA);
  decodeTPC(cc.sliceA, CTF::BLCsliceA);
  decodeTPC(cc.timeA, CTF::BLCtimeA);
  decodeTPC(cc.padA, CTF::BLCpadA);

  if (mCombineColumns) {
    detail::MergedColumnsDecoder<CTF::NBitsQTot, CTF::NBitsQMax>::decode(cc.qTotU, cc.qMaxU, CTF::BLCqTotU, decodeTPC);
  } else {
    decodeTPC(cc.qTotU, CTF::BLCqTotU);
    decodeTPC(cc.qMaxU, CTF::BLCqMaxU);
  }

  decodeTPC(cc.flagsU, CTF::BLCflagsU);
  decodeTPC(cc.padDiffU, CTF::BLCpadDiffU);
  decodeTPC(cc.timeDiffU, CTF::BLCtimeDiffU);

  if (mCombineColumns) {
    detail::MergedColumnsDecoder<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>::decode(cc.sigmaPadU, cc.sigmaTimeU, CTF::BLCsigmaPadU, decodeTPC);
  } else {
    decodeTPC(cc.sigmaPadU, CTF::BLCsigmaPadU);
    decodeTPC(cc.sigmaTimeU, CTF::BLCsigmaTimeU);
  }

  decodeTPC(cc.nTrackClusters, CTF::BLCnTrackClusters);
  decodeTPC(cc.nSliceRowClusters, CTF::BLCnSliceRowClusters);
}

} // namespace tpc
} // namespace o2

#endif // O2_TPC_CTFCODER_H
