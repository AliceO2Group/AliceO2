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

/// \file EncodedBlock.h
/// \brief Set of entropy-encoded blocks

///  Used to store a CTF of particular detector. Can be build as a flat buffer which can be directly messaged between DPL devices

#ifndef ALICEO2_ENCODED_BLOCKS_H
#define ALICEO2_ENCODED_BLOCKS_H
// #undef NDEBUG
// #include <cassert>
#include <type_traits>
#include <cstddef>
#include <Rtypes.h>
#include <any>

#include "TTree.h"
#include "CommonUtils/StringUtils.h"
#include "Framework/Logger.h"
#include "DetectorsCommonDataFormats/CTFDictHeader.h"
#include "DetectorsCommonDataFormats/CTFIOSize.h"
#include "DetectorsCommonDataFormats/ANSHeader.h"
#include "DetectorsCommonDataFormats/internal/Packer.h"
#include "DetectorsCommonDataFormats/Metadata.h"
#ifndef __CLING__
#include "DetectorsCommonDataFormats/internal/ExternalEntropyCoder.h"
#include "DetectorsCommonDataFormats/internal/InplaceEntropyCoder.h"
#include "rANS/compat.h"
#include "rANS/histogram.h"
#include "rANS/serialize.h"
#include "rANS/factory.h"
#include "rANS/metrics.h"
#include "rANS/utils.h"
#endif

namespace o2
{
namespace ctf
{

namespace detail
{

template <class, class Enable = void>
struct is_iterator : std::false_type {
};

template <class T>
struct is_iterator<T, std::enable_if_t<
                        std::is_base_of_v<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category> ||
                        std::is_same_v<std::output_iterator_tag, typename std::iterator_traits<T>::iterator_category>>>
  : std::true_type {
};

template <class T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

inline constexpr bool mayEEncode(Metadata::OptStore opt) noexcept
{
  return (opt == Metadata::OptStore::EENCODE) || (opt == Metadata::OptStore::EENCODE_OR_PACK);
}

inline constexpr bool mayPack(Metadata::OptStore opt) noexcept
{
  return (opt == Metadata::OptStore::PACK) || (opt == Metadata::OptStore::EENCODE_OR_PACK);
}

} // namespace detail
constexpr size_t PackingThreshold = 512;

constexpr size_t Alignment = 16;

constexpr int WrappersSplitLevel = 99;
constexpr int WrappersCompressionLevel = 1;

/// This is the type of the vector to be used for the EncodedBlocks buffer allocation
using BufferType = uint8_t; // to avoid every detector using different types, we better define it here

/// align size to given diven number of bytes
inline size_t alignSize(size_t sizeBytes)
{
  auto res = sizeBytes % Alignment;
  return res ? sizeBytes + (Alignment - res) : sizeBytes;
}

/// relocate pointer by the difference of addresses
template <class T>
inline T* relocatePointer(const char* oldBase, char* newBase, const T* ptr)
{
  return (ptr != nullptr) ? reinterpret_cast<T*>(newBase + (reinterpret_cast<const char*>(ptr) - oldBase)) : nullptr;
}

template <typename source_T, typename dest_T, std::enable_if_t<(sizeof(dest_T) >= sizeof(source_T)), bool> = true>
inline constexpr size_t calculateNDestTElements(size_t nElems) noexcept
{
  const size_t srcBufferSize = nElems * sizeof(source_T);
  return srcBufferSize / sizeof(dest_T) + (srcBufferSize % sizeof(dest_T) != 0);
};

template <typename source_T, typename dest_T, std::enable_if_t<(sizeof(dest_T) >= sizeof(source_T)), bool> = true>
inline size_t calculatePaddedSize(size_t nElems) noexcept
{
  const size_t sizeOfSourceT = sizeof(source_T);
  const size_t sizeOfDestT = sizeof(dest_T);

  // this is equivalent to (sizeOfSourceT / sizeOfDestT) * std::ceil(sizeOfSourceArray/ sizeOfDestT)
  return (sizeOfDestT / sizeOfSourceT) * calculateNDestTElements<source_T, dest_T>(nElems);
};

///>>======================== Auxiliary classes =======================>>

/// registry struct for the buffer start and offsets of writable space
struct Registry {
  char* head = nullptr;     //! pointer on the head of the CTF
  int nFilledBlocks = 0;    // number of filled blocks = next block to fill (must be strictly consecutive)
  size_t offsFreeStart = 0; //! offset of the start of the writable space (wrt head), in bytes!!!
  size_t size = 0;          // full size in bytes!!!

  /// calculate the pointer of the head of the writable space
  char* getFreeBlockStart() const
  {
    assert(offsFreeStart <= size);
    return head + offsFreeStart;
  }

  /// size in bytes available to fill data
  size_t getFreeSize() const
  {
    return size - offsFreeStart;
  }

  char* getFreeBlockEnd() const
  {
    assert(offsFreeStart <= size);
    return getFreeBlockStart() + getFreeSize();
  }

  ClassDefNV(Registry, 1);
};

/// binary blob for single entropy-compressed column: metadata + (optional) dictionary and data buffer + their sizes
template <typename W = uint32_t>
struct Block {

  Registry* registry = nullptr; //! non-persistent info for in-memory ops
  int nDict = 0;                // dictionary length (if any)
  int nData = 0;                // length of data
  int nLiterals = 0;            // length of literals vector (if any)
  int nStored = 0;              // total length
  W* payload = nullptr;         //[nStored];

  inline const W* getDict() const { return nDict ? payload : nullptr; }
  inline const W* getData() const { return nData ? (payload + nDict) : nullptr; }
  inline const W* getDataPointer() const { return payload ? (payload + nDict) : nullptr; } // needed when nData is not set yet
  inline const W* getLiterals() const { return nLiterals ? (payload + nDict + nData) : nullptr; }
  inline const W* getEndOfBlock() const
  {
    if (!registry) {
      return nullptr;
    }
    // get last legal W*, since unaligned data is undefined behavior!
    const size_t delta = reinterpret_cast<uintptr_t>(registry->getFreeBlockEnd()) % sizeof(W);
    return reinterpret_cast<const W*>(registry->getFreeBlockEnd() - delta);
  }

  inline W* getCreatePayload() { return payload ? payload : (registry ? (payload = reinterpret_cast<W*>(registry->getFreeBlockStart())) : nullptr); }
  inline W* getCreateDict() { return payload ? payload : getCreatePayload(); }
  inline W* getCreateData() { return payload ? (payload + nDict) : getCreatePayload(); }
  inline W* getCreateLiterals() { return payload ? payload + (nDict + nData) : getCreatePayload(); }
  inline W* getEndOfBlock() { return const_cast<W*>(static_cast<const Block&>(*this).getEndOfBlock()); };

  inline auto getOffsDict() { return reinterpret_cast<std::uintptr_t>(getCreateDict()) - reinterpret_cast<std::uintptr_t>(registry->head); }
  inline auto getOffsData() { return reinterpret_cast<std::uintptr_t>(getCreateData()) - reinterpret_cast<std::uintptr_t>(registry->head); }
  inline auto getOffsLiterals() { return reinterpret_cast<std::uintptr_t>(getCreateLiterals()) - reinterpret_cast<std::uintptr_t>(registry->head); }

  inline void setNDict(int _ndict)
  {
    nDict = _ndict;
    nStored += nDict;
  }

  inline void setNData(int _ndata)
  {
    nData = _ndata;
    nStored += nData;
  }

  inline void setNLiterals(int _nliterals)
  {
    nLiterals = _nliterals;
    nStored += nLiterals;
  }

  inline int getNDict() const { return nDict; }
  inline int getNData() const { return nData; }
  inline int getNLiterals() const { return nLiterals; }
  inline int getNStored() const { return nStored; }

  ~Block()
  {
    if (!registry) { // this is a standalone block owning its data
      delete[] payload;
    }
  }

  /// clear itself
  void clear()
  {
    nDict = 0;
    nData = 0;
    nLiterals = 0;
    nStored = 0;
    payload = nullptr;
  }

  /// estimate free size needed to add new block
  static size_t estimateSize(int n)
  {
    return alignSize(n * sizeof(W));
  }

  // store a dictionary in an empty block
  void storeDict(int _ndict, const W* _dict)
  {
    if (getNStored() > 0) {
      throw std::runtime_error("trying to write in occupied block");
    }
    size_t sz = estimateSize(_ndict);
    assert(registry); // this method is valid only for flat version, which has a registry
    assert(sz <= registry->getFreeSize());
    assert((_ndict > 0) == (_dict != nullptr));
    setNDict(_ndict);
    if (nDict) {
      memcpy(getCreateDict(), _dict, _ndict * sizeof(W));
      realignBlock();
    }
  };

  // store a dictionary to a block which can either be empty or contain a dict.
  void storeData(int _ndata, const W* _data)
  {
    if (getNStored() > getNDict()) {
      throw std::runtime_error("trying to write in occupied block");
    }

    size_t sz = estimateSize(_ndata);
    assert(registry); // this method is valid only for flat version, which has a registry
    assert(sz <= registry->getFreeSize());
    assert((_ndata > 0) == (_data != nullptr));
    setNData(_ndata);
    if (nData) {
      memcpy(getCreateData(), _data, _ndata * sizeof(W));
      realignBlock();
    }
  }

  // store a dictionary to a block which can either be empty or contain a dict.
  void storeLiterals(int _nliterals, const W* _literals)
  {
    if (getNStored() > getNDict() + getNData()) {
      throw std::runtime_error("trying to write in occupied block");
    }

    size_t sz = estimateSize(_nliterals);
    assert(registry); // this method is valid only for flat version, which has a registry
    assert(sz <= registry->getFreeSize());
    //    assert((_nliterals > 0) == (_literals != nullptr));
    setNLiterals(_nliterals);
    if (nLiterals) {
      memcpy(getCreateLiterals(), _literals, _nliterals * sizeof(W));
      realignBlock();
    }
  }

  // resize block and free up unused buffer space.
  void realignBlock()
  {
    size_t sz = estimateSize(getNStored());
    registry->offsFreeStart = (reinterpret_cast<char*>(payload) - registry->head) + sz;
  }

  /// store binary blob data (buffer filled from head to tail)
  void store(int _ndict, int _ndata, int _nliterals, const W* _dict, const W* _data, const W* _literals)
  {
    size_t sz = estimateSize(_ndict + _ndata + _nliterals);
    assert(registry); // this method is valid only for flat version, which has a registry
    assert(sz <= registry->getFreeSize());
    assert((_ndict > 0) == (_dict != nullptr));
    assert((_ndata > 0) == (_data != nullptr));
    //    assert(_literals == _data + _nliterals);
    setNDict(_ndict);
    setNData(_ndata);
    setNLiterals(_nliterals);
    getCreatePayload(); // do this even for empty block!!!
    if (getNStored()) {
      payload = reinterpret_cast<W*>(registry->getFreeBlockStart());
      if (getNDict()) {
        memcpy(getCreateDict(), _dict, _ndict * sizeof(W));
      }
      if (getNData()) {
        memcpy(getCreateData(), _data, _ndata * sizeof(W));
      }
      if (getNLiterals()) {
        memcpy(getCreateLiterals(), _literals, _nliterals * sizeof(W));
      }
    }
    realignBlock();
  }

  /// relocate to different head position
  void relocate(const char* oldHead, char* newHeadData, char* newHeadRegistry)
  {
    payload = relocatePointer(oldHead, newHeadData, payload);
    registry = relocatePointer(oldHead, newHeadRegistry, registry);
  }

  ClassDefNV(Block, 1);
}; // namespace ctf

///<<======================== Auxiliary classes =======================<<

template <typename H, int N, typename W = uint32_t>
class EncodedBlocks
{
 public:
  typedef EncodedBlocks<H, N, W> base;

#ifndef __CLING__
  template <typename source_T>
  using dictionaryType = std::variant<rans::RenormedSparseHistogram<source_T>, rans::RenormedDenseHistogram<source_T>>;
#endif

  void setHeader(const H& h)
  {
    mHeader = h;
  }
  const H& getHeader() const { return mHeader; }
  H& getHeader() { return mHeader; }
  std::shared_ptr<H> cloneHeader() const { return std::shared_ptr<H>(new H(mHeader)); } // for dictionary creation

  const auto& getRegistry() const { return mRegistry; }

  const auto& getMetadata() const { return mMetadata; }

  auto& getMetadata(int i) const
  {
    assert(i < N);
    return mMetadata[i];
  }

  auto& getBlock(int i) const
  {
    assert(i < N);
    return mBlocks[i];
  }

#ifndef __CLING__
  template <typename source_T>
  dictionaryType<source_T> getDictionary(int i, ANSHeader ansVersion = ANSVersionUnspecified) const
  {
    const auto& block = getBlock(i);
    const auto& metadata = getMetadata(i);
    ansVersion = checkANSVersion(ansVersion);

    assert(static_cast<int64_t>(std::numeric_limits<source_T>::min()) <= static_cast<int64_t>(metadata.max));
    assert(static_cast<int64_t>(std::numeric_limits<source_T>::max()) >= static_cast<int64_t>(metadata.min));

    // check consistency of metadata and type
    [&]() {
      const int64_t sourceMin = std::numeric_limits<source_T>::min();
      const int64_t sourceMax = std::numeric_limits<source_T>::max();

      auto view = rans::trim(rans::HistogramView{block.getDict(), block.getDict() + block.getNDict(), metadata.min});
      const int64_t dictMin = view.getMin();
      const int64_t dictMax = view.getMax();
      assert(dictMin >= metadata.min);
      assert(dictMax <= metadata.max);

      if ((dictMin < sourceMin) || (dictMax > sourceMax)) {
        if (ansVersion == ANSVersionCompat && mHeader.majorVersion == 1 && mHeader.minorVersion == 0 && mHeader.dictTimeStamp < 1653192000000) {
          LOGP(warn, "value range of dictionary and target datatype are incompatible: target type [{},{}] vs dictionary [{},{}], tolerate in compat mode for old dictionaries", sourceMin, sourceMax, dictMin, dictMax);
        } else {
          throw std::runtime_error(fmt::format("value range of dictionary and target datatype are incompatible: target type [{},{}] vs dictionary [{},{}]", sourceMin, sourceMax, dictMin, dictMax));
        }
      }
    }();

    if (ansVersion == ANSVersionCompat) {
      rans::DenseHistogram<source_T> histogram{block.getDict(), block.getDict() + block.getNDict(), metadata.min};
      return rans::compat::renorm(std::move(histogram), metadata.probabilityBits);
    } else if (ansVersion == ANSVersion1) {
      // dictionary is loaded from an explicit dict file and is stored densly
      if (getANSHeader() == ANSVersionUnspecified) {
        rans::DenseHistogram<source_T> histogram{block.getDict(), block.getDict() + block.getNDict(), metadata.min};
        size_t renormingBits = rans::utils::sanitizeRenormingBitRange(metadata.probabilityBits);
        LOG_IF(debug, renormingBits != metadata.probabilityBits)
          << fmt::format("While reading metadata from external dictionary, rANSV1 is rounding renorming precision from {} to {}", metadata.probabilityBits, renormingBits);
        return rans::renorm(std::move(histogram), renormingBits, rans::RenormingPolicy::ForceIncompressible);
      } else {
        // dictionary is elias-delta coded inside the block
        if constexpr (sizeof(source_T) > 2) {
          return rans::readRenormedSetDictionary(block.getDict(), block.getDict() + block.getNDict(),
                                                 static_cast<source_T>(metadata.min), static_cast<source_T>(metadata.max),
                                                 metadata.probabilityBits);
        } else {
          return rans::readRenormedDictionary(block.getDict(), block.getDict() + block.getNDict(),
                                              static_cast<source_T>(metadata.min), static_cast<source_T>(metadata.max),
                                              metadata.probabilityBits);
        }
      }
    } else {
      throw std::runtime_error(fmt::format("Failed to load serialized Dictionary. Unsupported ANS Version: {}", static_cast<std::string>(ansVersion)));
    }
  };
#endif

  void setANSHeader(const ANSHeader& h)
  {
    mANSHeader = h;
  }
  const ANSHeader& getANSHeader() const { return mANSHeader; }
  ANSHeader& getANSHeader() { return mANSHeader; }

  static constexpr int getNBlocks() { return N; }

  static size_t getMinAlignedSize() { return alignSize(sizeof(base)); }

  /// cast arbitrary buffer head to container class. Head is supposed to respect the alignment
  static auto get(void* head) { return reinterpret_cast<EncodedBlocks*>(head); }
  static auto get(const void* head) { return reinterpret_cast<const EncodedBlocks*>(head); }

  /// get const image of the container wrapper, with pointers in the image relocated to new head
  static auto getImage(const void* newHead);

  /// create container from arbitrary buffer of predefined size (in bytes!!!). Head is supposed to respect the alignment
  static auto create(void* head, size_t sz);

  /// create container from vector. Head is supposed to respect the alignment
  template <typename VD>
  static auto create(VD& v);

  /// estimate free size needed to add new block
  static size_t estimateBlockSize(int n) { return Block<W>::estimateSize(n); }

  /// check if empty and valid
  bool empty() const { return (mRegistry.offsFreeStart == alignSize(sizeof(*this))) && (mRegistry.size >= mRegistry.offsFreeStart); }

  /// check if flat and valid
  bool flat() const { return mRegistry.size > 0 && (mRegistry.size >= mRegistry.offsFreeStart) && (mBlocks[0].registry == &mRegistry) && (mBlocks[N - 1].registry == &mRegistry); }

  /// clear itself
  void clear();

  /// Compactify by eliminating empty space
  size_t compactify() { return (mRegistry.size = estimateSize()); }

  /// total allocated size in bytes
  size_t size() const { return mRegistry.size; }

  /// size remaining for additional data
  size_t getFreeSize() const { return mRegistry.getFreeSize(); }

  /// expand the storage to new size in bytes
  template <typename buffer_T>
  static auto expand(buffer_T& buffer, size_t newsizeBytes);

  /// copy itself to flat buffer created on the fly from the vector
  template <typename V>
  void copyToFlat(V& vec);

  /// copy itself to flat buffer created on the fly at the provided pointer. The destination block should be at least of size estimateSize()
  void copyToFlat(void* base) { fillFlatCopy(create(base, estimateSize())); }

  /// attach to tree
  size_t appendToTree(TTree& tree, const std::string& name) const;

  /// read from tree to non-flat object
  void readFromTree(TTree& tree, const std::string& name, int ev = 0);

  /// read from tree to destination buffer vector
  template <typename VD>
  static void readFromTree(VD& vec, TTree& tree, const std::string& name, int ev = 0);

  /// encode vector src to bloc at provided slot
  template <typename VE, typename buffer_T>
  inline o2::ctf::CTFIOSize encode(const VE& src, int slot, uint8_t symbolTablePrecision, Metadata::OptStore opt, buffer_T* buffer = nullptr, const std::any& encoderExt = {}, float memfc = 1.f)
  {
    return encode(std::begin(src), std::end(src), slot, symbolTablePrecision, opt, buffer, encoderExt, memfc);
  }

  /// encode vector src to bloc at provided slot
  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize encode(const input_IT srcBegin, const input_IT srcEnd, int slot, uint8_t symbolTablePrecision, Metadata::OptStore opt, buffer_T* buffer = nullptr, const std::any& encoderExt = {}, float memfc = 1.f);

  /// decode block at provided slot to destination vector (will be resized as needed)
  template <class container_T, class container_IT = typename container_T::iterator>
  o2::ctf::CTFIOSize decode(container_T& dest, int slot, const std::any& decoderExt = {}) const;

  /// decode block at provided slot to destination pointer, the needed space assumed to be available
  template <typename D_IT, std::enable_if_t<detail::is_iterator_v<D_IT>, bool> = true>
  o2::ctf::CTFIOSize decode(D_IT dest, int slot, const std::any& decoderExt = {}) const;

#ifndef __CLING__
  /// create a special EncodedBlocks containing only dictionaries made from provided vector of frequency tables
  static std::vector<char> createDictionaryBlocks(const std::vector<rans::DenseHistogram<int32_t>>& vfreq, const std::vector<Metadata>& prbits);
#endif

  /// print itself
  void print(const std::string& prefix = "", int verbosity = 1) const;
  void dump(const std::string& prefix = "", int ncol = 20) const;

 protected:
  static_assert(N > 0, "number of encoded blocks < 1");

  Registry mRegistry;                //
  ANSHeader mANSHeader;              //  ANS header
  H mHeader;                         //  detector specific header
  std::array<Metadata, N> mMetadata; //  compressed block's details
  std::array<Block<W>, N> mBlocks;   //! this is in fact stored, but to overcome TBuffer limits we have to define the branches per block!!!

  inline static constexpr Metadata::OptStore FallbackStorageType{Metadata::OptStore::NONE};

  /// setup internal structure and registry for given buffer size (in bytes!!!)
  void init(size_t sz);

  /// relocate to different head position, newHead points on start of the dynamic buffer holding the data.
  /// the address of the static part might be actually different (wrapper). This different newHead and
  /// wrapper addresses must be used when the buffer pointed by newHead is const (e.g. received from the
  /// DPL input), in this case we create a wrapper, which points on these const data
  static void relocate(const char* oldHead, char* newHead, char* wrapper, size_t newsize = 0);

  /// Estimate size of the buffer needed to store all compressed data in a contiguous block of memory, accounting for the alignment
  /// This method is to be called after reading object from the tree as a non-flat object!
  size_t estimateSize() const;

  /// do the same using metadata info
  size_t estimateSizeFromMetadata() const;

  /// Create its own flat copy in the destination empty flat object
  void fillFlatCopy(EncodedBlocks& dest) const;

  /// add and fill single branch
  template <typename D>
  static size_t fillTreeBranch(TTree& tree, const std::string& brname, D& dt, int compLevel, int splitLevel = 99);

  /// read single branch
  template <typename D>
  static bool readTreeBranch(TTree& tree, const std::string& brname, D& dt, int ev = 0);

  template <typename T>
  auto expandStorage(size_t slot, size_t nElemets, T* buffer = nullptr) -> decltype(auto);

  inline ANSHeader checkANSVersion(ANSHeader ansVersion) const
  {
    auto ctfANSHeader = getANSHeader();
    ANSHeader ret{ANSVersionUnspecified};

    const bool isEqual{ansVersion == ctfANSHeader};
    const bool isHeaderUnspecified{ctfANSHeader == ANSVersionUnspecified};

    if (isEqual) {
      if (isHeaderUnspecified) {
        throw std::runtime_error{fmt::format("Missmatch of ANSVersions, trying to encode/decode CTF with ANS Version Header {} with ANS Version {}",
                                             static_cast<std::string>(ctfANSHeader),
                                             static_cast<std::string>(ansVersion))};
      } else {
        ret = ctfANSHeader;
      }
    } else {
      if (isHeaderUnspecified) {
        ret = ansVersion;
      } else {
        ret = ctfANSHeader;
      }
    }

    return ret;
  };

  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize entropyCodeRANSCompat(const input_IT srcBegin, const input_IT srcEnd, int slot, uint8_t symbolTablePrecision, buffer_T* buffer = nullptr, const std::any& encoderExt = {}, float memfc = 1.f);

  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize entropyCodeRANSV1(const input_IT srcBegin, const input_IT srcEnd, int slot, Metadata::OptStore opt, buffer_T* buffer = nullptr, const std::any& encoderExt = {}, float memfc = 1.f);

  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize encodeRANSV1External(const input_IT srcBegin, const input_IT srcEnd, int slot, const std::any& encoderExt, buffer_T* buffer = nullptr, double_t sizeEstimateSafetyFactor = 1);

  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize encodeRANSV1Inplace(const input_IT srcBegin, const input_IT srcEnd, int slot, Metadata::OptStore opt, buffer_T* buffer = nullptr, double_t sizeEstimateSafetyFactor = 1);

#ifndef __CLING__
  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize pack(const input_IT srcBegin, const input_IT srcEnd, int slot, rans::Metrics<typename std::iterator_traits<input_IT>::value_type> metrics, buffer_T* buffer = nullptr);

  template <typename input_IT, typename buffer_T>
  inline o2::ctf::CTFIOSize pack(const input_IT srcBegin, const input_IT srcEnd, int slot, buffer_T* buffer = nullptr)
  {
    using source_type = typename std::iterator_traits<input_IT>::value_type;

    rans::Metrics<source_type> metrics{};

    const auto [minIter, maxIter] = std::minmax_element(srcBegin, srcEnd);
    if (minIter != maxIter) {
      metrics.getDatasetProperties().min = *minIter;
      metrics.getDatasetProperties().max = *maxIter;
      metrics.getDatasetProperties().alphabetRangeBits = rans::utils::getRangeBits(metrics.getDatasetProperties().min,
                                                                                   metrics.getDatasetProperties().max);
    }

    return pack(srcBegin, srcEnd, slot, metrics, buffer);
  }
#endif

  template <typename input_IT, typename buffer_T>
  o2::ctf::CTFIOSize store(const input_IT srcBegin, const input_IT srcEnd, int slot, Metadata::OptStore opt, buffer_T* buffer = nullptr);

  // decode
  template <typename dst_IT>
  CTFIOSize decodeCompatImpl(dst_IT dest, int slot, const std::any& decoderExt) const;

  template <typename dst_IT>
  CTFIOSize decodeRansV1Impl(dst_IT dest, int slot, const std::any& decoderExt) const;

  template <typename dst_IT>
  CTFIOSize decodeUnpackImpl(dst_IT dest, int slot) const;

  template <typename dst_IT>
  CTFIOSize decodeCopyImpl(dst_IT dest, int slot) const;

  ClassDefNV(EncodedBlocks, 3);
};

///_____________________________________________________________________________
/// read from tree to non-flat object
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::readFromTree(TTree& tree, const std::string& name, int ev)
{
  readTreeBranch(tree, o2::utils::Str::concat_string(name, "_wrapper."), *this, ev);
  for (int i = 0; i < N; i++) {
    readTreeBranch(tree, o2::utils::Str::concat_string(name, "_block.", std::to_string(i), "."), mBlocks[i], ev);
  }
}

///_____________________________________________________________________________
/// read from tree to destination buffer vector
template <typename H, int N, typename W>
template <typename VD>
void EncodedBlocks<H, N, W>::readFromTree(VD& vec, TTree& tree, const std::string& name, int ev)
{
  auto tmp = create(vec);
  if (!readTreeBranch(tree, o2::utils::Str::concat_string(name, "_wrapper."), *tmp, ev)) {
    throw std::runtime_error(fmt::format("Failed to read CTF header for {}", name));
  }
  tmp = tmp->expand(vec, tmp->estimateSizeFromMetadata());
  const auto& meta = tmp->getMetadata();
  for (int i = 0; i < N; i++) {
    Block<W> bl;
    readTreeBranch(tree, o2::utils::Str::concat_string(name, "_block.", std::to_string(i), "."), bl, ev);
    assert(meta[i].nDictWords == bl.getNDict());
    assert(meta[i].nDataWords == bl.getNData());
    assert(meta[i].nLiteralWords == bl.getNLiterals());
    tmp->mBlocks[i].store(bl.getNDict(), bl.getNData(), bl.getNLiterals(), bl.getDict(), bl.getData(), bl.getLiterals());
  }
}

///_____________________________________________________________________________
/// attach to tree
template <typename H, int N, typename W>
size_t EncodedBlocks<H, N, W>::appendToTree(TTree& tree, const std::string& name) const
{
  long s = 0;
  s += fillTreeBranch(tree, o2::utils::Str::concat_string(name, "_wrapper."), const_cast<base&>(*this), WrappersCompressionLevel, WrappersSplitLevel);
  for (int i = 0; i < N; i++) {
    int compression = mMetadata[i].opt == Metadata::OptStore::ROOTCompression ? 1 : 0;
    s += fillTreeBranch(tree, o2::utils::Str::concat_string(name, "_block.", std::to_string(i), "."), const_cast<Block<W>&>(mBlocks[i]), compression);
  }
  tree.SetEntries(tree.GetEntries() + 1);
  return s;
}

///_____________________________________________________________________________
/// read single branch
template <typename H, int N, typename W>
template <typename D>
bool EncodedBlocks<H, N, W>::readTreeBranch(TTree& tree, const std::string& brname, D& dt, int ev)
{
  auto* br = tree.GetBranch(brname.c_str());
  if (!br) {
    LOG(debug) << "Branch " << brname << " is absent";
    return false;
  }
  auto* ptr = &dt;
  br->SetAddress(&ptr);
  br->GetEntry(ev);
  br->ResetAddress();
  return true;
}

///_____________________________________________________________________________
/// add and fill single branch
template <typename H, int N, typename W>
template <typename D>
inline size_t EncodedBlocks<H, N, W>::fillTreeBranch(TTree& tree, const std::string& brname, D& dt, int compLevel, int splitLevel)
{
  auto* br = tree.GetBranch(brname.c_str());
  if (!br) {
    br = tree.Branch(brname.c_str(), &dt, 512, splitLevel);
    br->SetCompressionLevel(compLevel);
  }
  return br->Fill();
}

///_____________________________________________________________________________
/// Create its own flat copy in the destination empty flat object
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::fillFlatCopy(EncodedBlocks& dest) const
{
  assert(dest.empty() && dest.mRegistry.getFreeSize() < estimateSize());
  dest.mANSHeader = mANSHeader;
  dest.mHeader = mHeader;
  dest.mMetadata = mMetadata;
  for (int i = 0; i < N; i++) {
    dest.mBlocks[i].store(mBlocks[i].getNDict(), mBlocks[i].getNData(), mBlocks[i].getDict(), mBlocks[i].getData());
  }
}

///_____________________________________________________________________________
/// Copy itself to flat buffer created on the fly from the vector
template <typename H, int N, typename W>
template <typename V>
void EncodedBlocks<H, N, W>::copyToFlat(V& vec)
{
  auto vtsz = sizeof(typename std::remove_reference<decltype(vec)>::type::value_type), sz = estimateSize();
  vec.resize(sz / vtsz);
  copyToFlat(vec.data());
}

///_____________________________________________________________________________
/// Estimate size of the buffer needed to store all compressed data in a contiguos block of memory, accounting for alignment
/// This method is to be called after reading object from the tree as a non-flat object!
template <typename H, int N, typename W>
size_t EncodedBlocks<H, N, W>::estimateSize() const
{
  size_t sz = 0;
  sz += alignSize(sizeof(*this));
  for (int i = 0; i < N; i++) {
    sz += alignSize(mBlocks[i].nStored * sizeof(W));
  }
  return sz;
}

///_____________________________________________________________________________
/// Estimate size from metadata
/// This method is to be called after reading object from the tree as a non-flat object!
template <typename H, int N, typename W>
size_t EncodedBlocks<H, N, W>::estimateSizeFromMetadata() const
{
  size_t sz = alignSize(sizeof(*this));
  for (int i = 0; i < N; i++) {
    sz += alignSize((mMetadata[i].nDictWords + mMetadata[i].nDataWords + mMetadata[i].nLiteralWords) * sizeof(W));
  }
  return sz;
}

///_____________________________________________________________________________
/// expand the storage to new size in bytes
template <typename H, int N, typename W>
template <typename buffer_T>
auto EncodedBlocks<H, N, W>::expand(buffer_T& buffer, size_t newsizeBytes)
{
  auto buftypesize = sizeof(typename std::remove_reference<decltype(buffer)>::type::value_type);
  auto* oldHead = get(buffer.data())->mRegistry.head;
  buffer.resize(alignSize(newsizeBytes) / buftypesize);
  relocate(oldHead, reinterpret_cast<char*>(buffer.data()), reinterpret_cast<char*>(buffer.data()), newsizeBytes);
  return get(buffer.data());
}

///_____________________________________________________________________________
/// relocate to different head position, newHead points on start of the dynamic buffer holding the data.
/// the address of the static part might be actually different (wrapper). This different newHead and
/// wrapper addresses must be used when the buffer pointed by newHead is const (e.g. received from the
/// DPL input), in this case we create a wrapper, which points on these const data
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::relocate(const char* oldHead, char* newHead, char* wrapper, size_t newsize)
{
  auto newStr = get(wrapper);
  for (int i = 0; i < N; i++) {
    newStr->mBlocks[i].relocate(oldHead, newHead, wrapper);
  }
  newStr->mRegistry.head = newHead; // newHead points on the real data
  // if asked, update the size
  if (newsize) { // in bytes!!!
    assert(newStr->estimateSize() <= newsize);
    newStr->mRegistry.size = newsize;
  }
}

///_____________________________________________________________________________
/// setup internal structure and registry for given buffer size (in bytes!!!)
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::init(size_t sz)
{
  mRegistry.head = reinterpret_cast<char*>(this);
  mRegistry.size = sz;
  mRegistry.offsFreeStart = alignSize(sizeof(*this));
  for (int i = 0; i < N; i++) {
    mMetadata[i].clear();
    mBlocks[i].registry = &mRegistry;
    mBlocks[i].clear();
  }
}

///_____________________________________________________________________________
/// clear itself
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::clear()
{
  for (int i = 0; i < N; i++) {
    mBlocks[i].clear();
    mMetadata[i].clear();
  }
  mRegistry.offsFreeStart = alignSize(sizeof(*this));
}

///_____________________________________________________________________________
/// get const image of the container wrapper, with pointers in the image relocated to new head
template <typename H, int N, typename W>
auto EncodedBlocks<H, N, W>::getImage(const void* newHead)
{
  assert(newHead);
  auto image(*get(newHead)); // 1st make a shalow copy
  // now fix its pointers
  // we don't modify newHead, but still need to remove constness for relocation interface
  relocate(image.mRegistry.head, const_cast<char*>(reinterpret_cast<const char*>(newHead)), reinterpret_cast<char*>(&image));

  return image;
}

///_____________________________________________________________________________
/// create container from arbitrary buffer of predefined size (in bytes!!!). Head is supposed to respect the alignment
template <typename H, int N, typename W>
inline auto EncodedBlocks<H, N, W>::create(void* head, size_t sz)
{
  const H defh;
  auto b = get(head);
  b->init(sz);
  b->setHeader(defh);
  return b;
}

///_____________________________________________________________________________
/// create container from arbitrary buffer of predefined size (in bytes!!!). Head is supposed to respect the alignment
template <typename H, int N, typename W>
template <typename VD>
inline auto EncodedBlocks<H, N, W>::create(VD& v)
{
  size_t vsz = sizeof(typename std::remove_reference<decltype(v)>::type::value_type); // size of the element of the buffer
  auto baseSize = getMinAlignedSize() / vsz;
  if (v.size() < baseSize) {
    v.resize(baseSize);
  }
  return create(v.data(), v.size() * vsz);
}

///_____________________________________________________________________________
/// print itself
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::print(const std::string& prefix, int verbosity) const
{
  verbosity = 5;
  if (verbosity > 0) {
    LOG(info) << prefix << "Container of " << N << " blocks, size: " << size() << " bytes, unused: " << getFreeSize();
    for (int i = 0; i < N; i++) {
      LOG(info) << "Block " << i << " for " << static_cast<uint32_t>(mMetadata[i].messageLength) << " message words of "
                << static_cast<uint32_t>(mMetadata[i].messageWordSize) << " bytes |"
                << " NDictWords: " << mBlocks[i].getNDict() << " NDataWords: " << mBlocks[i].getNData()
                << " NLiteralWords: " << mBlocks[i].getNLiterals();
    }
  } else if (verbosity == 0) {
    size_t inpSize = 0, ndict = 0, ndata = 0, nlit = 0;
    for (int i = 0; i < N; i++) {
      inpSize += mMetadata[i].messageLength * mMetadata[i].messageWordSize;
      ndict += mBlocks[i].getNDict();
      ndata += mBlocks[i].getNData();
      nlit += mBlocks[i].getNLiterals();
    }
    LOG(info) << prefix << N << " blocks, input size: " << inpSize << ", output size: " << size()
              << " NDictWords: " << ndict << " NDataWords: " << ndata << " NLiteralWords: " << nlit;
  }
}

///_____________________________________________________________________________
template <typename H, int N, typename W>
template <class container_T, class container_IT>
inline o2::ctf::CTFIOSize EncodedBlocks<H, N, W>::decode(container_T& dest,                // destination container
                                                         int slot,                         // slot of the block to decode
                                                         const std::any& decoderExt) const // optional externally provided decoder
{
  dest.resize(mMetadata[slot].messageLength); // allocate output buffer
  return decode(std::begin(dest), slot, decoderExt);
}

///_____________________________________________________________________________
template <typename H, int N, typename W>
template <typename D_IT, std::enable_if_t<detail::is_iterator_v<D_IT>, bool>>
CTFIOSize EncodedBlocks<H, N, W>::decode(D_IT dest,                        // iterator to destination
                                         int slot,                         // slot of the block to decode
                                         const std::any& decoderExt) const // optional externally provided decoder
{

  // get references to the right data
  const auto& ansVersion = getANSHeader();
  const auto& block = mBlocks[slot];
  const auto& md = mMetadata[slot];

  if (!block.getNStored()) {
    return {0, md.getUncompressedSize(), md.getCompressedSize()};
  }

  if (ansVersion == ANSVersionCompat) {
    if (md.opt == Metadata::OptStore::EENCODE) {
      return decodeCompatImpl(dest, slot, decoderExt);
    } else {
      return decodeCopyImpl(dest, slot);
    }
  } else if (ansVersion == ANSVersion1) {
    if (md.opt == Metadata::OptStore::EENCODE) {
      return decodeRansV1Impl(dest, slot, decoderExt);
    } else if (md.opt == Metadata::OptStore::PACK) {
      return decodeUnpackImpl(dest, slot);
    } else {
      return decodeCopyImpl(dest, slot);
    }
  } else {
    throw std::runtime_error("unsupported ANS Version");
  }
};

#ifndef __CLING__
template <typename H, int N, typename W>
template <typename dst_IT>
CTFIOSize EncodedBlocks<H, N, W>::decodeCompatImpl(dst_IT dstBegin, int slot, const std::any& decoderExt) const
{

  // get references to the right data
  const auto& block = mBlocks[slot];
  const auto& md = mMetadata[slot];

  using dst_type = typename std::iterator_traits<dst_IT>::value_type;
  using decoder_type = typename rans::compat::decoder_type<dst_type>;

  std::optional<decoder_type> inplaceDecoder{};
  if (md.nDictWords > 0) {
    inplaceDecoder = decoder_type{std::get<rans::RenormedDenseHistogram<dst_type>>(this->getDictionary<dst_type>(slot))};
  } else if (!decoderExt.has_value()) {
    throw std::runtime_error("neither dictionary nor external decoder provided");
  }

  auto getDecoder = [&]() -> const decoder_type& {
    if (inplaceDecoder.has_value()) {
      return inplaceDecoder.value();
    } else {
      return std::any_cast<const decoder_type&>(decoderExt);
    }
  };

  const size_t NDecoderStreams = rans::compat::defaults::CoderPreset::nStreams;

  if (block.getNLiterals()) {
    auto* literalsEnd = reinterpret_cast<const dst_type*>(block.getLiterals()) + md.nLiterals;
    getDecoder().process(block.getData() + block.getNData(), dstBegin, md.messageLength, NDecoderStreams, literalsEnd);
  } else {
    getDecoder().process(block.getData() + block.getNData(), dstBegin, md.messageLength, NDecoderStreams);
  }
  return {0, md.getUncompressedSize(), md.getCompressedSize()};
};

template <typename H, int N, typename W>
template <typename dst_IT>
CTFIOSize EncodedBlocks<H, N, W>::decodeRansV1Impl(dst_IT dstBegin, int slot, const std::any& decoderExt) const
{

  // get references to the right data
  const auto& block = mBlocks[slot];
  const auto& md = mMetadata[slot];

  using dst_type = typename std::iterator_traits<dst_IT>::value_type;
  using decoder_type = typename rans::defaultDecoder_type<dst_type>;

  std::optional<decoder_type> inplaceDecoder{};
  if (md.nDictWords > 0) {
    std::visit([&](auto&& arg) { inplaceDecoder = decoder_type{arg}; }, this->getDictionary<dst_type>(slot));
  } else if (!decoderExt.has_value()) {
    throw std::runtime_error("no dictionary nor external decoder provided");
  }

  auto getDecoder = [&]() -> const decoder_type& {
    if (inplaceDecoder.has_value()) {
      return inplaceDecoder.value();
    } else {
      return std::any_cast<const decoder_type&>(decoderExt);
    }
  };

  // verify decoders
  [&]() {
    const decoder_type& decoder = getDecoder();
    const size_t decoderSymbolTablePrecision = decoder.getSymbolTablePrecision();

    if (md.probabilityBits != decoderSymbolTablePrecision) {
      throw std::runtime_error(fmt::format(
        "Missmatch in decoder renorming precision vs metadata:{} Bits vs {} Bits.",
        md.probabilityBits, decoderSymbolTablePrecision));
    }

    if (md.streamSize != rans::utils::getStreamingLowerBound_v<typename decoder_type::coder_type>) {
      throw std::runtime_error("Streaming lower bound of dataset and decoder do not match");
    }
  }();

  // do the actual decoding
  if (block.getNLiterals()) {
    std::vector<dst_type> literals(md.nLiterals);
    rans::unpack(block.getLiterals(), md.nLiterals, literals.data(), md.literalsPackingWidth, md.literalsPackingOffset);
    getDecoder().process(block.getData() + block.getNData(), dstBegin, md.messageLength, md.nStreams, literals.end());
  } else {
    getDecoder().process(block.getData() + block.getNData(), dstBegin, md.messageLength, md.nStreams);
  }
  return {0, md.getUncompressedSize(), md.getCompressedSize()};
};

template <typename H, int N, typename W>
template <typename dst_IT>
CTFIOSize EncodedBlocks<H, N, W>::decodeUnpackImpl(dst_IT dest, int slot) const
{
  using dest_t = typename std::iterator_traits<dst_IT>::value_type;

  const auto& block = mBlocks[slot];
  const auto& md = mMetadata[slot];

  const size_t packingWidth = md.probabilityBits;
  const dest_t offset = md.min;
  rans::unpack(block.getData(), md.messageLength, dest, packingWidth, offset);
  return {0, md.getUncompressedSize(), md.getCompressedSize()};
};

template <typename H, int N, typename W>
template <typename dst_IT>
CTFIOSize EncodedBlocks<H, N, W>::decodeCopyImpl(dst_IT dest, int slot) const
{
  // get references to the right data
  const auto& block = mBlocks[slot];
  const auto& md = mMetadata[slot];

  using dest_t = typename std::iterator_traits<dst_IT>::value_type;
  using decoder_t = typename rans::compat::decoder_type<dest_t>;
  using destPtr_t = typename std::iterator_traits<dst_IT>::pointer;

  destPtr_t srcBegin = reinterpret_cast<destPtr_t>(block.payload);
  destPtr_t srcEnd = srcBegin + md.messageLength * sizeof(dest_t);
  std::copy(srcBegin, srcEnd, dest);

  return {0, md.getUncompressedSize(), md.getCompressedSize()};
};

///_____________________________________________________________________________
template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
o2::ctf::CTFIOSize EncodedBlocks<H, N, W>::encode(const input_IT srcBegin,      // iterator begin of source message
                                                  const input_IT srcEnd,        // iterator end of source message
                                                  int slot,                     // slot in encoded data to fill
                                                  uint8_t symbolTablePrecision, // encoding into
                                                  Metadata::OptStore opt,       // option for data compression
                                                  buffer_T* buffer,             // optional buffer (vector) providing memory for encoded blocks
                                                  const std::any& encoderExt,   // optional external encoder
                                                  float memfc)                  // memory allocation margin factor
{
  // fill a new block
  assert(slot == mRegistry.nFilledBlocks);
  mRegistry.nFilledBlocks++;

  const size_t messageLength = std::distance(srcBegin, srcEnd);
  // cover three cases:
  // * empty source message: no co
  // * source message to pass through without any entropy coding
  // * source message where entropy coding should be applied

  // case 1: empty source message
  if (messageLength == 0) {
    mMetadata[slot] = Metadata{};
    mMetadata[slot].opt = Metadata::OptStore::NODATA;
    return {};
  }
  if (detail::mayEEncode(opt)) {
    const ANSHeader& ansVersion = getANSHeader();
    if (ansVersion == ANSVersionCompat) {
      return entropyCodeRANSCompat(srcBegin, srcEnd, slot, symbolTablePrecision, buffer, encoderExt, memfc);
    } else if (ansVersion == ANSVersion1) {
      return entropyCodeRANSV1(srcBegin, srcEnd, slot, opt, buffer, encoderExt, memfc);
    } else {
      throw std::runtime_error(fmt::format("Unsupported ANS Coder Version: {}.{}", ansVersion.majorVersion, ansVersion.minorVersion));
    }
  } else if (detail::mayPack(opt)) {
    return pack(srcBegin, srcEnd, slot, buffer);
  } else {
    return store(srcBegin, srcEnd, slot, opt, buffer);
  }
};

template <typename H, int N, typename W>
template <typename T>
[[nodiscard]] auto EncodedBlocks<H, N, W>::expandStorage(size_t slot, size_t nElements, T* buffer) -> decltype(auto)
{
  // after previous relocation this (hence its data members) are not guaranteed to be valid
  auto* old = get(buffer->data());
  auto* thisBlock = &(old->mBlocks[slot]);
  auto* thisMetadata = &(old->mMetadata[slot]);

  // resize underlying buffer of block if necessary and update all pointers.
  auto* const blockHead = get(thisBlock->registry->head);                // extract pointer from the block, as "this" might be invalid
  const size_t additionalSize = blockHead->estimateBlockSize(nElements); // additionalSize is in bytes!!!
  if (additionalSize >= thisBlock->registry->getFreeSize()) {
    LOGP(debug, "Slot {} with {} available words needs to allocate {} bytes for a total of {} words.", slot, thisBlock->registry->getFreeSize(), additionalSize, nElements);
    if (buffer) {
      blockHead->expand(*buffer, blockHead->size() + (additionalSize - blockHead->getFreeSize()));
      thisMetadata = &(get(buffer->data())->mMetadata[slot]);
      thisBlock = &(get(buffer->data())->mBlocks[slot]); // in case of resizing this and any this.xxx becomes invalid
    } else {
      throw std::runtime_error("failed to allocate additional space in provided external buffer");
    }
  }
  return std::make_pair(thisBlock, thisMetadata);
};

template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
o2::ctf::CTFIOSize EncodedBlocks<H, N, W>::entropyCodeRANSCompat(const input_IT srcBegin, const input_IT srcEnd, int slot, uint8_t symbolTablePrecision, buffer_T* buffer, const std::any& encoderExt, float memfc)
{
  using storageBuffer_t = W;
  using input_t = typename std::iterator_traits<input_IT>::value_type;
  using ransEncoder_t = typename rans::compat::encoder_type<input_t>;
  using ransState_t = typename ransEncoder_t::coder_type::state_type;
  using ransStream_t = typename ransEncoder_t::stream_type;

  // assert at compile time that output types align so that padding is not necessary.
  static_assert(std::is_same_v<storageBuffer_t, ransStream_t>);
  static_assert(std::is_same_v<storageBuffer_t, typename rans::count_t>);

  auto* thisBlock = &mBlocks[slot];
  auto* thisMetadata = &mMetadata[slot];

  // build symbol statistics
  constexpr size_t SizeEstMarginAbs = 10 * 1024;
  const float SizeEstMarginRel = 1.5 * memfc;

  const size_t messageLength = std::distance(srcBegin, srcEnd);
  rans::DenseHistogram<input_t> frequencyTable{};
  rans::compat::encoder_type<input_t> inplaceEncoder{};

  try {
    std::tie(inplaceEncoder, frequencyTable) = [&]() {
      if (encoderExt.has_value()) {
        return std::make_tuple(ransEncoder_t{}, rans::DenseHistogram<input_t>{});
      } else {
        auto histogram = rans::makeDenseHistogram::fromSamples(srcBegin, srcEnd);
        auto encoder = rans::compat::makeEncoder::fromHistogram(histogram, symbolTablePrecision);
        return std::make_tuple(std::move(encoder), std::move(histogram));
      }
    }();
  } catch (const rans::HistogramError& error) {
    LOGP(warning, "Failed to build Dictionary for rANS encoding, using fallback option");
    return store(srcBegin, srcEnd, slot, this->FallbackStorageType, buffer);
  }
  const ransEncoder_t& encoder = encoderExt.has_value() ? std::any_cast<const ransEncoder_t&>(encoderExt) : inplaceEncoder;

  // estimate size of encode buffer
  int dataSize = rans::compat::calculateMaxBufferSizeB(messageLength, rans::compat::getAlphabetRangeBits(encoder.getSymbolTable())); // size in bytes
  // preliminary expansion of storage based on dict size + estimated size of encode buffer
  dataSize = SizeEstMarginAbs + int(SizeEstMarginRel * (dataSize / sizeof(storageBuffer_t))) + (sizeof(input_t) < sizeof(storageBuffer_t)); // size in words of output stream

  const auto view = rans::trim(rans::makeHistogramView(frequencyTable));
  std::tie(thisBlock, thisMetadata) = expandStorage(slot, view.size() + dataSize, buffer);

  // store dictionary first

  if (!view.empty()) {
    thisBlock->storeDict(view.size(), view.data());
    LOGP(info, "StoreDict {} bytes, offs: {}:{}", view.size() * sizeof(W), thisBlock->getOffsDict(), thisBlock->getOffsDict() + view.size() * sizeof(W));
  }
  // vector of incompressible literal symbols
  std::vector<input_t> literals;
  // directly encode source message into block buffer.
  storageBuffer_t* const blockBufferBegin = thisBlock->getCreateData();
  const size_t maxBufferSize = thisBlock->registry->getFreeSize(); // note: "this" might be not valid after expandStorage call!!!
  const auto [encodedMessageEnd, literalsEnd] = encoder.process(srcBegin, srcEnd, blockBufferBegin, std::back_inserter(literals));
  rans::utils::checkBounds(encodedMessageEnd, blockBufferBegin + maxBufferSize / sizeof(W));
  dataSize = encodedMessageEnd - thisBlock->getDataPointer();
  thisBlock->setNData(dataSize);
  thisBlock->realignBlock();
  LOGP(info, "StoreData {} bytes, offs: {}:{}", dataSize * sizeof(W), thisBlock->getOffsData(), thisBlock->getOffsData() + dataSize * sizeof(W));
  // update the size claimed by encode message directly inside the block

  // store incompressible symbols if any
  const size_t nLiteralSymbols = literals.size();
  const size_t nLiteralWords = [&]() {
    if (!literals.empty()) {
      const size_t nSymbols = literals.size();
      // introduce padding in case literals don't align;
      const size_t nLiteralSymbolsPadded = calculatePaddedSize<input_t, storageBuffer_t>(nSymbols);
      literals.resize(nLiteralSymbolsPadded, {});

      const size_t nLiteralStorageElems = calculateNDestTElements<input_t, storageBuffer_t>(nSymbols);
      std::tie(thisBlock, thisMetadata) = expandStorage(slot, nLiteralStorageElems, buffer);
      thisBlock->storeLiterals(nLiteralStorageElems, reinterpret_cast<const storageBuffer_t*>(literals.data()));
      LOGP(info, "StoreLiterals {} bytes, offs: {}:{}", nLiteralStorageElems * sizeof(W), thisBlock->getOffsLiterals(), thisBlock->getOffsLiterals() + nLiteralStorageElems * sizeof(W));
      return nLiteralStorageElems;
    }
    return size_t(0);
  }();

  LOGP(info, "Min, {} Max, {}, size, {}, nSamples {}", view.getMin(), view.getMax(), view.size(), frequencyTable.getNumSamples());

  *thisMetadata = detail::makeMetadataRansCompat<input_t, ransState_t, ransStream_t>(encoder.getNStreams(),
                                                                                     messageLength,
                                                                                     nLiteralSymbols,
                                                                                     encoder.getSymbolTable().getPrecision(),
                                                                                     view.getMin(),
                                                                                     view.getMax(),
                                                                                     view.size(),
                                                                                     dataSize,
                                                                                     nLiteralWords);

  return {0, thisMetadata->getUncompressedSize(), thisMetadata->getCompressedSize()};
}

template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
o2::ctf::CTFIOSize EncodedBlocks<H, N, W>::entropyCodeRANSV1(const input_IT srcBegin, const input_IT srcEnd, int slot, Metadata::OptStore opt, buffer_T* buffer, const std::any& encoderExt, float memfc)
{
  CTFIOSize encoderStatistics{};

  const size_t nSamples = std::distance(srcBegin, srcEnd);
  if (detail::mayPack(opt) && nSamples < PackingThreshold) {
    encoderStatistics = pack(srcBegin, srcEnd, slot, buffer);
  } else {

    if (encoderExt.has_value()) {
      encoderStatistics = encodeRANSV1External(srcBegin, srcEnd, slot, encoderExt, buffer, memfc);
    } else {
      encoderStatistics = encodeRANSV1Inplace(srcBegin, srcEnd, slot, opt, buffer, memfc);
    }
  }
  return encoderStatistics;
}

template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
CTFIOSize EncodedBlocks<H, N, W>::encodeRANSV1External(const input_IT srcBegin, const input_IT srcEnd, int slot, const std::any& encoderExt, buffer_T* buffer, double_t sizeEstimateSafetyFactor)
{
  using storageBuffer_t = W;
  using input_t = typename std::iterator_traits<input_IT>::value_type;
  using ransEncoder_t = typename internal::ExternalEntropyCoder<input_t>::encoder_type;
  using ransState_t = typename ransEncoder_t::coder_type::state_type;
  using ransStream_t = typename ransEncoder_t::stream_type;

  // assert at compile time that output types align so that padding is not necessary.
  static_assert(std::is_same_v<storageBuffer_t, ransStream_t>);
  static_assert(std::is_same_v<storageBuffer_t, typename rans::count_t>);

  auto* thisBlock = &mBlocks[slot];
  auto* thisMetadata = &mMetadata[slot];

  const size_t messageLength = std::distance(srcBegin, srcEnd);
  internal::ExternalEntropyCoder<input_t> encoder{std::any_cast<const ransEncoder_t&>(encoderExt)};

  const size_t payloadSizeWords = encoder.template computePayloadSizeEstimate<storageBuffer_t>(messageLength);
  std::tie(thisBlock, thisMetadata) = expandStorage(slot, payloadSizeWords, buffer);

  // encode payload
  auto encodedMessageEnd = encoder.encode(srcBegin, srcEnd, thisBlock->getCreateData(), thisBlock->getEndOfBlock());
  const size_t dataSize = std::distance(thisBlock->getCreateData(), encodedMessageEnd);
  thisBlock->setNData(dataSize);
  thisBlock->realignBlock();
  LOGP(info, "StoreData {} bytes, offs: {}:{}", dataSize * sizeof(storageBuffer_t), thisBlock->getOffsData(), thisBlock->getOffsData() + dataSize * sizeof(storageBuffer_t));
  // update the size claimed by encoded message directly inside the block

  // encode literals
  size_t literalsSize = 0;
  if (encoder.getNIncompressibleSamples() > 0) {
    const size_t literalsBufferSizeWords = encoder.template computePackedIncompressibleSize<storageBuffer_t>();
    std::tie(thisBlock, thisMetadata) = expandStorage(slot, literalsBufferSizeWords, buffer);
    auto literalsEnd = encoder.writeIncompressible(thisBlock->getCreateLiterals(), thisBlock->getEndOfBlock());
    literalsSize = std::distance(thisBlock->getCreateLiterals(), literalsEnd);
    thisBlock->setNLiterals(literalsSize);
    thisBlock->realignBlock();
    LOGP(info, "StoreLiterals {} bytes, offs: {}:{}", literalsSize * sizeof(storageBuffer_t), thisBlock->getOffsLiterals(), thisBlock->getOffsLiterals() + literalsSize * sizeof(storageBuffer_t));
  }

  // write metadata
  const auto& symbolTable = encoder.getEncoder().getSymbolTable();
  *thisMetadata = detail::makeMetadataRansV1<input_t, ransState_t, ransStream_t>(encoder.getEncoder().getNStreams(),
                                                                                 rans::utils::getStreamingLowerBound_v<typename ransEncoder_t::coder_type>,
                                                                                 messageLength,
                                                                                 encoder.getNIncompressibleSamples(),
                                                                                 symbolTable.getPrecision(),
                                                                                 symbolTable.getOffset(),
                                                                                 symbolTable.getOffset() + symbolTable.size(),
                                                                                 encoder.getIncompressibleSymbolOffset(),
                                                                                 encoder.getIncompressibleSymbolPackingBits(),
                                                                                 0,
                                                                                 dataSize,
                                                                                 literalsSize);

  return {0, thisMetadata->getUncompressedSize(), thisMetadata->getCompressedSize()};
};

template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
CTFIOSize EncodedBlocks<H, N, W>::encodeRANSV1Inplace(const input_IT srcBegin, const input_IT srcEnd, int slot, Metadata::OptStore opt, buffer_T* buffer, double_t sizeEstimateSafetyFactor)
{
  using storageBuffer_t = W;
  using input_t = typename std::iterator_traits<input_IT>::value_type;
  using ransEncoder_t = typename rans::denseEncoder_type<input_t>;
  using ransState_t = typename ransEncoder_t::coder_type::state_type;
  using ransStream_t = typename ransEncoder_t::stream_type;

  // assert at compile time that output types align so that padding is not necessary.
  static_assert(std::is_same_v<storageBuffer_t, ransStream_t>);
  static_assert(std::is_same_v<storageBuffer_t, typename rans::count_t>);

  auto* thisBlock = &mBlocks[slot];
  auto* thisMetadata = &mMetadata[slot];

  internal::InplaceEntropyCoder<input_t> encoder{};
  rans::SourceProxy<input_IT> proxy{srcBegin, srcEnd, [](input_IT begin, input_IT end) {
                                      const size_t nSamples = std::distance(begin, end);
                                      return (!std::is_pointer_v<input_IT> && (nSamples < rans::utils::pow2(23)));
                                    }};

  try {
    if (proxy.isCached()) {
      encoder = internal::InplaceEntropyCoder<input_t>{proxy.beginCache(), proxy.endCache()};
    } else {
      encoder = internal::InplaceEntropyCoder<input_t>{proxy.beginIter(), proxy.endIter()};
    }
  } catch (const rans::HistogramError& error) {
    LOGP(warning, "Failed to build Dictionary for rANS encoding, using fallback option");
    if (proxy.isCached()) {
      return store(proxy.beginCache(), proxy.endCache(), slot, this->FallbackStorageType, buffer);
    } else {
      return store(proxy.beginIter(), proxy.endIter(), slot, this->FallbackStorageType, buffer);
    }
  }

  const rans::Metrics<input_t>& metrics = encoder.getMetrics();

  if constexpr (sizeof(input_t) > 2) {
    const auto& dp = metrics.getDatasetProperties();
    LOGP(info, "Metrics:{{slot: {}, numSamples: {}, min: {}, max: {}, alphabetRangeBits: {}, nUsedAlphabetSymbols: {}, preferPacking: {}}}", slot, dp.numSamples, dp.min, dp.max, dp.alphabetRangeBits, dp.nUsedAlphabetSymbols, metrics.getSizeEstimate().preferPacking());
  }

  if (detail::mayPack(opt) && metrics.getSizeEstimate().preferPacking()) {
    if (proxy.isCached()) {
      return pack(proxy.beginCache(), proxy.endCache(), slot, metrics, buffer);
    } else {
      return pack(proxy.beginIter(), proxy.endIter(), slot, metrics, buffer);
    };
  }

  encoder.makeEncoder();

  const rans::SizeEstimate sizeEstimate = metrics.getSizeEstimate();
  const size_t bufferSizeWords = rans::utils::nBytesTo<storageBuffer_t>((sizeEstimate.getCompressedDictionarySize() +
                                                                         sizeEstimate.getCompressedDatasetSize() +
                                                                         sizeEstimate.getIncompressibleSize()) *
                                                                        sizeEstimateSafetyFactor);
  std::tie(thisBlock, thisMetadata) = expandStorage(slot, bufferSizeWords, buffer);

  // encode dict
  auto encodedDictEnd = encoder.writeDictionary(thisBlock->getCreateDict(), thisBlock->getEndOfBlock());
  const size_t dictSize = std::distance(thisBlock->getCreateDict(), encodedDictEnd);
  thisBlock->setNDict(dictSize);
  thisBlock->realignBlock();
  LOGP(info, "StoreDict {} bytes, offs: {}:{}", dictSize * sizeof(storageBuffer_t), thisBlock->getOffsDict(), thisBlock->getOffsDict() + dictSize * sizeof(storageBuffer_t));

  // encode payload
  auto encodedMessageEnd = thisBlock->getCreateData();
  if (proxy.isCached()) {
    encodedMessageEnd = encoder.encode(proxy.beginCache(), proxy.endCache(), thisBlock->getCreateData(), thisBlock->getEndOfBlock());
  } else {
    encodedMessageEnd = encoder.encode(proxy.beginIter(), proxy.endIter(), thisBlock->getCreateData(), thisBlock->getEndOfBlock());
  }
  const size_t dataSize = std::distance(thisBlock->getCreateData(), encodedMessageEnd);
  thisBlock->setNData(dataSize);
  thisBlock->realignBlock();
  LOGP(info, "StoreData {} bytes, offs: {}:{}", dataSize * sizeof(storageBuffer_t), thisBlock->getOffsData(), thisBlock->getOffsData() + dataSize * sizeof(storageBuffer_t));
  // update the size claimed by encoded message directly inside the block

  // encode literals
  size_t literalsSize{};
  if (encoder.getNIncompressibleSamples() > 0) {
    auto literalsEnd = encoder.writeIncompressible(thisBlock->getCreateLiterals(), thisBlock->getEndOfBlock());
    literalsSize = std::distance(thisBlock->getCreateLiterals(), literalsEnd);
    thisBlock->setNLiterals(literalsSize);
    thisBlock->realignBlock();
    LOGP(info, "StoreLiterals {} bytes, offs: {}:{}", literalsSize * sizeof(storageBuffer_t), thisBlock->getOffsLiterals(), thisBlock->getOffsLiterals() + literalsSize * sizeof(storageBuffer_t));
  }

  // write metadata
  *thisMetadata = detail::makeMetadataRansV1<input_t, ransState_t, ransStream_t>(encoder.getNStreams(),
                                                                                 rans::utils::getStreamingLowerBound_v<typename ransEncoder_t::coder_type>,
                                                                                 std::distance(srcBegin, srcEnd),
                                                                                 encoder.getNIncompressibleSamples(),
                                                                                 encoder.getSymbolTablePrecision(),
                                                                                 *metrics.getCoderProperties().min,
                                                                                 *metrics.getCoderProperties().max,
                                                                                 metrics.getDatasetProperties().min,
                                                                                 metrics.getDatasetProperties().alphabetRangeBits,
                                                                                 dictSize,
                                                                                 dataSize,
                                                                                 literalsSize);

  return {0, thisMetadata->getUncompressedSize(), thisMetadata->getCompressedSize()};
}; // namespace ctf

template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
o2::ctf::CTFIOSize EncodedBlocks<H, N, W>::pack(const input_IT srcBegin, const input_IT srcEnd, int slot, rans::Metrics<typename std::iterator_traits<input_IT>::value_type> metrics, buffer_T* buffer)
{
  using storageBuffer_t = W;
  using input_t = typename std::iterator_traits<input_IT>::value_type;

  const size_t messageLength = std::distance(srcBegin, srcEnd);

  internal::Packer<input_t> packer{metrics};
  size_t packingBufferWords = packer.template getPackingBufferSize<storageBuffer_t>(messageLength);
  auto [thisBlock, thisMetadata] = expandStorage(slot, packingBufferWords, buffer);

  auto packedMessageEnd = packer.pack(srcBegin, srcEnd, thisBlock->getCreateData(), thisBlock->getEndOfBlock());
  const size_t packeSize = std::distance(thisBlock->getCreateData(), packedMessageEnd);
  thisBlock->setNData(packeSize);
  thisBlock->realignBlock();

  LOGP(info, "StoreData {} bytes, offs: {}:{}", packeSize * sizeof(storageBuffer_t), thisBlock->getOffsData(), thisBlock->getOffsData() + packeSize * sizeof(storageBuffer_t));

  *thisMetadata = detail::makeMetadataPack<input_t>(messageLength, packer.getPackingWidth(), packer.getOffset(), packeSize);
  return {0, thisMetadata->getUncompressedSize(), thisMetadata->getCompressedSize()};
};

template <typename H, int N, typename W>
template <typename input_IT, typename buffer_T>
o2::ctf::CTFIOSize EncodedBlocks<H, N, W>::store(const input_IT srcBegin, const input_IT srcEnd, int slot, Metadata::OptStore opt, buffer_T* buffer)
{
  using storageBuffer_t = W;
  using input_t = typename std::iterator_traits<input_IT>::value_type;

  const size_t messageLength = std::distance(srcBegin, srcEnd);
  // introduce padding in case literals don't align;
  const size_t nSourceElemsPadded = calculatePaddedSize<input_t, storageBuffer_t>(messageLength);
  std::vector<input_t> tmp(nSourceElemsPadded, {});
  std::copy(srcBegin, srcEnd, std::begin(tmp));

  const size_t nBufferElems = calculateNDestTElements<input_t, storageBuffer_t>(messageLength);
  auto [thisBlock, thisMetadata] = expandStorage(slot, nBufferElems, buffer);
  thisBlock->storeData(nBufferElems, reinterpret_cast<const storageBuffer_t*>(tmp.data()));

  *thisMetadata = detail::makeMetadataStore<input_t, storageBuffer_t>(messageLength, opt, nBufferElems);

  return {0, thisMetadata->getUncompressedSize(), thisMetadata->getCompressedSize()};
};

/// create a special EncodedBlocks containing only dictionaries made from provided vector of frequency tables
template <typename H, int N, typename W>
std::vector<char> EncodedBlocks<H, N, W>::createDictionaryBlocks(const std::vector<rans::DenseHistogram<int32_t>>& vfreq, const std::vector<Metadata>& vmd)
{

  if (vfreq.size() != N) {
    throw std::runtime_error(fmt::format("mismatch between the size of frequencies vector {} and number of blocks {}", vfreq.size(), N));
  }
  size_t sz = alignSize(sizeof(EncodedBlocks<H, N, W>));
  for (int ib = 0; ib < N; ib++) {
    sz += Block<W>::estimateSize(vfreq[ib].size());
  }
  std::vector<char> vdict(sz); // memory space for dictionary
  auto dictBlocks = create(vdict.data(), sz);
  for (int ib = 0; ib < N; ib++) {
    const auto& thisHistogram = vfreq[ib];
    const auto view = rans::trim(rans::makeHistogramView(thisHistogram));

    if (!view.empty()) {
      LOG(info) << "adding dictionary of " << view.size() << " words for block " << ib << ", min/max= " << view.getMin() << "/" << view.getMax();
      dictBlocks->mBlocks[ib].storeDict(view.size(), view.data());
      dictBlocks = get(vdict.data()); // !!! rellocation might have invalidated dictBlocks pointer
      dictBlocks->mMetadata[ib] = vmd[ib];
      dictBlocks->mMetadata[ib].opt = Metadata::OptStore::ROOTCompression; // we will compress the dictionary with root!
      dictBlocks->mBlocks[ib].realignBlock();
    } else {
      dictBlocks->mMetadata[ib].opt = Metadata::OptStore::NONE;
    }
    dictBlocks->mRegistry.nFilledBlocks++;
  }
  return vdict;
}
#endif

template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::dump(const std::string& prefix, int ncol) const
{
  for (int ibl = 0; ibl < getNBlocks(); ibl++) {
    const auto& blc = getBlock(ibl);
    std::string ss;
    LOGP(info, "{} Bloc:{} Dict: {} words", prefix, ibl, blc.getNDict());
    const auto* ptr = blc.getDict();
    for (int i = 0; i < blc.getNDict(); i++) {
      if (i && (i % ncol) == 0) {
        LOG(info) << ss;
        ss.clear();
      }
      ss += fmt::format(" {:#010x}", ptr[i]);
    }
    if (!ss.empty()) {
      LOG(info) << ss;
      ss.clear();
    }
    LOG(info) << "\n";
    LOGP(info, "{} Bloc:{} Data: {} words", prefix, ibl, blc.getNData());
    ptr = blc.getData();
    for (int i = 0; i < blc.getNData(); i++) {
      if (i && (i % ncol) == 0) {
        LOG(info) << ss;
        ss.clear();
      }
      ss += fmt::format(" {:#010x}", ptr[i]);
    }
    if (!ss.empty()) {
      LOG(info) << ss;
      ss.clear();
    }
    LOG(info) << "\n";
    LOGP(info, "{} Bloc:{} Literals: {} words", prefix, ibl, blc.getNLiterals());
    ptr = blc.getData();
    for (int i = 0; i < blc.getNLiterals(); i++) {
      if (i && (i % 20) == 0) {
        LOG(info) << ss;
        ss.clear();
      }
      ss += fmt::format(" {:#010x}", ptr[i]);
    }
    if (!ss.empty()) {
      LOG(info) << ss;
      ss.clear();
    }
    LOG(info) << "\n";
  }
}

} // namespace ctf
} // namespace o2

#endif
