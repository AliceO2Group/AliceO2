// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EncodedBlock.h
/// \brief Set of entropy-encoded blocks

///  Used to store a CTF of particular detector. Can be build as a flat buffer which can be directly messaged between DPL devices

#ifndef ALICEO2_ENCODED_BLOCKS_H
#define ALICEO2_ENCODED_BLOCKS_H

#include <type_traits>
#include <Rtypes.h>
#include "rANS/rans.h"
#include "TTree.h"
#include "CommonUtils/StringUtils.h"
#include "Framework/Logger.h"

namespace o2
{
namespace ctf
{
using namespace o2::rans;
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

///>>======================== Auxiliary classes =======================>>

/// registry struct for the buffer start and offsets of writable space
struct Registry {
  char* head = nullptr;
  int nFilledBlocks = 0;    // number of filled blocks = next block to fill (must be strictly consecutive)
  size_t offsFreeStart = 0; // offset of the start of the writable space (wrt head), in bytes!!!
  size_t size = 0;          // full size in bytes!!!

  /// calculate the pointer of the head of the writable space
  char* getFreeBlockStart() const
  {
    assert(offsFreeStart <= size);
    return head + offsFreeStart;
  }

  /// advance the head of the writeable space
  void shrinkFreeBlock(size_t sz)
  {
    offsFreeStart = alignSize(offsFreeStart + sz);
    assert(offsFreeStart <= size);
  }

  /// size in bytes available to fill data
  size_t getFreeSize() const
  {
    return size - offsFreeStart;
  }

  ClassDefNV(Registry, 1);
};

/// binary blob for single entropy-compressed column: metadata + (optional) dictionary and data buffer + their sizes
template <typename W = uint32_t>
struct Block {

  Registry* registry = nullptr; //! non-persistent info for in-memory ops
  int nDict = 0;                // dictionary length (if any)
  int nStored = 0;              // total payload: data + dictionary length
  W* payload = nullptr;         //[nStored];

  W* getData() { return payload + nDict; }
  W* getDict() { return nDict ? payload : nullptr; }
  const W* getData() const { return payload + nDict; }
  const W* getDict() const { return nDict ? payload : nullptr; }
  int getNData() const { return nStored - nDict; }
  int getNDict() const { return nDict; }

  ~Block()
  {
    if (!registry) { // this is a standalone block owning its data
      delete[] payload;
    }
  }

  /// clear itself
  void clear()
  {
    nDict = nStored = 0;
    payload = nullptr;
  }

  /// estimate free size needed to add new block
  static size_t estimateSize(int _ndict, int _ndata)
  {
    return alignSize((_ndict + _ndata) * sizeof(W));
  }

  /// store binary blob data (buffer filled from head to tail)
  void store(int _ndict, int _ndata, const W* _dict, const W* _data)
  {
    size_t sz = estimateSize(_ndict, _ndata);
    assert(registry); // this method is valid only for flat version, which has a registry
    assert(sz <= registry->getFreeSize());
    assert((_ndict > 0) == (_dict != nullptr));
    assert((_ndata > 0) == (_data != nullptr));
    nStored = _ndict + _ndata;
    nDict = _ndict;
    if (nStored) {
      auto ptr = payload = reinterpret_cast<W*>(registry->getFreeBlockStart());
      if (_dict) {
        memcpy(ptr, _dict, _ndict * sizeof(W));
        ptr += _ndict;
      }
      if (_data) {
        memcpy(ptr, _data, _ndata * sizeof(W));
      }
    }
    registry->shrinkFreeBlock(sz);
  }

  /// relocate to different head position
  void relocate(const char* oldHead, char* newHeadData, char* newHeadRegistry)
  {
    payload = relocatePointer(oldHead, newHeadData, payload);
    registry = relocatePointer(oldHead, newHeadRegistry, registry);
  }

  ClassDefNV(Block, 1);
};

///<<======================== Auxiliary classes =======================<<

template <typename H, int N, typename W = uint32_t>
class EncodedBlocks
{
 public:
  typedef EncodedBlocks<H, N, W> base;

  void setHeader(const H& h) { mHeader = h; }
  const H& getHeader() const { return mHeader; }
  H& getHeader() { return mHeader; }

  const auto& getMetadata() const { return mMetadata; }

  void setANSHeader(const o2::rans::Header& h) { mANSHeader = h; }
  const o2::rans::Header& getANSHeader() const { return mANSHeader; }
  o2::rans::Header& getANSHeader() { return mANSHeader; }

  static constexpr int getNBlocks() { return N; }

  static size_t getMinAlignedSize() { return alignSize(sizeof(base)); }

  /// cast arbitrary buffer head to container class. Head is supposed to respect the alignment
  static auto get(void* head) { return reinterpret_cast<EncodedBlocks*>(head); }
  static const auto get(const void* head) { return reinterpret_cast<const EncodedBlocks*>(head); }

  /// get const image of the container wrapper, with pointers in the image relocated to new head
  static const auto getImage(const void* newHead);

  /// create container from arbitrary buffer of predefined size (in bytes!!!). Head is supposed to respect the alignment
  static auto create(void* head, size_t sz);

  /// create container from vector. Head is supposed to respect the alignment
  template <typename VD>
  static auto create(VD& v);

  /// estimate free size needed to add new block
  static size_t estimateBlockSize(int _ndict, int _ndata) { return Block<W>::estimateSize(_ndict, _ndata); }

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
  template <typename VB>
  static auto expand(VB& buffer, size_t newsizeBytes);

  /// copy itself to flat buffer created on the fly from the vector
  template <typename V>
  void copyToFlat(V& vec);

  /// copy itself to flat buffer created on the fly at the provided pointer. The destination block should be at least of size estimateSize()
  void copyToFlat(void* base) { fillFlatCopy(create(base, estimateSize())); }

  /// attach to tree
  void appendToTree(TTree& tree, const std::string& name) const;

  /// read from tree to non-flat object
  void readFromTree(TTree& tree, const std::string& name, int ev = 0);

  /// read from tree to destination buffer vector
  template <typename VD>
  static void readFromTree(VD& vec, TTree& tree, const std::string& name, int ev = 0);

  /// encode vector src to bloc at provided slot
  template <typename VE, typename VB>
  inline void encode(const VE& src, int slot, uint8_t probabilityBits, o2::rans::Metadata::OptStore opt, VB* buffer = nullptr)
  {
    encode(&(*src.begin()), &(*src.end()), slot, probabilityBits, opt, buffer);
  }

  /// encode vector src to bloc at provided slot
  template <typename S, typename VB>
  void encode(const S* const srcBegin, const S* const srcEnd, int slot, uint8_t probabilityBits, o2::rans::Metadata::OptStore opt, VB* buffer = nullptr);

  /// decode block at provided slot to destination vector (will be resized as needed)
  template <typename VD>
  void decode(VD& dest, int slot) const;

  /// decode block at provided slot to destination pointer, the needed space assumed to be available
  template <typename D>
  void decode(D* dest, int slot) const;

  /// print itself
  void print() const;

 protected:
  static_assert(N > 0, "number of encoded blocks < 1");

  Registry mRegistry;                //! not stored
  o2::rans::Header mANSHeader;       //  ANS header
  H mHeader;                         //  detector specific header
  std::array<Metadata, N> mMetadata; //  compressed block's details
  std::array<Block<W>, N> mBlocks;   //! this is in fact stored, but to overcome TBuffer limits we have to define the branches per block!!!

  /// setup internal structure and registry for given buffer size (in bytes!!!)
  void init(size_t sz);

  /// relocate to different head position, newHead points on start of the dynamic buffer holding the data.
  /// the address of the static part might be actually different (wrapper). This different newHead and
  /// wrapper addresses must be used when the buffer pointed by newHead is const (e.g. received from the
  /// DPL input), in this case we create a wrapper, which points on these const data
  static void relocate(const char* oldHead, char* newHead, char* wrapper, size_t newsize = 0);

  /// Estimate size of the buffer needed to store all compressed data in a contiguos block of memory, accounting for alignment
  /// This method is to be called after reading object from the tree as a non-flat object!
  size_t estimateSize() const;

  /// do the same using metadata info
  size_t estimateSizeFromMetadata() const;

  /// Create its own flat copy in the destination empty flat object
  void fillFlatCopy(EncodedBlocks& dest) const;

  /// add and fill single branch
  template <typename D>
  static void fillTreeBranch(TTree& tree, const std::string& brname, D& dt, int compLevel, int splitLevel = 99);

  /// read single branch
  template <typename D>
  static void readTreeBranch(TTree& tree, const std::string& brname, D& dt, int ev = 0);

  ClassDefNV(EncodedBlocks, 1);
};

///_____________________________________________________________________________
/// read from tree to non-flat object
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::readFromTree(TTree& tree, const std::string& name, int ev)
{
  readTreeBranch(tree, o2::utils::concat_string(name, "_wrapper."), *this, ev);
  for (int i = 0; i < N; i++) {
    readTreeBranch(tree, o2::utils::concat_string(name, "_block.", std::to_string(i), "."), mBlocks[i]);
  }
}

///_____________________________________________________________________________
/// read from tree to destination buffer vector
template <typename H, int N, typename W>
template <typename VD>
void EncodedBlocks<H, N, W>::readFromTree(VD& vec, TTree& tree, const std::string& name, int ev)
{
  auto tmp = create(vec);
  readTreeBranch(tree, o2::utils::concat_string(name, "_wrapper."), *tmp, ev);
  tmp = tmp->expand(vec, tmp->estimateSizeFromMetadata());
  for (int i = 0; i < N; i++) {
    Block<W> bl;
    readTreeBranch(tree, o2::utils::concat_string(name, "_block.", std::to_string(i), "."), bl);
    tmp->mBlocks[i].store(bl.getNDict(), bl.getNData(), bl.getDict(), bl.getData());
  }
}

///_____________________________________________________________________________
/// attach to tree
template <typename H, int N, typename W>
void EncodedBlocks<H, N, W>::appendToTree(TTree& tree, const std::string& name) const
{
  fillTreeBranch(tree, o2::utils::concat_string(name, "_wrapper."), const_cast<base&>(*this), WrappersCompressionLevel, WrappersSplitLevel);
  for (int i = 0; i < N; i++) {
    int compression = mMetadata[i].opt == o2::rans::Metadata::ROOTCompression ? 1 : 0;
    fillTreeBranch(tree, o2::utils::concat_string(name, "_block.", std::to_string(i), "."), const_cast<Block<W>&>(mBlocks[i]), compression);
  }
}

///_____________________________________________________________________________
/// read single branch
template <typename H, int N, typename W>
template <typename D>
inline void EncodedBlocks<H, N, W>::readTreeBranch(TTree& tree, const std::string& brname, D& dt, int ev)
{
  auto* br = tree.GetBranch(brname.c_str());
  assert(br);
  auto* ptr = &dt;
  br->SetAddress(&ptr);
  br->GetEntry(ev);
  br->ResetAddress();
}

///_____________________________________________________________________________
/// add and fill single branch
template <typename H, int N, typename W>
template <typename D>
inline void EncodedBlocks<H, N, W>::fillTreeBranch(TTree& tree, const std::string& brname, D& dt, int compLevel, int splitLevel)
{
  auto* br = tree.Branch(brname.c_str(), &dt, 512, splitLevel);
  br->SetCompressionLevel(compLevel);
  br->Fill();
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
    sz += alignSize((mMetadata[i].nDictWords + mMetadata[i].nDataWords) * sizeof(W));
  }
  return sz;
}

///_____________________________________________________________________________
/// expand the storage to new size in bytes
template <typename H, int N, typename W>
template <typename VB>
auto EncodedBlocks<H, N, W>::expand(VB& buffer, size_t newsizeBytes)
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
const auto EncodedBlocks<H, N, W>::getImage(const void* newHead)
{
  auto image(*get(newHead)); // 1st make a shalow copy
  // now fix its pointers
  // we don't modify newHead, but still need to remove constness for relocation interface
  relocate(image.mRegistry.head, const_cast<char*>(reinterpret_cast<const char*>(newHead)), reinterpret_cast<char*>(&image));
  image.print();

  return std::move(image);
}

///_____________________________________________________________________________
/// create container from arbitrary buffer of predefined size (in bytes!!!). Head is supposed to respect the alignment
template <typename H, int N, typename W>
inline auto EncodedBlocks<H, N, W>::create(void* head, size_t sz)
{
  auto b = get(head);
  b->init(sz);
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
void EncodedBlocks<H, N, W>::print() const
{
  LOG(INFO) << "Container " << N << " blocks, size: " << size() << " bytes, unused: " << getFreeSize();
  for (int i = 0; i < N; i++) {
    LOG(INFO) << "Block " << i << " NDictWords: " << mBlocks[i].getNDict() << " NDataWords: " << mBlocks[i].getNData();
  }
}

///_____________________________________________________________________________
template <typename H, int N, typename W>
template <typename VD>
inline void EncodedBlocks<H, N, W>::decode(VD& dest,       // destination container
                                           int slot) const // slot of the block to decode
{
  dest.resize(mMetadata[slot].messageLength); // allocate output buffer
  decode(dest.data(), slot);
}

///_____________________________________________________________________________
template <typename H, int N, typename W>
template <typename D>
void EncodedBlocks<H, N, W>::decode(D* dest,        // destination pointer
                                    int slot) const // slot of the block to decode
{
  // get references to the right data
  const auto& block = mBlocks[slot];
  const auto& md = mMetadata[slot];

  // decode
  if (md.opt == o2::rans::Metadata::EENCODE) {
    assert(block.getNDict());
    o2::rans::SymbolStatistics stats(block.getDict(), block.getDict() + block.getNDict(), md.min, md.max, md.messageLength);
    o2::rans::Decoder64<D> decoder(stats, md.probabilityBits);
    decoder.process(dest, block.getData(), md.messageLength);
  } else { // data was stored as is
    std::memcpy(dest, block.payload, md.messageLength * sizeof(D));
  }
}

///_____________________________________________________________________________
template <typename H, int N, typename W>
template <typename S, typename VB>
void EncodedBlocks<H, N, W>::encode(const S* const srcBegin,          // begin of source message
                                    const S* const srcEnd,            // end of source message
                                    int slot,                         // slot in encoded data to fill
                                    uint8_t probabilityBits,          // encoding into
                                    o2::rans::Metadata::OptStore opt, // option for data compression
                                    VB* buffer)                       // optional buffer (vector) providing memory for encoded blocks
{

  // symbol statistics and encoding
  assert(slot == mRegistry.nFilledBlocks);
  using stream_t = typename o2::rans::Encoder64<S>::stream_t;
  std::vector<stream_t> encoderBuffer;
  static_assert(std::is_same<W, stream_t>());
  stream_t* encodedMessageStart = nullptr;
  const stream_t* dictStart = nullptr;
  int dictSize = 0, dataSize = 0;
  o2::rans::SymbolStatistics stats{srcBegin, srcEnd}; // need to create it, even if not used
  if (opt == o2::rans::Metadata::EENCODE) {
    stats.rescaleToNBits(probabilityBits);
    const auto buffSize = o2::rans::calculateMaxBufferSize(stats.getMessageLength(),
                                                           stats.getAlphabetRangeBits(),
                                                           sizeof(S));
    encoderBuffer.resize(buffSize);
    const o2::rans::Encoder64<S> encoder{stats, probabilityBits};
    encodedMessageStart = &(*encoder.process(encoderBuffer.begin(), encoderBuffer.end(), srcBegin, srcEnd));
    dictStart = stats.getFrequencyTable().data();
    dictSize = stats.getFrequencyTable().size();
    dataSize = &(*encoderBuffer.end()) - encodedMessageStart; // number of elements to store
  } else {                                                    // store original data w/o EEncoding
    auto szb = (srcEnd - srcBegin) * sizeof(S);
    dataSize = szb / sizeof(stream_t) + (sizeof(S) < sizeof(stream_t));
    encoderBuffer.resize(dataSize);
    memcpy(encoderBuffer.data(), srcBegin, szb);
    encodedMessageStart = &(*encoderBuffer.begin());
  }
  auto szNeed = estimateBlockSize(dictSize, dataSize); // size in bytes!!!
  auto* bl = &mBlocks[slot];
  auto* meta = &mMetadata[slot];
  if (szNeed >= getFreeSize()) {
    LOG(INFO) << "Slot " << slot << ": free size: " << getFreeSize() << ", need " << szNeed;
    if (buffer) {
      expand(*buffer, size() + (szNeed - getFreeSize()));
      meta = &(get(buffer->data())->mMetadata[slot]);
      bl = &(get(buffer->data())->mBlocks[slot]); // in case of resizing this and any this.xxx becomes invalid
    } else {
      throw std::runtime_error("no room for encoded block in provided container");
    }
  }
  *meta = o2::rans::Metadata{stats.getMessageLength(), sizeof(uint64_t), sizeof(stream_t), probabilityBits, opt,
                             stats.getMinSymbol(), stats.getMaxSymbol(), dictSize, dataSize};
  bl->store(dictSize, dataSize, dictStart, encodedMessageStart);
  bl->registry->nFilledBlocks++;
}

} // namespace ctf
} // namespace o2

#endif
