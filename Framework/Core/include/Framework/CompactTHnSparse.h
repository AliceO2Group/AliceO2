// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_COMPACTTHNSPARSE_H_
#define O2_FRAMEWORK_COMPACTTHNSPARSE_H_

// A sparse n-dimensional histogram with the THnBase interface and a fraction of
// the memory of THnSparse.
//
// THnSparse spends ~60 of its ~70 bytes per filled bin on a hash table from
// bin coordinates to bin index, and briefly needs about three times its size
// whenever that table grows. Here the filled bins are kept in a
// BlockSparseArray keyed by the linear bin index: sizeof(T) + 2 bytes per filled
// bin in sparse regions (+8 with Sumw2), dense arrays where the bins are dense.
//
// Bin indices are linear indices over all cells, under/overflow included (as in
// THn), so GetBinContent(bin) works for any bin; GetNbins() is the number of
// filled bins (as in THnSparse). Everything else (Fill, Add, Merge, Scale,
// projections, statistics) comes from THnBase. toTHnSparse() gives the
// equivalent THnSparse, with the same bins, errors and statistics.
//
// It is meant to be filled and written out (the HistogramRegistry writes it as
// the equivalent THnSparse). Reading it back by bin number is where it differs
// from THnSparse, whose bin numbers count the filled bins: loops over
// i < GetNbins() with GetBinContent(i, coord) visit cells, not filled bins (use
// THnIter instead), and the THnBase functions built on that numbering
// (ComputeIntegral, GetRandom, PrintEntries, Print("c"), Fit) throw here.

#include "Framework/BlockSparseArray.h"
#include "Framework/RuntimeError.h"

#include <TAxis.h>
#include <THnBase.h>
#include <THnSparse.h>
#include <THnSparse_Internal.h>
#include <TArrayD.h>
#include <TArrayF.h>
#include <TBuffer.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

/// Type-independent interface, used by the HistogramRegistry
class CompactTHnSparse : public THnBase
{
 public:
  /// The equivalent THnSparse (THnSparseF for float contents, THnSparseD for double), owned by the caller.
  virtual THnSparse* toTHnSparse(const char* name = nullptr) const = 0;
  /// Bytes of bin storage (blocks and block tables, excluding malloc overhead)
  virtual size_t storageBytes() const = 0;
  /// Size of one bin content in bytes
  virtual int contentSize() const = 0;

  // THnBase implements these (non-virtual) with THnSparse's numbering of the filled bins
  // 0..GetNbins()-1, which is not the bin numbering here: they would use the wrong bins
  // (ComputeIntegral would write out of bounds). Read the written THnSparse instead.
  Double_t ComputeIntegral() { unsupported("ComputeIntegral"); }
  void GetRandom(Double_t*, Bool_t = kTRUE) { unsupported("GetRandom"); }
  void PrintEntries(Long64_t = 0, Long64_t = -1, Option_t* = nullptr) const { unsupported("PrintEntries"); }
  TFitResultPtr Fit(TF1*, Option_t* = "", Option_t* = "") { unsupported("Fit"); }
  void Print(Option_t* option = "") const override
  {
    if (option && (std::strchr(option, 'c') || std::strchr(option, 'C'))) {
      unsupported("Print with the bin contents");
    }
    THnBase::Print(option);
  }

 protected:
  CompactTHnSparse() = default;
  using THnBase::THnBase;

 private:
  [[noreturn]] void unsupported(const char* what) const
  {
    throw o2::framework::runtime_error_f("%s is not supported by %s (%s): convert it with toTHnSparse() first", what, IsA()->GetName(), GetName());
  }

  ClassDefOverride(CompactTHnSparse, 1); // Interface of the sparse n-dimensional histograms with compact bin storage
};

template <typename T>
class CompactTHnSparseT : public CompactTHnSparse
{
 public:
  CompactTHnSparseT() = default;
  CompactTHnSparseT(const char* name, const char* title, Int_t dim, const Int_t* nbins,
                    const Double_t* xmin = nullptr, const Double_t* xmax = nullptr, Int_t blockBits = o2::framework::BlockSparseArray<T>::kDefaultBlockBits)
    : CompactTHnSparse(name, title, dim, nbins, xmin, xmax), mBlockBits(blockBits)
  {
    InitStorage(const_cast<Int_t*>(nbins), 0);
  }
  CompactTHnSparseT(const char* name, const char* title, const std::vector<TAxis>& axes, Int_t blockBits = o2::framework::BlockSparseArray<T>::kDefaultBlockBits)
    : CompactTHnSparse(name, title, axes), mBlockBits(blockBits)
  {
    std::vector<Int_t> nbins;
    for (auto& a : axes) {
      nbins.push_back(a.GetNbins());
    }
    InitStorage(nbins.data(), 0);
  }
  ~CompactTHnSparseT() override = default;

  ROOT::Internal::THnBaseBinIter* CreateIter(Bool_t respectAxisRange) const override;
  /// Number of filled bins
  Long64_t GetNbins() const override { return mStore ? Long64_t(mStore->filledBins()) : 0; }

  /// Linear bin index of the bin coordinates, -1 if one is outside [0, nbins + 1]
  Long64_t GetBin(const Int_t* idx) const override
  {
    Long64_t bin = 0;
    for (Int_t d = 0; d < fNdimensions; ++d) {
      if (idx[d] < 0 || idx[d] > GetAxis(d)->GetNbins() + 1) {
        return -1;
      }
      bin += Long64_t(idx[d]) * mStride[d];
    }
    return bin;
  }
  Long64_t GetBin(const Double_t* x) const override
  {
    Long64_t bin = 0; // FindBin of a value is always in [0, nbins + 1]
    for (Int_t d = 0; d < fNdimensions; ++d) {
      bin += Long64_t(GetAxis(d)->FindBin(x[d])) * mStride[d];
    }
    return bin;
  }
  Long64_t GetBin(const char* name[]) const override
  {
    Long64_t bin = 0;
    for (Int_t d = 0; d < fNdimensions; ++d) {
      Int_t i = GetAxis(d)->FindBin(name[d]);
      if (i < 0) {
        return -1; // unknown label
      }
      bin += Long64_t(i) * mStride[d];
    }
    return bin;
  }
  Long64_t GetBin(const Int_t* idx, Bool_t /*allocate*/ = kTRUE) override { return std::as_const(*this).GetBin(idx); }
  Long64_t GetBin(const Double_t* x, Bool_t /*allocate*/ = kTRUE) override { return std::as_const(*this).GetBin(x); }
  Long64_t GetBin(const char* name[], Bool_t /*allocate*/ = kTRUE) override { return std::as_const(*this).GetBin(name); }

  // Contents are summed in double and stored as T, as THnSparse does, so both give identical bins
  void FillBin(Long64_t bin, Double_t w) override
  {
    if (!isValid(bin)) {
      return;
    }
    auto r = mStore->ref(uint64_t(bin), true);
    *r.v = T(Double_t(*r.v) + w);
    if (r.s) {
      *r.s += w * w;
    }
    FillBinBase(w);
  }

  using THnBase::SetBinContent;
  void SetBinContent(Long64_t bin, Double_t v) override
  {
    if (isValid(bin)) {
      *mStore->ref(uint64_t(bin), true).v = T(v);
    }
  }
  void SetBinError2(Long64_t bin, Double_t e2) override
  {
    if (!isValid(bin)) {
      return;
    }
    if (!GetCalculateErrors()) {
      Sumw2();
    }
    *mStore->ref(uint64_t(bin), true).s = e2;
  }
  using THnBase::AddBinContent;
  void AddBinContent(Long64_t bin, Double_t v = 1.) override
  {
    if (!isValid(bin)) {
      return;
    }
    if (auto r = mStore->ref(uint64_t(bin), v != 0.)) { // adding 0 does not make a bin filled
      *r.v = T(Double_t(*r.v) + v);
    }
  }
  void AddBinError2(Long64_t bin, Double_t e2) override
  {
    if (!isValid(bin)) {
      return;
    }
    if (!GetCalculateErrors()) {
      Sumw2();
    }
    if (auto r = mStore->ref(uint64_t(bin), e2 != 0.)) {
      *r.s += e2;
    }
  }
  using THnBase::GetBinContent;
  /// Content of the bin with linear index `bin`; its coordinates in `idx` if given
  Double_t GetBinContent(Long64_t bin, Int_t* idx = nullptr) const override
  {
    if (!isValid(bin)) {
      return 0.;
    }
    if (idx) {
      coordinates(bin, idx);
    }
    return mStore->get(uint64_t(bin));
  }
  Double_t GetBinError2(Long64_t bin) const override
  {
    if (!isValid(bin)) {
      return 0.;
    }
    return GetCalculateErrors() ? mStore->getSumw2(uint64_t(bin)) : mStore->get(uint64_t(bin));
  }

  void Sumw2() override
  {
    if (GetCalculateErrors()) {
      return;
    }
    fTsumw2 = 0.;
    mStore->enableSumw2();
  }
  void Reset(Option_t* option = "") override
  {
    ResetBase(option);
    makeStore();
  }

  /// Whether `bin` is a linear bin index of this histogram (0 <= bin < number of cells)
  bool isValid(Long64_t bin) const { return mStore && bin >= 0 && bin < mCells; }
  /// Coordinates (bin numbers per axis) of the linear bin index `bin`
  void coordinates(Long64_t bin, Int_t* idx) const
  {
    for (Int_t d = 0; d < fNdimensions; ++d) {
      idx[d] = Int_t(bin / mStride[d]);
      bin %= mStride[d];
    }
  }

  THnSparse* toTHnSparse(const char* name = nullptr) const override;
  size_t storageBytes() const override { return mStore ? mStore->bytes() : 0; }
  int contentSize() const override { return sizeof(T); }
  const o2::framework::BlockSparseArray<T>& storage() const { return *mStore; }

 protected:
  void InitStorage(Int_t* nbins, Int_t /*chunkSize*/) override;

 private:
  void makeStore();

  Int_t mBlockBits = o2::framework::BlockSparseArray<T>::kDefaultBlockBits;
  std::vector<Long64_t> mStride;                                        ///<! linear index stride per axis
  Long64_t mCells = 0;                                                  ///<! number of cells, under/overflow included
  Int_t mKeyBits = 0;                                                   ///<! bits of the linear bin index
  std::unique_ptr<o2::framework::BlockSparseArray<T>> mStore = nullptr; ///<! the filled bins

  ClassDefOverride(CompactTHnSparseT, 1); // Sparse n-dimensional histogram with compact bin storage
};

typedef CompactTHnSparseT<Float_t> CompactTHnSparseF;
typedef CompactTHnSparseT<Double_t> CompactTHnSparseD;

namespace o2::framework::detail
{
/// Iterates over the filled bins of a CompactTHnSparseT in increasing bin index
template <typename T>
class CompactTHnSparseIter : public ROOT::Internal::THnBaseBinIter
{
 public:
  CompactTHnSparseIter(Bool_t respectAxisRange, const CompactTHnSparseT<T>* hist)
    : ROOT::Internal::THnBaseBinIter(respectAxisRange), mHist(hist), mCoord(hist->GetNdimensions(), -1) {}

  Int_t GetCoord(Int_t dim) const override { return mCoord[dim]; }
  Long64_t Next(Int_t* coord = nullptr) override
  {
    uint64_t key;
    T v;
    double s;
    while (mHist && mHist->storage().next(mCursor, key, v, s)) {
      mHist->coordinates(Long64_t(key), mCoord.data());
      if (RespectsAxisRange() && !mHist->IsInRange(mCoord.data())) {
        fHaveSkippedBin = kTRUE;
        continue;
      }
      if (coord) {
        std::copy(mCoord.begin(), mCoord.end(), coord);
      }
      return Long64_t(key);
    }
    mHist = nullptr;
    return -1;
  }

 private:
  const CompactTHnSparseT<T>* mHist;
  std::vector<Int_t> mCoord;
  typename o2::framework::BlockSparseArray<T>::Cursor mCursor;
};
} // namespace o2::framework::detail

template <typename T>
ROOT::Internal::THnBaseBinIter* CompactTHnSparseT<T>::CreateIter(Bool_t respectAxisRange) const
{
  return new o2::framework::detail::CompactTHnSparseIter<T>(respectAxisRange, this);
}

template <typename T>
void CompactTHnSparseT<T>::InitStorage(Int_t* nbins, Int_t /*chunkSize*/)
{
  // last axis fastest, as in THn
  constexpr uint64_t maxCells = uint64_t(1) << 62;
  mStride.assign(fNdimensions, 1);
  uint64_t cells = 1;
  for (Int_t d = fNdimensions - 1; d >= 0; --d) {
    mStride[d] = Long64_t(cells);
    uint64_t n = uint64_t(nbins[d]) + 2;
    if (cells > maxCells / n) {
      Fatal("InitStorage", "%s: the %d axes have more than 2^62 cells (under/overflow included), which do not fit a 64-bit bin index; use THnSparse", GetName(), fNdimensions);
    }
    cells *= n;
  }
  mCells = Long64_t(cells);
  mKeyBits = 0;
  while (mKeyBits < 64 && (uint64_t(1) << mKeyBits) < cells) {
    ++mKeyBits;
  }
  makeStore();
}

template <typename T>
void CompactTHnSparseT<T>::makeStore()
{
  mStore = std::make_unique<o2::framework::BlockSparseArray<T>>(mKeyBits, mBlockBits, GetCalculateErrors());
}

namespace o2::framework::detail
{
// Builds a THnSparse from bins given in any order without its hash table from coordinates to
// bin index, which takes ~85 of THnSparse's ~95 bytes per bin here and is never written out:
// THnSparse rebuilds it from the bin chunks on the first lookup (FillExMap), as for a
// THnSparse read from a file. A subclass only to reach THnSparse's protected chunk functions
// and THnBase's statistics; it adds no members and no ClassDef, so IsA() and what is written
// out are those of THnSparseT<A>.
template <typename A>
class THnSparseBuilder : public THnSparseT<A>
{
 public:
  THnSparseBuilder(const char* name, const char* title, const std::vector<TAxis>& axes, THnBase const& source)
    : THnSparseT<A>(name, title, axes)
  {
    // THnSparse's compact coordinates: per axis the bits of nbins + 2, packed from the first
    // axis up (THnSparseCoordCompression in THnSparse.cxx; the format of fCoordinates on file)
    Int_t shift = 0;
    for (Int_t d = 0; d < this->fNdimensions; ++d) {
      mBitOffsets.push_back(shift);
      shift += numBits(this->GetAxis(d)->GetNbins() + 2);
    }
    mBufferSize = (shift + 7) / 8;
    mBuffer.assign(std::max(mBufferSize, Int_t(sizeof(ULong64_t))), 0);
    if (source.GetCalculateErrors()) {
      this->Sumw2();
    }
  }

  /// appends the bin with these (per-axis) coordinates; each bin must be appended only once
  void append(const Int_t* coord, Double_t content, Double_t error2)
  {
    pack(coord);
    THnSparseArrayChunk* chunk = this->GetNChunks() ? this->GetChunk(this->GetNChunks() - 1) : nullptr;
    if (!chunk || chunk->GetEntries() == this->GetChunkSize()) {
      chunk = this->AddChunk();
      if (chunk->fSingleCoordinateSize != mBufferSize) {
        throw runtime_error_f("THnSparseBuilder: coordinate size %d, THnSparse's is %d", mBufferSize, chunk->fSingleCoordinateSize);
      }
    }
    Int_t i = chunk->GetEntries();
    chunk->AddBin(i, mBuffer.data());
    chunk->fContent->SetAt(content, i);
    if (chunk->fSumw2) {
      chunk->fSumw2->SetAt(error2, i);
    }
    this->SetFilledBins(this->GetNbins() + 1);
  }

  void copyStatistics(THnBase const& source)
  {
    this->fEntries = source.GetEntries();
    this->fTsumw = source.GetSumw();
    this->fTsumw2 = source.GetSumw2();
    for (Int_t d = 0; d < this->fNdimensions; ++d) {
      this->fTsumwx[d] = source.GetSumwx(d);
      this->fTsumwx2[d] = source.GetSumwx2(d);
    }
  }

 private:
  static Int_t numBits(Int_t n)
  {
    Int_t r = (n > 0);
    while (n /= 2) {
      ++r;
    }
    return r;
  }

  // as THnSparseCoordCompression::SetBufferFromCoord
  void pack(const Int_t* coord)
  {
    if (mBufferSize <= 8) {
      ULong64_t l64buf = 0;
      for (Int_t d = 0; d < this->fNdimensions; ++d) {
        l64buf += ((ULong64_t)((UInt_t)coord[d])) << mBitOffsets[d];
      }
      std::memcpy(mBuffer.data(), &l64buf, sizeof(ULong64_t));
      return;
    }
    std::memset(mBuffer.data(), 0, mBufferSize);
    for (Int_t d = 0; d < this->fNdimensions; ++d) {
      ULong64_t val = coord[d];
      Char_t* pbuf = mBuffer.data() + mBitOffsets[d] / 8;
      const Int_t bitShift = mBitOffsets[d] % 8;
      *pbuf += 0xff & (val << bitShift);
      val = val >> (8 - bitShift);
      while (val) {
        ++pbuf;
        *pbuf += 0xff & val;
        val = val >> 8;
      }
    }
  }

  std::vector<Int_t> mBitOffsets;
  Int_t mBufferSize = 0;
  std::vector<Char_t> mBuffer;
};
} // namespace o2::framework::detail

template <typename T>
THnSparse* CompactTHnSparseT<T>::toTHnSparse(const char* name) const
{
  std::vector<TAxis> axes;
  for (Int_t d = 0; d < fNdimensions; ++d) {
    axes.push_back(*GetAxis(d));
  }
  using Array = std::conditional_t<std::is_same_v<T, Float_t>, TArrayF, TArrayD>;
  auto h = std::make_unique<o2::framework::detail::THnSparseBuilder<Array>>(name ? name : GetName(), GetTitle(), axes, *this);
  // every stored bin, zero-content ones included: THnSparse keeps those once allocated too
  std::vector<Int_t> idx(fNdimensions);
  mStore->forEach([&](uint64_t k, T v, double s) {
    coordinates(Long64_t(k), idx.data());
    h->append(idx.data(), v, s);
  });
  h->copyStatistics(*this);
  return h.release();
}

// The filled bins are written as (bin index, content[, sumw2]) arrays after the THnBase part.
namespace o2::framework::detail
{
// TBuffer's array functions take an Int_t count
template <typename V>
void readArray(TBuffer& b, V* p, Long64_t n)
{
  for (Long64_t done = 0; done < n;) {
    Int_t k = Int_t(std::min<Long64_t>(n - done, std::numeric_limits<Int_t>::max()));
    b.ReadFastArray(p + done, k);
    done += k;
  }
}
template <typename V>
void writeArray(TBuffer& b, const V* p, Long64_t n)
{
  for (Long64_t done = 0; done < n;) {
    Int_t k = Int_t(std::min<Long64_t>(n - done, std::numeric_limits<Int_t>::max()));
    b.WriteFastArray(p + done, k);
    done += k;
  }
}
} // namespace o2::framework::detail

template <typename T>
void CompactTHnSparseT<T>::Streamer(TBuffer& b)
{
  using namespace o2::framework::detail;
  if (b.IsReading()) {
    UInt_t start, count;
    b.ReadVersion(&start, &count);
    THnBase::Streamer(b);
    b >> mBlockBits;
    if (mBlockBits < 6 || mBlockBits > 16) {
      Error("Streamer", "invalid block size 2^%d, using 2^%d", mBlockBits, o2::framework::BlockSparseArray<T>::kDefaultBlockBits);
      mBlockBits = o2::framework::BlockSparseArray<T>::kDefaultBlockBits;
    }
    std::vector<Int_t> nbins(fNdimensions);
    for (Int_t d = 0; d < fNdimensions; ++d) {
      nbins[d] = GetAxis(d)->GetNbins();
    }
    InitStorage(nbins.data(), 0);
    Long64_t n;
    b >> n;
    if (n < 0 || n > mCells) {
      Error("Streamer", "%s: invalid number of filled bins %lld", GetName(), n);
      b.CheckByteCount(start, count, IsA());
      return;
    }
    std::vector<Long64_t> keys(n);
    std::vector<T> vals(n);
    std::vector<Double_t> sumw2;
    readArray(b, keys.data(), n);
    readArray(b, vals.data(), n);
    if (GetCalculateErrors()) {
      sumw2.resize(n);
      readArray(b, sumw2.data(), n);
    }
    Long64_t bad = 0;
    for (Long64_t i = 0; i < n; ++i) {
      if (!isValid(keys[i])) {
        ++bad;
        continue;
      }
      mStore->add(uint64_t(keys[i]), vals[i], sumw2.empty() ? 0. : sumw2[i]);
    }
    if (bad) {
      Error("Streamer", "%s: %lld bin indices out of range were dropped", GetName(), bad);
    }
    b.CheckByteCount(start, count, IsA());
  } else {
    UInt_t count = b.WriteVersion(IsA(), kTRUE);
    THnBase::Streamer(b);
    b << mBlockBits;
    Long64_t n = GetNbins();
    std::vector<Long64_t> keys;
    std::vector<T> vals;
    std::vector<Double_t> sumw2;
    keys.reserve(n);
    vals.reserve(n);
    if (GetCalculateErrors()) {
      sumw2.reserve(n);
    }
    if (mStore) {
      mStore->forEach([&](uint64_t k, T v, double s) {
        keys.push_back(Long64_t(k));
        vals.push_back(v);
        if (GetCalculateErrors()) {
          sumw2.push_back(s);
        }
      });
    }
    b << n;
    writeArray(b, keys.data(), n);
    writeArray(b, vals.data(), n);
    if (GetCalculateErrors()) {
      writeArray(b, sumw2.data(), n);
    }
    b.SetByteCount(count, kTRUE);
  }
}

#endif // O2_FRAMEWORK_COMPACTTHNSPARSE_H_
