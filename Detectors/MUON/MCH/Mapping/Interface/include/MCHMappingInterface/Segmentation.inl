// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

namespace o2
{
namespace mch
{
namespace mapping
{

inline Segmentation::Segmentation(int deid) : mDetElemId{deid}, mBending{CathodeSegmentation(deid, true)}, mNonBending{CathodeSegmentation(deid, false)}, mPadIndexOffset{mBending.nofPads()}
{
}

inline bool Segmentation::operator==(const Segmentation& rhs) const
{
  return mDetElemId == rhs.mDetElemId;
}

inline bool Segmentation::operator!=(const Segmentation& rhs) const { return !(rhs == *this); }

inline void swap(Segmentation& a, Segmentation& b)
{
  using std::swap;
  swap(a.mDetElemId, b.mDetElemId);
  swap(a.mPadIndexOffset, b.mPadIndexOffset);
}

inline Segmentation::Segmentation(const Segmentation& seg) = default;

inline Segmentation::Segmentation(const Segmentation&& seg) : mBending{std::move(seg.mBending)}, mNonBending{std::move(seg.mNonBending)}, mDetElemId{seg.mDetElemId}, mPadIndexOffset{seg.mPadIndexOffset}
{
}

inline Segmentation& Segmentation::operator=(Segmentation seg)
{
  swap(*this, seg);
  return *this;
}

inline int Segmentation::findPadByFEE(int dualSampaId, int dualSampaChannel) const
{
  bool isBending = dualSampaId < 1024;
  int catPadIndex;
  if (isBending) {
    catPadIndex = mBending.findPadByFEE(dualSampaId, dualSampaChannel);
    if (!mBending.isValid(catPadIndex)) {
      return -1;
    }
  } else {
    catPadIndex = mNonBending.findPadByFEE(dualSampaId, dualSampaChannel);
    if (!mNonBending.isValid(catPadIndex)) {
      return -1;
    }
  }
  return padC2DE(catPadIndex, isBending);
}

inline bool Segmentation::isValid(int dePadIndex) const
{
  return dePadIndex >= 0 && dePadIndex < nofPads();
}

inline int Segmentation::padC2DE(int catPadIndex, bool isBending) const
{
  if (isBending) {
    return catPadIndex;
  }
  return catPadIndex + mPadIndexOffset;
}

inline void Segmentation::catSegPad(int dePadIndex, const CathodeSegmentation*& catseg, int& padcuid) const
{
  if (!isValid(dePadIndex)) {
    catseg = nullptr;
    return;
  }
  if (dePadIndex < mPadIndexOffset) {
    catseg = &mBending;
    padcuid = dePadIndex;
    return;
  }
  catseg = &mNonBending;
  padcuid = dePadIndex - mPadIndexOffset;
}

inline bool Segmentation::findPadPairByPosition(double x, double y, int& b, int& nb) const
{
  b = mBending.findPadByPosition(x, y);
  nb = mNonBending.findPadByPosition(x, y);
  if (!mBending.isValid(b)) {
    if (mNonBending.isValid(nb)) {
      nb = padC2DE(nb, false);
    }
    return false;
  }
  if (!mNonBending.isValid(nb)) {
    b = padC2DE(b, true);
    return false;
  }
  b = padC2DE(b, true);
  nb = padC2DE(nb, false);
  return true;
}

template <typename CALLABLE>
void Segmentation::forEachPad(CALLABLE&& func) const
{
  mBending.forEachPad(func);
  int offset{mPadIndexOffset};
  mNonBending.forEachPad([&offset, &func](int catPadIndex) {
    func(catPadIndex + offset);
  });
}

template <typename CALLABLE>
void Segmentation::forEachPadInArea(double xmin, double ymin, double xmax, double ymax, CALLABLE&& func) const
{
  mBending.forEachPadInArea(xmin, ymin, xmax, ymax, func);
  int offset{mPadIndexOffset};
  mNonBending.forEachPadInArea(xmin, ymin, xmax, ymax, [&offset, &func](int catPadIndex) {
    func(catPadIndex + offset);
  });
}

template <typename CALLABLE>
void Segmentation::forEachNeighbouringPad(int dePadIndex, CALLABLE&& func) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);

  int offset{0};
  if (!isBendingPad(dePadIndex)) {
    offset = mPadIndexOffset;
  }
  catSeg->forEachNeighbouringPad(catPadIndex, [&offset, &func](int cindex) {
    func(cindex + offset);
  });
}

inline void Segmentation::forEachDualSampa(std::function<void(int dualSampaId)> func) const
{
  mBending.forEachDualSampa(func);
  mNonBending.forEachDualSampa(func);
}

inline std::string Segmentation::padAsString(int dePadIndex) const
{
  if (!isValid(dePadIndex)) {
    return "invalid pad with index=" + std::to_string(dePadIndex);
  }
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  auto s = catSeg->padAsString(catPadIndex);
  if (isBendingPad(dePadIndex)) {
    s += " (B)";
  } else {
    s += " (NB)";
  }
  return s;
}

inline int Segmentation::padDualSampaId(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padDualSampaId(catPadIndex);
}

inline int Segmentation::padDualSampaChannel(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padDualSampaChannel(catPadIndex);
}

inline double Segmentation::padPositionX(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padPositionX(catPadIndex);
}

inline double Segmentation::padPositionY(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padPositionY(catPadIndex);
}

inline double Segmentation::padSizeX(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padSizeX(catPadIndex);
}

inline double Segmentation::padSizeY(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{nullptr};
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padSizeY(catPadIndex);
}

inline int Segmentation::detElemId() const { return mDetElemId; }
inline int Segmentation::nofPads() const { return bending().nofPads() + nonBending().nofPads(); }
inline int Segmentation::nofDualSampas() const { return bending().nofDualSampas() + nonBending().nofDualSampas(); }

inline const CathodeSegmentation& Segmentation::bending() const { return mBending; }
inline const CathodeSegmentation& Segmentation::nonBending() const { return mNonBending; }

} // namespace mapping
} // namespace mch
} // namespace o2
