// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PadOriginal.h
/// \brief Definition of the pad used by the original cluster finder algorithm
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_PADORIGINAL_H_
#define ALICEO2_MCH_PADORIGINAL_H_

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <TMath.h>

namespace o2
{
namespace mch
{

/// pad for internal use
class PadOriginal
{
 public:
  enum padStatus {
    kZero = 0x0,       ///< pad "basic" state
    kUseForFit = 0x1,  ///< should be used for fit
    kCoupled = 0x10,   ///< coupled to another cluster of pixels
    kOver = 0x100,     ///< processing is over
    kMustKeep = 0x1000 ///< do not kill (for pixels)
  };

  PadOriginal() = delete;
  PadOriginal(double x, double y, double dx, double dy, double charge,
              bool isSaturated = false, int plane = -1, int digitIdx = -1, int status = kZero)
    : mx(x), my(y), mdx(dx), mdy(dy), mCharge(charge), mIsSaturated(isSaturated), mPlane(plane), mDigitIdx(digitIdx), mStatus(status) {}
  ~PadOriginal() = default;

  PadOriginal(const PadOriginal& cl) = default;
  PadOriginal& operator=(const PadOriginal& cl) = default;
  PadOriginal(PadOriginal&&) = default;
  PadOriginal& operator=(PadOriginal&&) = default;

  /// set position in x (cm)
  void setx(double x) { mx = x; }
  /// set position in y (cm)
  void sety(double y) { my = y; }
  /// return position in x (cm)
  double x() const { return mx; }
  /// return position in y (cm)
  double y() const { return my; }
  /// return position in x or y (cm)
  double xy(int ixy) const { return (ixy == 0) ? mx : my; }

  /// set half dimension in x (cm)
  void setdx(double dx) { mdx = dx; }
  /// set half dimension in y (cm)
  void setdy(double dy) { mdy = dy; }
  /// return half dimension in x (cm)
  double dx() const { return mdx; }
  /// return half dimension in y (cm)
  double dy() const { return mdy; }
  /// return half dimension in x or y (cm)
  double dxy(int ixy) const { return (ixy == 0) ? mdx : mdy; }

  /// set the charge
  void setCharge(double charge) { mCharge = charge; }
  /// return the charge
  double charge() const { return mCharge; }

  /// return 0 if bending pad, 1 if non-bending pad or -1 if none (e.g. pixel)
  int plane() const { return mPlane; }

  /// return the index of the corresponding digit
  int digitIndex() const { return mDigitIdx; }
  /// return whether this is a real pad or a virtual pad
  bool isReal() const { return mDigitIdx >= 0; }

  /// set the status word
  void setStatus(int status) { mStatus = status; }
  /// return the status word
  int status() const { return mStatus; }

  /// return whether this pad is saturated or not
  bool isSaturated() const { return mIsSaturated; }

 private:
  double mx = 0.;            ///< position in x (cm)
  double my = 0.;            ///< position in y (cm)
  double mdx = 0.;           ///< half dimension in x (cm)
  double mdy = 0.;           ///< half dimension in y (cm)
  double mCharge = 0.;       ///< pad charge
  bool mIsSaturated = false; ///< whether this pad is saturated or not
  int mPlane = -1;           ///< 0 = bending; 1 = non-bending, -1 = none
  int mDigitIdx = -1;        ///< index of the corresponding digit
  int mStatus = kZero;       ///< status word
};

//_________________________________________________________________________________________________
inline auto findPad(std::vector<PadOriginal>& pads, double x, double y, double minCharge)
{
  /// find a pad with a position (x,y) and a charge >= minCharge in the array of pads
  /// return an iterator to the pad or throw an exception if no pad is found

  auto match = [x, y, minCharge](const PadOriginal& pad) {
    return pad.charge() >= minCharge && TMath::Abs(pad.x() - x) < 1.e-3 && TMath::Abs(pad.y() - y) < 1.e-3;
  };

  auto itPad = std::find_if(pads.begin(), pads.end(), match);

  if (itPad == pads.end()) {
    throw std::runtime_error("Pad not found");
  }

  return itPad;
}

//_____________________________________________________________________________
inline bool areOverlapping(const PadOriginal& pad1, const PadOriginal& pad2, double precision)
{
  /// check if the two pads overlap within the given precision (cm):
  /// positive precision = decrease pad size, i.e. pads touching each other do not overlap
  /// negative precision = increase pad size, i.e. pads touching each other, including corners, overlap

  if (pad1.x() - pad1.dx() > pad2.x() + pad2.dx() - precision ||
      pad1.x() + pad1.dx() < pad2.x() - pad2.dx() + precision ||
      pad1.y() - pad1.dy() > pad2.y() + pad2.dy() - precision ||
      pad1.y() + pad1.dy() < pad2.y() - pad2.dy() + precision) {
    return false;
  }

  return true;
}

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PADORIGINAL_H_
