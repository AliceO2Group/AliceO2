// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterShape.h
/// \brief Cluster shape class for the ALPIDE response simulation

#ifndef ALICEO2_ITSMFT_CLUSTERSHAPE_H_
#define ALICEO2_ITSMFT_CLUSTERSHAPE_H_

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to describe the cluster shape in the ITSU simulation    //
// Author: Davide Pagano                                         //
///////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <sstream>
#include <vector>
#include <algorithm>

namespace o2
{
namespace itsmft
{

class ClusterShape : public TObject
{

 public:
  ClusterShape();
  ClusterShape(UInt_t, UInt_t);
  ClusterShape(UInt_t, UInt_t, const std::vector<UInt_t>&);
  ~ClusterShape() override;

  // Set the number of rows
  inline void SetNRows(UInt_t Nrows) { mNrows = Nrows; }

  // Set the number of cols
  inline void SetNCols(UInt_t Ncols) { mNcols = Ncols; }

  // Add a pixel position to the shape [0, r*c[
  inline void AddShapeValue(UInt_t pos) { mShape.push_back(pos); }

  // Check whether the shape has the
  Bool_t IsValidShape();

  // Return an unique ID based on the cluster size and shape
  Long64_t GetShapeID() const;

  // Get the number of rows of the cluster
  inline UInt_t GetNRows() const { return mNrows; }

  inline void SetCenter(UInt_t r, UInt_t c)
  {
    mCenterR = r;
    mCenterC = c;
  }

  // Get the center of rows of the cluster
  inline UInt_t GetCenterR() const { return mCenterR; }

  // Get the center of cols of the cluster
  inline UInt_t GetCenterC() const { return mCenterC; }

  // Get the index of the center (0-based)
  inline UInt_t GetCenterIndex() const
  {
    return RowColToIndex(mCenterR, mCenterC);
  }

  // Get the number of cols of the cluster
  inline UInt_t GetNCols() const { return mNcols; }

  // Get the number of fired pixels of the cluster
  inline UInt_t GetNFiredPixels() const { return mShape.size(); }

  // Get the position of the pixel with the specified index
  inline UInt_t GetValue(UInt_t index) const { return mShape[index]; }

  // Get the shape of the cluster
  inline void GetShape(std::vector<UInt_t>& v) const { v = mShape; }

  // Check whether the cluster has the specified pixel on
  Bool_t HasElement(UInt_t) const;

  // Return a string with the positions of the fired pixels in the cluster
  inline std::string ShapeSting() const
  {
    return ShapeSting(mShape);
  }

  // r and c are 0-based. The returned index is 0-based as well
  inline UInt_t RowColToIndex(UInt_t r, UInt_t c) const
  {
    return r * mNcols + c;
  }

  // Static function to get a string with the positions of the fired pixels
  // in the passed shape vector
  static std::string ShapeSting(const std::vector<UInt_t>& shape)
  {
    std::stringstream out;
    for (UInt_t i = 0; i < shape.size(); ++i) {
      out << shape[i];
      if (i < shape.size() - 1)
        out << " ";
    }
    return out.str();
  }

  friend std::ostream& operator<<(std::ostream& out, const ClusterShape& v)
  {
    UInt_t index = 0;
    for (Int_t r = -1; r < (Int_t)v.mNrows; ++r) {
      for (UInt_t c = 0; c < v.mNcols; ++c) {
        if (r == -1) {
          if (c == 0)
            out << "  ";
          out << c;
          if (c < v.mNcols - 1)
            out << " ";
        } else {
          if (c == 0)
            out << r << " ";
          index = r * v.mNcols + c;
          if (std::find(begin(v.mShape), end(v.mShape), index) != end(v.mShape))
            out << "X";
          else
            out << " ";
          if (c < v.mNcols - 1)
            out << " ";
        }
      }
      out << std::endl;
    }
    return out;
  }

 private:
  UInt_t ComputeCenter(UInt_t);

  UInt_t mNrows;
  UInt_t mNcols;
  UInt_t mCenterR;
  UInt_t mCenterC;
  std::vector<UInt_t> mShape;

  ClassDefOverride(ClusterShape, 1);
};
} // namespace itsmft
} // namespace o2
#endif
