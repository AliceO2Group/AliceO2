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

namespace o2 {
  namespace ITSMFT {

    class ClusterShape : public TObject {

    public:
      ClusterShape();
      ClusterShape(UInt_t, UInt_t);
      ClusterShape(UInt_t, UInt_t, const std::vector<UInt_t>&);
      ~ClusterShape() override;

      // Set the number of rows
      inline void SetNRows(UInt_t Nrows) {mNrows = Nrows;}

      // Set the number of cols
      inline void SetNCols(UInt_t Ncols) {mNcols = Ncols;}

      // Add a pixel position to the shape [0, r*c[
      inline void AddShapeValue(UInt_t pos) {mShape.push_back(pos);}

      // Check whether the shape has the
      Bool_t IsValidShape();

      // Return an unique ID based on the cluster size and shape
      Long64_t GetShapeID() const;

      // Get the number of rows of the cluster
      inline UInt_t GetNRows() const {return mNrows;}

      // Get the number of cols of the cluster
      inline UInt_t GetNCols() const {return mNcols;}

      // Get the number of fired pixels of the cluster
      inline UInt_t GetNFiredPixels() const {return mShape.size();}

      // Get the position of the pixel with the specified index
      inline UInt_t GetValue(UInt_t index) const {return mShape[index];}

      // Get the shape of the cluster
      inline void GetShape(std::vector<UInt_t>& v) const {v = mShape;}

      // Check whether the cluster has the specified pixel on
      Bool_t HasElement(UInt_t) const;

      // Return a string with the positions of the fired pixels in the cluster
      inline std::string ShapeSting() const {
        return ShapeSting(mShape);
      }

      // Static function to get a string with the positions of the fired pixels
      // in the passed shape vector
      static std::string ShapeSting(const std::vector<UInt_t>& shape) {
        std::stringstream out;
        for (UInt_t i = 0; i < shape.size(); ++i) {
          out << shape[i];
          if (i < shape.size()-1) out << " ";
        }
        return out.str();
      }

      friend std::ostream &operator<<(std::ostream &out, const ClusterShape &v) {
        UInt_t index = 0;
        for (Int_t r = -1; r < (Int_t) v.mNrows; ++r) {
          for (UInt_t c = 0; c < v.mNcols; ++c) {
            if (r == -1) {
              if (c == 0) out << "  ";
              out << c;
              if (c < v.mNcols-1) out << " ";
            } else {
              if (c == 0) out << r << " ";
              index = r*v.mNcols + c;
              if (std::find(begin(v.mShape), end(v.mShape), index) != end(v.mShape)) out << "X";
              else out << " ";
              if (c < v.mNcols-1) out << " ";
            }
          }
          out << std::endl;
        }
        return out;
      }

    private:
      UInt_t  mNrows;
      UInt_t  mNcols;
      std::vector<UInt_t> mShape;

      ClassDefOverride(ClusterShape,1)
    };
  }
}
#endif
