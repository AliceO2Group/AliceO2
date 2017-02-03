/// \file ClusterShape.h
/// \brief Cluster shape class for the ALPIDE response simulation

#ifndef ALICEO2_ITS_CLUSTERSHAPE_H_
#define ALICEO2_ITS_CLUSTERSHAPE_H_

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to describe the cluster shape in the ITSU simulation    //
// Author: Davide Pagano                                         //
///////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <sstream>
#include <vector>

namespace AliceO2 {
  namespace ITS {

    class ClusterShape : public TObject {

    public:
      ClusterShape();
      ClusterShape(UInt_t, UInt_t);
      ClusterShape(UInt_t, UInt_t, const std::vector<UInt_t>&);
      virtual ~ClusterShape();

      // Set the number of rows
      inline void SetNRows(UInt_t Nrows) {fNrows = Nrows;}

      // Set the number of cols
      inline void SetNCols(UInt_t Ncols) {fNcols = Ncols;}

      // Add a pixel position to the shape [0, r*c[
      inline void AddShapeValue(UInt_t pos) {fShape.push_back(pos);}

      // Check whether the shape has the
      Bool_t IsValidShape();

      // Return an unique ID based on the cluster size and shape
      Long64_t GetShapeID() const;

      // Get the number of rows of the cluster
      inline UInt_t GetNRows() const {return fNrows;}

      // Get the number of cols of the cluster
      inline UInt_t GetNCols() const {return fNcols;}

      // Get the number of fired pixels of the cluster
      inline UInt_t GetNFiredPixels() const {return fShape.size();}

      // Get the position of the pixel with the specified index
      inline UInt_t GetValue(UInt_t index) const {return fShape[index];}

      // Get the shape of the cluster
      inline void GetShape(std::vector<UInt_t>& v) const {v = fShape;}

      // Check whether the cluster has the specified pixel on
      Bool_t HasElement(UInt_t) const;

      // Return a string with the positions of the fired pixels in the cluster
      inline std::string ShapeSting() const {
        return ShapeSting(fShape);
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
        for (Int_t r = -1; r < (Int_t) v.fNrows; ++r) {
          for (UInt_t c = 0; c < v.fNcols; ++c) {
            if (r == -1) {
              if (c == 0) out << "  ";
              out << c;
              if (c < v.fNcols-1) out << " ";
            } else {
              if (c == 0) out << r << " ";
              index = r*v.fNcols + c;
              if (find(begin(v.fShape), end(v.fShape), index) != end(v.fShape)) out << "X";
              else out << " ";
              if (c < v.fNcols-1) out << " ";
            }
          }
          out << std::endl;
        }
        return out;
      }

    private:
      UInt_t  fNrows;
      UInt_t  fNcols;
      std::vector<UInt_t> fShape;

      ClassDef(ClusterShape,1)
    };
  }
}
#endif
