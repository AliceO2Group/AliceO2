/// \file Digi.h
/// \brief Definition of the ITSMFT digit
#ifndef ALICEO2_ITSMFT_DIGIT_H
#define ALICEO2_ITSMFT_DIGIT_H

#ifndef __CLING__

#include <boost/serialization/base_object.hpp> // for base_object

#endif

#include "FairTimeStamp.h" // for FairTimeStamp
#include "Rtypes.h"        // for Double_t, ULong_t, etc

namespace boost
{
namespace serialization
{
class access;
}
}

namespace o2
{
namespace ITSMFT
{
/// \class Digit
/// \brief Digit class for the ITS
///
class Digit : public FairTimeStamp
{
 public:
  /// Default constructor
  Digit();

  /// Constructor, initializing values for position, charge and time
  /// @param chipindex Global index of the pixel chip
  /// @param pixelindex Index of the pixel within the chip
  /// @param charge Accumulated charge of digit
  /// @param timestamp Time at which the digit was created
  Digit(UShort_t chipindex, UShort_t row, UShort_t col, Double_t charge, Double_t timestamp);

  /// Destructor
  ~Digit() override;

  /// Addition operator
  /// Adds the charge of 2 digits
  /// @param lhs First digit for addition (also serves as time stamp)
  /// @param rhs Second digit for addition
  /// @return Digit with the summed charge of two digits and the time stamp of the first one
  const Digit operator+(const Digit& other);

  /// Operator +=
  /// Adds charge in other digit to this digit
  /// @param other Digit added to this digit
  /// @return Digit after addition
  Digit& operator+=(const Digit& other);

  /// Get the index of the chip
  /// @return Index of the chip
  UShort_t getChipIndex() const { return mChipIndex; }
  /// Get the column of the pixel within the chip
  /// @return column of the pixel within the chip
  UShort_t getColumn() const { return mCol; }
  /// Get the row of the pixel within the chip
  /// @return row of the pixel within the chip
  UShort_t getRow() const { return mRow; }
  /// Get the accumulated charged of the digit
  /// @return charge of the digit
  Double_t getCharge() const { return mCharge; }
  /// Get the labels connected to this digit
  Int_t getLabel(Int_t idx) const { return mLabels[idx]; }
  /// Add Label to the list of Monte-Carlo labels
  void setLabel(Int_t idx, Int_t label) { mLabels[idx] = label; }
  /// Set the index of the chip
  /// @param index The chip index
  void setChipIndex(UShort_t index) { mChipIndex = index; }
  /// Set the index of the pixel within the chip
  /// @param index Index of the pixel within the chip
  void setPixelIndex(UShort_t row, UShort_t col)
  {
    mRow = row;
    mCol = col;
  }

  /// Set the charge of the digit
  /// @param charge The charge of the the digit
  void setCharge(Double_t charge) { mCharge = charge; }
  /// Check whether digit is equal to other digit.
  /// Comparison is done based on the chip index and pixel index
  /// @param other The digit to compare with
  /// @return True if digits are equal, false otherwise
  bool equal(FairTimeStamp* other) override
  {
    Digit* mydigi = dynamic_cast<Digit*>(other);
    if (mydigi) {
      if (mChipIndex == mydigi->getChipIndex() && mCol == mydigi->getColumn() && mRow == mydigi->getRow()) {
        return true;
      }
    }
    return false;
  }

  /// Test if the current digit is lower than the other
  /// Comparison is done based on the chip index and pixel index. Two
  /// options are possible for true:
  /// -# Chip index of this digit smaller than chip index of other digit
  /// -# Chip indices are equal, but pixel index of this chip is lower
  /// @param other The digit to compare with
  /// @return True if this digit has a lower total index, false otherwise
  virtual bool operator<(const Digit& other) const
  {
    /* if (mChipIndex < other.mChipIndex || */
    /*     (mChipIndex == other.mChipIndex && mCol < other.mCol)) { */
    /*       return true; */
    /* } */
    return false;
  }

  /// Print function: Print basic digit information on the  output stream
  /// @param output Stream to put the digit on
  /// @return The output stream
  std::ostream& print(std::ostream& output) const;

  /// Streaming operator for the digit class
  /// Using streaming functionality defined in function Print
  /// @param output The stream where the digit is written to
  /// @param digi The digit to put on the stream
  /// @return The output stream
  friend std::ostream& operator<<(std::ostream& output, const Digit& digi)
  {
    digi.print(output);
    return output;
  }

  /// Serialization method of the Digit using boost serialization
  /// @param ar Archive where digit is appended
  /// @param version Unused
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar& boost::serialization::base_object<FairTimeStamp>(*this);
    ar& mChipIndex;
    ar& mRow;
    ar& mCol;
    ar& mCharge;
    ar& mLabels;
  }

 private:
#ifndef __CLING__

  friend class boost::serialization::access;

#endif
  UShort_t mChipIndex; ///< Chip index
  UShort_t mRow;       ///< Pixel index in X
  UShort_t mCol;       ///< Pixel index in Z
  Double_t mCharge;    ///< Accumulated charge
  Int_t mLabels[3];    ///< Particle labels associated to this digit

  ClassDefOverride(Digit, 2);
};
}
}

#endif /* ALICEO2_ITSMFT_DIGIT_H */
