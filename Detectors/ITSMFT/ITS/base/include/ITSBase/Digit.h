/// \file AliITSUpgradeDigi.h
/// \brief Digits structure for upgrade ITS
#ifndef ALICEO2_ITS_DIGIT_H
#define ALICEO2_ITS_DIGIT_H

#ifndef __CLING__

#include <boost/serialization/base_object.hpp>  // for base_object

#endif

#include "FairTimeStamp.h"                      // for FairTimeStamp
#include "Rtypes.h"                             // for Double_t, ULong_t, etc

namespace boost { namespace serialization { class access; }}

namespace AliceO2 {
namespace ITS {

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
    virtual ~Digit();

    /// Addition operator
    /// Adds the charge of 2 digits
    /// @param lhs First digit for addition (also serves as time stamp)
    /// @param rhs Second digit for addition
    /// @return Digit with the summed charge of two digits and the time stamp of the first one
    const Digit operator+(const Digit &other);

    /// Operator +=
    /// Adds charge in other digit to this digit
    /// @param other Digit added to this digit
    /// @return Digit after addition
    Digit &operator+=(const Digit &other);

    /// Get the index of the chip
    /// @return Index of the chip
    UShort_t GetChipIndex() const
    { return fChipIndex; }

    /// Get the column of the pixel within the chip
    /// @return column of the pixel within the chip
    UShort_t GetColumn() const
    { return fCol; }

    /// Get the row of the pixel within the chip
    /// @return row of the pixel within the chip
    UShort_t GetRow() const
    { return fRow; }

    /// Get the accumulated charged of the digit
    /// @return charge of the digit
    Double_t GetCharge() const
    { return fCharge; }

    /// Get the labels connected to this digit
    Int_t GetLabel(Int_t idx) const
    { return fLabels[idx]; }

    /// Add Label to the list of Monte-Carlo labels
    void SetLabel(Int_t idx, Int_t label)
    { fLabels[idx]=label; }

    /// Set the index of the chip
    /// @param index The chip index
    void SetChipIndex(UShort_t index)
    { fChipIndex = index; }

    /// Set the index of the pixel within the chip
    /// @param index Index of the pixel within the chip
    void SetPixelIndex(UShort_t row, UShort_t col)
    { fRow = row; fCol = col; }

    /// Set the charge of the digit
    /// @param charge The charge of the the digit
    void SetCharge(Double_t charge)
    { fCharge = charge; }

    /// Check whether digit is equal to other digit.
    /// Comparison is done based on the chip index and pixel index
    /// @param other The digit to compare with
    /// @return True if digits are equal, false otherwise
    virtual bool equal(FairTimeStamp *other)
    {
      Digit *mydigi = dynamic_cast<Digit *>(other);
      if (mydigi) {
        if (fChipIndex == mydigi->GetChipIndex() &&
	    fCol == mydigi->GetColumn() &&
	    fRow == mydigi->GetRow()) { return true; }
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
    virtual bool operator<(const Digit &other) const
    {
      /* if (fChipIndex < other.fChipIndex || */
      /*     (fChipIndex == other.fChipIndex && fCol < other.fCol)) { */
      /*       return true; */
      /* } */
      return false;
    }

    /// Print function: Print basic digit information on the  output stream
    /// @param output Stream to put the digit on
    /// @return The output stream
    std::ostream &Print(std::ostream &output) const;

    /// Streaming operator for the digit class
    /// Using streaming functionality defined in function Print
    /// @param output The stream where the digit is written to
    /// @param digi The digit to put on the stream
    /// @return The output stream
    friend std::ostream &operator<<(std::ostream &output, const Digit &digi)
    {
      digi.Print(output);
      return output;
    }

    /// Serialization method of the Digit using boost serialization
    /// @param ar Archive where digit is appended
    /// @param version Unused
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar & boost::serialization::base_object<FairTimeStamp>(*this);
      ar & fChipIndex;
      ar & fRow;
      ar & fCol;
      ar & fCharge;
      ar & fLabels;
    }

  private:
#ifndef __CLING__

    friend class boost::serialization::access;

#endif
    UShort_t fChipIndex;         ///< Chip index
    UShort_t fRow;               ///< Pixel index in X
    UShort_t fCol;               ///< Pixel index in Z
    Double_t fCharge;            ///< Accumulated charge
    Int_t fLabels[3];            ///< Particle labels associated to this digit

  ClassDef(Digit, 2);
};
}
}

#endif /* ALICEO2_ITS_AliITSUpgradeDigi_H */
