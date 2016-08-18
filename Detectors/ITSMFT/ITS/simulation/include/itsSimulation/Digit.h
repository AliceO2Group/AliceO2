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
    Digit(Int_t chipindex, Double_t pixelindex, Double_t charge, Double_t timestamp);

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
    ULong_t GetChipIndex() const
    { return fChipIndex; }

    /// Get the index of the pixel within the chip
    /// @return index of the pixel within the chip
    ULong_t GetPixelIndex() const
    { return fPixelIndex; }

    /// Get the accumulated charged of the digit
    /// @return charge of the digit
    Double_t GetCharge() const
    { return fCharge; }

    /// Get the labels connected to this digit
    /// @return vector of track labels
    const std::vector<int> &GetListOfLabels() const
    { return fLabels; }

    /// Add Label to the list of Monte-Carlo labels
    /// @TODO: be confirmed how this is handled
    void AddLabel(Int_t label)
    { fLabels.push_back(label); }

    /// Set the index of the chip
    /// @param index The chip index
    void SetChipIndex(Int_t index)
    { fChipIndex = index; }

    /// Set the index of the pixel within the chip
    /// @param index Index of the pixel within the chip
    void SetPixelIndex(Int_t index)
    { fPixelIndex = index; }

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
        if (fChipIndex == mydigi->GetChipIndex() && fPixelIndex == mydigi->GetPixelIndex()) { return true; }
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
      if (fChipIndex < other.fChipIndex ||
          (fChipIndex == other.fChipIndex && fPixelIndex < other.fPixelIndex)) {
            return true;
      }
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
      ar & fPixelIndex;
      ar & fCharge;
      ar & fLabels;
    }

  private:
#ifndef __CLING__

    friend class boost::serialization::access;

#endif
    ULong_t fChipIndex;         ///< Chip index
    ULong_t fPixelIndex;        ///< Index of the pixel within the chip
    Double_t fCharge;            ///< Accumulated charge
    std::vector<int> fLabels;            ///< Particle labels associated to this digit (@TODO be confirmed)

  ClassDef(Digit, 1);
};
}
}

#endif /* ALICEO2_ITS_AliITSUpgradeDigi_H */
