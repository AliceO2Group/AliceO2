/// \file AliITSUpgradeDigi.h
/// \brief Digits structure for upgrade ITS
#ifndef ALICEO2_ITS_DIGIT_H
#define ALICEO2_ITS_DIGIT_H

#ifndef __CINT__
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#endif

#include "Riosfwd.h"
#include "RTypes.h"

#include "FairTimeStamp.h"

namespace AliceO2{
    namespace ITS{
        
        class Digit : public FairTimeStamp{
        public:
            Digit();
            Digit(Int_t index, Double_t charge, Double_t timestamp);
            virtual ~Digit();
            
            ULong_t GetChipIndex() const { return fIndex; }
            Double_t GetCharge() const { return fCharge; }
            const std::vector<int> &GetListOfLabels() const { return fLabels; }
            
            /// Add Label to the list of Monte-Carlo labels
            /// @TODO: be confirmed how this is handled
            void AddLabel(Int_t label) { fLabels.push_back(label); }
            
            /// Set the index of the chip
            /// \param index The chip index
            void SetChipIndex(Int_t index) { fIndex = index; }
            
            /// Set the charge of the digit
            /// \param charge The charge of the the digit
            void SetCharge(Double_t charge) { fCharge = charge; }
            
            /// \brief Test for equalness with other digit
            ///
            /// Comparison is done based on the chip index
            /// \param other The digit to compare with
            /// \return True if digits are equal, false otherwise
            virtual bool equal(FairTimeStamp *other){
                Digit *mydigi = dynamic_cast<Digit *>(other);
                if(mydigi){
                    if(fIndex == mydigi->GetChipIndex()) return true;
                }
                return false;
            }
            
            /// \brief Test if the current digit is lower than the other
            ///
            /// Comparison is done based on the chip index
            /// \param other The digit to compare with
            /// \return True if this digit has a lower chip index, false otherwise
            virtual bool operator<(const Digit &other){
                if(fIndex < other.fIndex) return true;
                return false;
            }
            
            friend std::ostream &operator<<(std::ostream &out, Digit &digi){
                out << "ITS Digit of chip index X[" << digi.GetChipIndex() << "] with charge " << digi.GetCharge() << " at time stamp" << digi.GetTimeStamp();
                return out;
            }
            
            template<class Archive>
            void serialize(Archive &ar, const unsigned int version){
                ar & boost::serialization::base_object<FairTimeStamp>(*this);
                ar & fIndex;
                ar & fCharge;
                ar & fLabels;
            }
            
        private:
#ifndef __CINT__
            friend class boost::serialization::access;
#endif
            ULong_t                 fIndex;             ///< Chip index
            Double_t                fCharge;            ///< Accumulated charge
            std::vector<int>        fLabels;            ///< Particle labels associated to this digit (@TODO be confirmed)
            
            ClassDef(Digit, 1);
        };
    }
}

#endif /* ALICEO2_ITS_AliITSUpgradeDigi_H */
