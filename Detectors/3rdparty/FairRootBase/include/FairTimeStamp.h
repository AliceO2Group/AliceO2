/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRTIMESTAMP_H
#define FAIRTIMESTAMP_H

#include "FairMultiLinkedData_Interface.h"   // for FairMultiLinkedData

#include <Rtypes.h>   // for Double_t, etc
#include <iosfwd>     // for ostream
#include <iostream>   // for ostream, cout

namespace boost {
namespace serialization {
class access;
}
}   // namespace boost
#include <boost/serialization/base_object.hpp>

class TObject;

/**
 * Base class for Time stamp information
 ** Aug. 2010
 **@author M.Al-Turany <m.al-turany@gsi.de>
 */
class FairTimeStamp : public FairMultiLinkedData_Interface
{
  public:
    /** Default constructor **/
    FairTimeStamp();
    /** Constructor with time **/
    FairTimeStamp(Double_t time);
    /** Constructor with time and time error **/
    FairTimeStamp(Double_t time, Double_t timeerror);

    /** Destructor **/
    virtual ~FairTimeStamp(){};
    /** Accessors **/
    Double_t GetTimeStamp() const { return fTimeStamp; }
    Double_t GetTimeStampError() const { return fTimeStampError; }
    /** Modifiers **/
    void SetTimeStamp(Double_t t) { fTimeStamp = t; }
    void SetTimeStampError(Double_t t) { fTimeStampError = t; }
    virtual Int_t Compare(const TObject* obj) const
    {
        if (this == obj) {
            return 0;
        }
        FairTimeStamp* tsobj = static_cast<FairTimeStamp*>(const_cast<TObject*>(obj));
        Double_t ts = tsobj->GetTimeStamp();
        Double_t tserror = tsobj->GetTimeStampError();
        if (fTimeStamp < ts) {
            return -1;
        } else if (fTimeStamp == ts && fTimeStampError < tserror) {
            return -1;
        } else if (fTimeStamp == ts && fTimeStampError == tserror) {
            return 0;
        } else {
            return 1;
        }
    }

    virtual std::ostream& PrintTimeInfo(std::ostream& out = std::cout) const;
    virtual Bool_t IsSortable() const { return kTRUE; };

    virtual bool equal(FairTimeStamp* data)
    {
        return (fTimeStamp == data->GetTimeStamp() && fTimeStampError == data->GetTimeStampError());
    }

    friend std::ostream& operator<<(std::ostream& out, const FairTimeStamp& link)
    {
        link.PrintTimeInfo(out);
        return out;
    }

    virtual bool operator<(const FairTimeStamp* rValue) const
    {
        if (GetTimeStamp() < rValue->GetTimeStamp())
            return true;
        else
            return false;
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // ar & boost::serialization::base_object<FairMultiLinkedData>(*this);
        ar& fTimeStamp;
        ar& fTimeStampError;
    }

  protected:
    friend class boost::serialization::access;

    Double_t fTimeStamp;      /** Time of digit or Hit  [ns] */
    Double_t fTimeStampError; /** Error on time stamp */

    ClassDef(FairTimeStamp, 4);
};

// -----   Default constructor   -------------------------------------------
inline FairTimeStamp::FairTimeStamp()
    : FairMultiLinkedData_Interface()
    , fTimeStamp(-1)
    , fTimeStampError(-1)
{}

// -----   Standard constructor   ------------------------------------------
inline FairTimeStamp::FairTimeStamp(Double_t time)
    : FairMultiLinkedData_Interface()
    , fTimeStamp(time)
    , fTimeStampError(-1)
{}

inline FairTimeStamp::FairTimeStamp(Double_t time, Double_t timeerror)
    : FairMultiLinkedData_Interface()
    , fTimeStamp(time)
    , fTimeStampError(timeerror)
{}

#endif
