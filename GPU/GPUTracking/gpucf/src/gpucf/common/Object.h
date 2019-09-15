// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <nonstd/optional.hpp>

#include <string>
#include <unordered_map>


namespace gpucf
{

class Object
{
public:
    static nonstd::optional<Object> tryParse(const std::string &);

    Object(const std::string &);

    bool hasKey(const std::string &) const;

    int   getInt(const std::string &) const;
    float getFloat(const std::string &) const;

    void set(const std::string &, int);
    void set(const std::string &, float);
    void set(const std::string &, const std::string &);

    std::string str() const;

    std::string getType() const;

private:
    std::string type;
    std::unordered_map<std::string, std::string> fields;
 
};

std::ostream &operator<<(std::ostream &, const Object &);
    
} // namespace gpucf


#define GET_IMPL(obj, fieldName, field, getterFunc) field = obj.getterFunc(fieldName)
#define GET_INT(obj, field) GET_IMPL(obj, #field, field, getInt)
#define GET_FLOAT(obj, field) GET_IMPL(obj, #field, field, getFloat)

#define SET_FIELD(obj, field) obj.set(#field, field)

// vim: set ts=4 sw=4 sts=4 expandtab:
