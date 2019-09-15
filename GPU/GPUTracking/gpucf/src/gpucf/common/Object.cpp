// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Object.h"

#include <regex>


using namespace gpucf;


nonstd::optional<Object> Object::tryParse(const std::string &str)
{
    static const std::string typeDelimiter  = ":";
    static const std::string fieldDelimiter = ",";
    static const std::string tupleDelimiter = "=";

    static const std::regex trim("\\s*(\\S*)\\s*");

    size_t typeEnd = str.find(typeDelimiter);

    if (typeEnd == std::string::npos)
    {
        return nonstd::nullopt;
    }

    std::string type = str.substr(0, typeEnd);

    if (type.empty())
    {
        return nonstd::nullopt; 
    }
    
    std::smatch sm;
    std::regex_match(type, sm, trim);

    type = sm[1];

    Object obj(type);



    size_t end = typeEnd;
    while (end != std::string::npos)
    {
        size_t pos = end+1;
        end = str.find(fieldDelimiter, pos);
        
        std::string fieldTuple = str.substr(pos, end-pos);

        size_t eqPos = fieldTuple.find(tupleDelimiter);

        if (eqPos == std::string::npos)
        {
            return nonstd::nullopt; 
        }

        std::string name = fieldTuple.substr(0, eqPos);

        std::regex_match(name, sm, trim);
        name = sm[1];

        if (name.empty())
        {
            return nonstd::nullopt;
        }

        std::string value = fieldTuple.substr(eqPos+1, std::string::npos);

        std::regex_match(value, sm, trim);
        value = sm[1];

        if (value.empty())
        {
            return nonstd::nullopt;
        }

        obj.set(name, value);
    }

    return obj;
}

Object::Object(const std::string &t)
    : type(t)
{
}

bool Object::hasKey(const std::string &key) const
{
    return fields.count(key) > 0; 
}

int Object::getInt(const std::string &key) const
{
    const std::string &value = fields.at(key);
    
    return std::stoi(value);
}

float Object::getFloat(const std::string &key) const
{
    const std::string &value = fields.at(key);

    return std::stof(value);
}

void Object::set(const std::string &key, int value)
{
    set(key, std::to_string(value));
}

void Object::set(const std::string &key, float value)
{
    set(key, std::to_string(value));
}

void Object::set(const std::string &key, const std::string &value)
{
    fields[key] = value; 
}

std::string Object::str() const
{
    std::stringstream ss;
    
    ss << type;

    bool firstIter = true;
    for (const auto &field : fields)
    {
        ss << (firstIter ? ": " : ", ") 
           << field.first << " = " << field.second;

        firstIter = false;
    }

    return ss.str();
}

std::string Object::getType() const
{
    return type;
}

std::ostream &gpucf::operator<<(std::ostream &o, const Object &obj)
{
    return o << obj.str();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
