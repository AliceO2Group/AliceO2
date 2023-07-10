// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SIMDATA_MCGENPROPERTIES_H_
#define ALICEO2_SIMDATA_MCGENPROPERTIES_H_

namespace o2
{
namespace mcgenstatus
{

// Value to check MCGenStatusEncoding::isEncoded against to decide whether or not the stored value is encoded or basically only the HepMC status code
// as it used to be
constexpr unsigned int isEncodedValue{5};

// internal structure to allow convenient manipulation of properties as bits on an int to (dis)entangle HepMC and specific generator status codes
union MCGenStatusEncoding {
  MCGenStatusEncoding() : fullEncoding(0) {}
  MCGenStatusEncoding(int enc) : fullEncoding(enc) {}
  // To be backward-compatible, only set transport to 1 if hepmc status is 1
  MCGenStatusEncoding(int hepmcIn, int genIn) : isEncoded(isEncodedValue), hepmc(hepmcIn), gen(genIn), reserved(0) {}
  int fullEncoding;
  struct {
    int hepmc : 9;              // HepMC status code
    int gen : 10;               // specific generator status code
    int reserved : 10;          // reserved bits for future usage
    unsigned int isEncoded : 3; // special bits to check whether or not the fullEncoding is a combination of HepMC and gen status codes
  };
};

inline bool isEncoded(MCGenStatusEncoding statusCode)
{
  return (statusCode.isEncoded == isEncodedValue);
}

inline bool isEncoded(int statusCode)
{
  return isEncoded(MCGenStatusEncoding(statusCode));
}

inline int getHepMCStatusCode(MCGenStatusEncoding enc)
{
  if (!isEncoded(enc)) {
    // in this case simply set hepmc code to what was given
    return enc.fullEncoding;
  }
  return enc.hepmc;
}

inline int getGenStatusCode(MCGenStatusEncoding enc)
{
  if (!isEncoded(enc)) {
    // in this case simply set hepmc code to what was given
    return enc.fullEncoding;
  }
  return enc.gen;
}

inline int getHepMCStatusCode(int encoded)
{
  return getHepMCStatusCode(MCGenStatusEncoding(encoded));
}

inline int getGenStatusCode(int encoded)
{
  return getGenStatusCode(MCGenStatusEncoding(encoded));
}

} // namespace mcgenstatus

namespace mcgenid
{

// Define some common properties that can be set for Generators
class GeneratorProperty
{
 public:
  typedef const char* Property;
  static constexpr Property GENERATORID{"generator_id"};
  static constexpr Property GENERATORDESCRIPTION{"generator_description"};
  static constexpr Property COCKTAILID{"cocktail_id"};
  static constexpr Property COCKTAILDESCRIPTIONMAP{"gcocktail_description_map"};
};

// internal structure to allow convenient manipulation of properties as bits on an int to (dis)entangle HepMC and specific generator status codes
union MCGenIdEncoding {
  MCGenIdEncoding() : fullEncoding(0) {}
  MCGenIdEncoding(int enc) : fullEncoding(enc) {}
  // To be backward-compatible, only set transport to 1 if hepmc status is 1
  MCGenIdEncoding(int generatorId, int sourceId, int cocktailId = -1) : generatorId(generatorId), sourceId(sourceId), cocktailId(cocktailId) {}
  short fullEncoding;
  struct {
    short generatorId : 5; // an additional identifier for a generator which can be set by the user
    short sourceId : 5;    // ID used in embedding scenarios
    short cocktailId : 6;  // reserved bits for future encodings, for instance to identify single cocktail constituents of a generator
  };
};

inline short getEncodedGenId(int generatorId, int sourceId, int cocktailId = -1)
{
  return MCGenIdEncoding(generatorId, sourceId, cocktailId).fullEncoding;
}

inline int getGeneratorId(short encoded)
{
  return static_cast<int>(MCGenIdEncoding(encoded).generatorId);
}

inline int getSourceId(short encoded)
{
  return static_cast<int>(MCGenIdEncoding(encoded).sourceId);
}

inline int getCocktailId(short encoded)
{
  return static_cast<int>(MCGenIdEncoding(encoded).cocktailId);
}

} // namespace mcgenid

} // namespace o2

#endif
