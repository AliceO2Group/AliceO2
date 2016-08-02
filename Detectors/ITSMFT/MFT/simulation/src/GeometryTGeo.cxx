/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author bogdan.vulpescu@cern.ch - 01/08/2016

#include "GeometryTGeo.h"
#include "Constants.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::GeometryTGeo)

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo() :
mNofDisks(0)
{
  // default constructor

  Build();

}

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo(const GeometryTGeo& src)
  : TObject(src),
    mNofDisks(src.mNofDisks)
{
  // copy constructor

}

//_____________________________________________________________________________
GeometryTGeo::~GeometryTGeo()
{
  // destructor

}

//_____________________________________________________________________________
GeometryTGeo& GeometryTGeo::operator=(const GeometryTGeo& src)
{
  // copy operator

  if (this != &src) {
    mNofDisks = src.mNofDisks;
  }

  return *this;

}

//_____________________________________________________________________________
void GeometryTGeo::Build()
{

  mNofDisks = AliceO2::MFT::Constants::sNofDisks;

}

