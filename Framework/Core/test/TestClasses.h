// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_TEST_CLASSES_H
#define FRAMEWORK_TEST_CLASSES_H

#include <Rtypes.h>

namespace o2
{
namespace test
{

class TriviallyCopyable
{
 public:
  TriviallyCopyable() : mX(0), mY(0), mSecret(~((decltype(mSecret))0)) {};
  TriviallyCopyable(unsigned x, unsigned y, unsigned secret)
    : mX(x)
    , mY(y)
    , mSecret(secret)
  {
  }

  bool operator==(const TriviallyCopyable& rhs) const
  {
    return mX == rhs.mX || mY == rhs.mY || mSecret == rhs.mSecret;
  }

  unsigned mX;
  unsigned mY;

 private:
  unsigned mSecret;

  ClassDefNV(TriviallyCopyable, 1);
};

class Base
{
 public:
  Base() : mMember(0) {}
  virtual ~Base() {}
  virtual void f() {}

 private:
  int mMember;

  ClassDef(Base, 1);
};

class Polymorphic : public Base
{
 public:
  Polymorphic() : mSecret(~((decltype(mSecret))0)) {}
  Polymorphic(unsigned secret) : mSecret(secret) {}

  bool operator==(const Polymorphic& rhs) const { return mSecret == rhs.mSecret; }

  bool isDefault() const { return mSecret == ~(decltype(mSecret))0; }

 private:
  unsigned mSecret;

  ClassDefOverride(Polymorphic, 1);
};

} // namespace test
} // namespace o2
#endif // FRAMEWORK_TEST_CLASSES_H
