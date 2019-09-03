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
  TriviallyCopyable() = default;
  TriviallyCopyable(unsigned x, unsigned y, unsigned secret)
    : mX(x), mY(y), mSecret(secret)
  {
  }

  bool operator==(const TriviallyCopyable& rhs) const
  {
    return mX == rhs.mX && mY == rhs.mY && mSecret == rhs.mSecret;
  }

  unsigned mX = 0;
  unsigned mY = 0;

 private:
  unsigned mSecret = ~((decltype(mSecret))0);

  ClassDefNV(TriviallyCopyable, 1);
};

class Base
{
 public:
  Base() = default;
  virtual ~Base() = default;
  virtual void f() {}

 private:
  int mMember = 0;

  ClassDef(Base, 1);
};

class Polymorphic : public Base
{
 public:
  Polymorphic() = default;
  Polymorphic(unsigned secret) : mSecret(secret) {}
  ~Polymorphic() override = default;

  bool operator==(const Polymorphic& rhs) const { return mSecret == rhs.mSecret; }

  bool isDefault() const { return mSecret == ~(decltype(mSecret))0; }

  unsigned get() const { return mSecret; }

  void f() override {}

 private:
  unsigned mSecret = ~((decltype(mSecret))0);

  ClassDefOverride(Polymorphic, 1);
};

} // namespace test
} // namespace o2
#endif // FRAMEWORK_TEST_CLASSES_H
