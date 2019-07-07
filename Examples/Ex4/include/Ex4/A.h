#ifndef _EX4_A_H_
#define _EX4_A_H_

#include "TObject.h"

namespace ex4
{
class A : public TObject
{
 public:
  A();

  int value() const;

  ClassDef(A, 1);
};

} // namespace ex4
#endif
