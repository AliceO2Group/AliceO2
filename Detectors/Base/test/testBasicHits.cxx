#define BOOST_TEST_MODULE Test BasicHits class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsBase/BaseHits.h"
#include "Math/GenVector/Transform3D.h"

namespace AliceO2
{
namespace Base
{
BOOST_AUTO_TEST_CASE(BasicXYZHit)
{
  using HitType = BasicXYZEHit<float>;
  HitType hit(1., 2., 3., 0.01, -1.1, -1, 1);

  BOOST_CHECK_CLOSE(hit.GetX(), 1., 1E-4);
  BOOST_CHECK_CLOSE(hit.GetY(), 2., 1E-4);
  BOOST_CHECK_CLOSE(hit.GetZ(), 3., 1E-4);
  BOOST_CHECK_CLOSE(hit.GetEnergyLoss(), -1.1, 1E-4);
  BOOST_CHECK_CLOSE(hit.GetTime(), 0.01, 1E-4);

  hit.SetX(0.);
  BOOST_CHECK_CLOSE(hit.GetX(), 0., 1E-4);

  // check coordinate transformation of the hit coordinate
  // check that is works with float + double
  // note that ROOT transformations are always double valued
  using ROOT::Math::Transform3D;
  Transform3D idtransf; // defaults to identity transformation

  auto transformed = idtransf(hit.GetPos());
  BOOST_CHECK_CLOSE(transformed.Y(), hit.GetY(), 1E-4);  
}

} // end namespace Base
} // end namespace AliceO2
