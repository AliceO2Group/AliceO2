#include "ITSReconstruction/CAaux.h"
#include "DetectorsBase/Constants.h"

using namespace o2::ITS::CA;
using o2::Base::Constants::kPI;
using std::array;

Cell::Cell(int xx,int yy, int zz, int dd0, int dd1, float curv, array<float,3> n)
  : m1OverR{curv},
  md0{dd0},
  md1{dd1},
  mN{n[0],n[1],n[2]},
  mVector{xx,yy,zz,1u} {
}

bool Cell::Combine(Cell &neigh, int idd) {
  // From outside inward
  if (this->y() == neigh.z() && this->x() == neigh.y()) { // Cells sharing two points
    mVector.push_back(idd);
    if (neigh.GetLevel() + 1 > GetLevel()) {
      SetLevel(neigh.GetLevel() + 1u);
    }
    return true;
  }
  return false;
}

Track::Track(float x, float a, array<float,Base::Track::kNParams> p, array<float,Base::Track::kCovMatSize> c, int *cl) :
  mT{x,a,p,c},
  mCl{},
  mLabel{-1},
  mChi2{0.f} {
    for (int i = 0; i < 7; ++i) mCl[i] = cl[i];
  }

bool Track::Update(const Cluster &cl) {
  array<float,2> p{cl.y,cl.z};
  const float dChi2 = mT.GetPredictedChi2(p,cl.cov);
  if (!mT.Update(p,cl.cov)) return false;
  else mChi2 += dChi2;
  return true;
}

bool Track::GetPhiZat(float r, float bfield,float &phi, float &z) const {
  float rp4=mT.GetCurvature(bfield);

  float xt=mT.GetY(), yt=mT.GetY();
  float x = 0.f, y = 0.f;
  float sn=sin(mT.GetAlpha()), cs=cos(mT.GetAlpha());
  float a = x*cs + y*sn;
  y = -x*sn + y*cs; x=a;
  xt-=x; yt-=y;

  sn=rp4*xt - mT.GetSnp(); cs=rp4*yt + sqrt((1.- mT.GetSnp())*(1.+mT.GetSnp()));
  a=2*(xt*mT.GetSnp() - yt*sqrt((1.-mT.GetSnp())*(1.+mT.GetSnp())))-rp4*(xt*xt + yt*yt);
  float d =  -a/(1 + sqrt(sn*sn + cs*cs));

  if (fabs(d) > r) {
    if (r>1e-1) return false;
    r = fabs(d);
  }

  float rcurr=sqrt(mT.GetX()*mT.GetX() + mT.GetY()*mT.GetY());
  float phicurr=mT.GetPhi();

  if (mT.GetX()>=0.) {
    phi=phicurr+asin(d/r)-asin(d/rcurr);
  } else {
    phi=phicurr+asin(d/r)+asin(d/rcurr)-kPI;
  }

  //return a phi in [0,2pi
  if (phi<0.) phi+=2.*kPI;
  else if (phi>=2.*kPI) phi-=2.*kPI;
  z=mT.GetZ()+mT.GetTgl()*(sqrt((r-d)*(r+d))-sqrt((rcurr-d)*(rcurr+d)));

  return true;
}

