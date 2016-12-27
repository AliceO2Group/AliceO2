/// \file TrackP
/// \brief Base track model for the Barrel, params only, w/o covariance
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_TRACK
#define ALICEO2_BASE_TRACK

#include <algorithm>
#include <array>
#include <iostream>
#include <string.h>

#include <Rtypes.h>

#include "DetectorsBase/Constants.h"
#include "DetectorsBase/Utils.h"

namespace AliceO2 {
  namespace Base {
    namespace Track {

      // aliases for track elements
      enum ParLabels : int {
        kY,kZ,kSnp,kTgl,kQ2Pt
      };
      enum CovLabels : int {
        kSigY2,
        kSigZY,kSigZ2,
        kSigSnpY,kSigSnpZ,kSigSnp2,
        kSigTglY,kSigTglZ,kSigTglSnp,kSigTgl2,
        kSigQ2PtY,kSigQ2PtZ,kSigQ2PtSnp,kSigQ2PtTgl,kSigQ2Pt2
      };

      constexpr int
        kNParams=5,
        kCovMatSize=15,
        kLabCovMatSize=21;

      constexpr float
        kCY2max = 100*100, // SigmaY<=100cm
        kCZ2max = 100*100, // SigmaZ<=100cm
        kCSnp2max = 1*1,     // SigmaSin<=1
        kCTgl2max = 1*1,     // SigmaTan<=1
        kC1Pt2max = 100 * 100, // Sigma1/Pt<=100 1/GeV
        kCalcdEdxAuto = -999.f ; // value indicating request for dedx calculation

      // helper function
      float BetheBlochSolid(float bg, float rho=2.33f,float kp1=0.20f,float kp2=3.00f,
			    float meanI=173e-9f,float meanZA=0.49848f);
      void g3helx3(float qfield, float step,std::array<float,7> &vect);


      class TrackParBase { // track parameterization, kinematics only. This base class cannot be instantiated
        public:

          ///const float* GetParam()              const { return mP; }
          float GetX()                         const { return mX; }
          float GetAlpha()                     const { return mAlpha; }
          float GetY()                         const { return mP[kY]; }
          float GetZ()                         const { return mP[kZ]; }
          float GetSnp()                       const { return mP[kSnp]; }
          float GetTgl()                       const { return mP[kTgl]; }
          float GetQ2Pt()                      const { return mP[kQ2Pt]; }

          // derived getters
          float GetCurvature(float b)          const { return mP[kQ2Pt]*b*Constants::kB2C;}
          float GetSign()                      const { return mP[kQ2Pt]>0 ? 1.f:-1.f;}
          float GetPhi()                       const { return asinf(GetSnp()) + GetAlpha();}
          float GetPhiPos()                    const;

          float GetP()                         const;
          float GetPt()                        const;
          void  GetXYZ(std::array<float,3> &xyz)           const;
          bool  GetPxPyPz(std::array<float,3> &pxyz)       const;
          bool  GetPosDir(std::array<float,9> &posdirp)    const;

          // parameters manipulation
          bool  RotateParam(float alpha);
          bool  PropagateParamTo(float xk, float b);
          bool  PropagateParamTo(float xk, const std::array<float,3> &b);
          void  InvertParam();

          void  PrintParam()                   const;

        protected:
          // to keep this class non-virtual but derivable the c-tors and d-tor are protected
          TrackParBase() : mX{0.},mAlpha{0.} {}
          TrackParBase(float x,float alpha, const std::array<float,kNParams> &par);
          TrackParBase(const std::array<float,3> &xyz,const std::array<float,3> &pxpypz, int sign, bool sectorAlpha=true);
          TrackParBase(const TrackParBase&) = default;
          TrackParBase(TrackParBase&&) = default;
          TrackParBase& operator=(const TrackParBase& src) = default;
          ~TrackParBase() = default;
          //
          float mX;                   /// X of track evaluation
          float mAlpha;               /// track frame angle
          float mP[kNParams] = {0.f}; /// 5 parameters: Y,Z,sin(phi),tg(lambda),q/pT
      };

      // rootcint does not swallow final keyword here
      class TrackParCov final : public TrackParBase { // track+error parameterization
        public:
          TrackParCov() { memset(mPC,0,kTrackPCSize*sizeof(float)); }
          TrackParCov(float x,float alpha, const float par[kNParams], const float cov[kCovMatSize]);
          TrackParCov(const float xyz[3],const float pxpypz[3],const float[kLabCovMatSize],
              int sign, bool sectorAlpha=true);

	  /* I. Belikov: Dangerous casts !
          operator TrackPar*() { return reinterpret_cast<TrackPar*>(this); }
          operator TrackPar()  { return *reinterpret_cast<TrackPar*>(this); }
          operator TrackPar&() { return *reinterpret_cast<TrackPar*>(this); }
	  */

          float& operator[](int i)                   { return mPC[i]; }
          float  operator[](int i)             const { return mPC[i]; }
          operator float*()                    const { return (float*)mPC; }
          const float* GetParam()              const { return &mPC[kY]; }
          const float* GetCov()                const { return &mPC[kSigY2]; }

          float GetX()                         const { return mPC[kX]; }
          float GetAlpha()                     const { return mPC[kAlpha]; }
          float GetY()                         const { return mPC[kY]; }
          float GetZ()                         const { return mPC[kZ]; }
          float GetSnp()                       const { return mPC[kSnp]; }
          float GetTgl()                       const { return mPC[kTgl]; }
          float GetQ2Pt()                      const { return mPC[kQ2Pt]; }


          float GetSigmaY2()                   const { return mPC[kSigY2]; }
          float GetSigmaZY()                   const { return mPC[kSigZY]; }
          float GetSigmaZ2()                   const { return mPC[kSigZ2]; }
          float GetSigmaSnpY()                 const { return mPC[kSigSnpY]; }
          float GetSigmaSnpZ()                 const { return mPC[kSigSnpZ]; }
          float GetSigmaSnp2()                 const { return mPC[kSigSnp2]; }
          float GetSigmaTglY()                 const { return mPC[kSigTglY]; }
          float GetSigmaTglZ()                 const { return mPC[kSigTglZ]; }
          float GetSigmaTglSnp()               const { return mPC[kSigTglSnp]; }
          float GetSigmaTgl2()                 const { return mPC[kSigTgl2]; }
          float GetSigma1PtY()                 const { return mPC[kSigQ2PtY]; }
          float GetSigma1PtZ()                 const { return mPC[kSigQ2PtZ]; }
          float GetSigma1PtSnp()               const { return mPC[kSigQ2PtSnp]; }
          float GetSigma1PtTgl()               const { return mPC[kSigQ2PtTgl]; }
          float GetSigma1Pt2()                 const { return mPC[kSigQ2Pt2]; }

          // derived getters
          float GetCurvature(float b)          const { return mPC[kQ2Pt]*b*kB2C;}
          float GetSign()                      const { return mPC[kQ2Pt]>0 ? 1.f:-1.f;}
          float GetP()                         const { return Param()->GetP(); }
          float GetPt()                        const { return Param()->GetPt(); }
          float GetPhi()                       const { return Param()->GetPhi(); }
          float GetPhiPos()                    const { return Param()->GetPhiPos(); }
          void  GetXYZ(float xyz[3])           const { Param()->GetXYZ(xyz); }
          bool  GetPxPyPz(float pxyz[3])       const { return Param()->GetPxPyPz(pxyz); }
          bool  GetPosDir(float posdirp[9])    const { return Param()->GetPosDir(posdirp); }

          // parameters manipulation
          bool  RotateParam(float alpha)             { return Param()->RotateParam(alpha); }
          bool  PropagateParamTo(float xk, float b)  { return Param()->PropagateParamTo(xk,b); }
          bool  PropagateParamTo(float xk, const float b[3]) {return Param()->PropagateParamTo(xk,b); }
          void  InvertParam()                        { Param()->InvertParam(); }

          bool  Rotate(float alpha);
          bool  PropagateTo(float xk, float b);
          bool  PropagateTo(float xk, const std::array<float,3> &b);
          void  Invert();

          float GetPredictedChi2(const std::array<float,2> &p, const std::array<float,3> &cov) const;
          bool  Update(const std::array<float,2> &p, const std::array<float,3> &cov);

          bool  CorrectForMaterial(float x2x0,float xrho,float mass,bool anglecorr=false,float dedx=kCalcdEdxAuto);

          void  ResetCovariance(float s2=0);
          void  CheckCovariance();

        protected:
          // internal cast to TrackPar
	  /* I. Belikov: Dangerous casts !
          const TrackPar* Param()              const { return reinterpret_cast<const TrackPar*>(this); }
          TrackPar* Param()                          { return reinterpret_cast<TrackPar*>(this); }
	  */
          const TrackPar* Param()              const { return (const TrackPar*)mPC; }
          TrackPar* Param()                          { return (TrackPar*)mPC; }
          bool TrackPar2Momentum(float p[3], float alpha);

        protected:
          float mPC[kTrackPCSize];  // x, alpha + 5 parameters + 15 errors

          static const float kCalcdEdxAuto; // value indicating request for dedx calculation
          ClassDef(TrackParCov, 1)
    };

      class TrackPar final : public TrackParBase { // track parameterization only
        public:
          TrackPar() {}
          TrackPar(float x,float alpha, const std::array<float,kNParams> &par) : TrackParBase{x,alpha,par} {}
          TrackPar(const std::array<float,3> &xyz, const std::array<float,3> &pxpypz,int sign, bool sectorAlpha=true);
          //
          void  Print() const {PrintParam();}
      };

      //____________________________________________________________
      inline TrackParBase::TrackParBase(float x, float alpha, const std::array<float, kNParams> &par) : mX{x}, mAlpha{alpha} {
        // explicit constructor
        std::copy(par.begin(), par.end(), mP);
      }

      //_______________________________________________________
      inline void TrackParBase::GetXYZ(std::array<float,3> &xyz) const {
        // track coordinates in lab frame
        xyz[0] = GetX();
        xyz[1] = GetY();
        xyz[2] = GetZ();
        Utils::RotateZ(xyz,GetAlpha());
      }

      //_______________________________________________________
      inline float TrackParBase::GetPhiPos() const {
        // angle of track position
        float xy[2]={GetX(),GetY()};
        return atan2(xy[1],xy[0]);
      }

      //____________________________________________________________
      inline float TrackParBase::GetP() const {
        // return the track momentum
        float ptI = fabs(GetQ2Pt());
        return (ptI>Constants::kAlmost0) ? sqrtf(1.f+ GetTgl()*GetTgl())/ptI : Constants::kVeryBig;
      }

      //____________________________________________________________
      inline float TrackParBase::GetPt() const {
        // return the track transverse momentum
        float ptI = fabs(GetQ2Pt());
        return (ptI>Constants::kAlmost0) ? 1.f/ptI : Constants::kVeryBig;
      }

      //============================================================

      //____________________________________________________________
      inline TrackParCov::TrackParCov(float x, float alpha, const std::array<float,kNParams> &par, const std::array<float,kCovMatSize> &cov)
	: TrackParBase{x,alpha,par} {
        // explicit constructor
        std::copy(cov.begin(), cov.end(), mC);
      }

      //============================================================

      //____________________________________________________________
      inline TrackPar::TrackPar(const std::array<float,3> &xyz, const std::array<float,3> &pxpypz,int sign, bool sectorAlpha)
	: TrackParBase{xyz,pxpypz,sign,sectorAlpha} {
        // explicit constructor
      }

    }
  }
}


#endif
