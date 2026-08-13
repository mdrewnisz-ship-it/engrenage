#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace FTARScalarX {
using namespace Loop;
using std::array;

CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE
CCTK_REAL potential(const CCTK_REAL phi, const CCTK_REAL V0,
                    const CCTK_REAL phis) {
  using std::tanh;
  const CCTK_REAL t = tanh(phi / phis);
  return V0 * t * t;
}

CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE
CCTK_REAL dpotential(const CCTK_REAL phi, const CCTK_REAL V0,
                     const CCTK_REAL phis) {
  using std::tanh;
  const CCTK_REAL t = tanh(phi / phis);
  return (2.0 * V0 / phis) * t * (1.0 - t * t);
}

template <typename GF>
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
d1(const GF &f, const PointDesc &p, const int d) {
  return (f(p.I - 2 * p.DI[d]) - 8.0 * f(p.I - p.DI[d]) +
          8.0 * f(p.I + p.DI[d]) - f(p.I + 2 * p.DI[d])) /
         (12.0 * p.DX[d]);
}

template <typename GF>
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
d2diag(const GF &f, const PointDesc &p, const int d) {
  return (-f(p.I + 2 * p.DI[d]) + 16.0 * f(p.I + p.DI[d]) -
          30.0 * f(p.I) + 16.0 * f(p.I - p.DI[d]) -
          f(p.I - 2 * p.DI[d])) /
         (12.0 * p.DX[d] * p.DX[d]);
}

template <typename GF>
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
d2mixed(const GF &f, const PointDesc &p, const int a, const int b) {
  constexpr int off[4] = {-2, -1, 1, 2};
  constexpr CCTK_REAL c[4] = {1.0, -8.0, 8.0, -1.0};
  CCTK_REAL s = 0.0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      s += c[i] * c[j] * f(p.I + off[i] * p.DI[a] + off[j] * p.DI[b]);
  return s / (144.0 * p.DX[a] * p.DX[b]);
}

struct Mat3 { CCTK_REAL a[3][3]; };

CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE Mat3
inverse3(const Mat3 &g) {
  const CCTK_REAL a = g.a[0][0], b = g.a[0][1], c = g.a[0][2];
  const CCTK_REAL d = g.a[1][1], e = g.a[1][2], f = g.a[2][2];
  const CCTK_REAL det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
  Mat3 q{};
  q.a[0][0] = (d * f - e * e) / det;
  q.a[0][1] = q.a[1][0] = (c * e - b * f) / det;
  q.a[0][2] = q.a[2][0] = (b * e - c * d) / det;
  q.a[1][1] = (a * f - c * c) / det;
  q.a[1][2] = q.a[2][1] = (b * c - a * e) / det;
  q.a[2][2] = (a * d - b * b) / det;
  return q;
}

template <typename GF1, typename GF2>
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE CCTK_REAL
d1ratio(const GF1 &num, const GF2 &den, const PointDesc &p, const int d) {
  const auto im2 = p.I - 2 * p.DI[d];
  const auto im1 = p.I - p.DI[d];
  const auto ip1 = p.I + p.DI[d];
  const auto ip2 = p.I + 2 * p.DI[d];
  return ((num(im2) / den(im2)) - 8.0 * (num(im1) / den(im1)) +
          8.0 * (num(ip1) / den(ip1)) - (num(ip2) / den(ip2))) /
         (12.0 * p.DX[d]);
}

template <typename GTXX, typename GTXY, typename GTXZ, typename GTYY,
          typename GTYZ, typename GTZZ, typename CHI>
CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
physical_metric(const GTXX &gtxx, const GTXY &gtxy, const GTXZ &gtxz,
                const GTYY &gtyy, const GTYZ &gtyz, const GTZZ &gtzz,
                const CHI &chi, const PointDesc &p, Mat3 &g, Mat3 &gi,
                CCTK_REAL dg[3][3][3]) {
  const CCTK_REAL invchi = 1.0 / chi(p.I);
  g.a[0][0] = gtxx(p.I) * invchi;
  g.a[0][1] = g.a[1][0] = gtxy(p.I) * invchi;
  g.a[0][2] = g.a[2][0] = gtxz(p.I) * invchi;
  g.a[1][1] = gtyy(p.I) * invchi;
  g.a[1][2] = g.a[2][1] = gtyz(p.I) * invchi;
  g.a[2][2] = gtzz(p.I) * invchi;
  gi = inverse3(g);
  for (int d = 0; d < 3; ++d) {
    dg[d][0][0] = d1ratio(gtxx, chi, p, d);
    dg[d][0][1] = dg[d][1][0] = d1ratio(gtxy, chi, p, d);
    dg[d][0][2] = dg[d][2][0] = d1ratio(gtxz, chi, p, d);
    dg[d][1][1] = d1ratio(gtyy, chi, p, d);
    dg[d][1][2] = dg[d][2][1] = d1ratio(gtyz, chi, p, d);
    dg[d][2][2] = d1ratio(gtzz, chi, p, d);
  }
}

CCTK_DEVICE CCTK_HOST inline CCTK_ATTRIBUTE_ALWAYS_INLINE void
contracted_christoffel(const Mat3 &gi, const CCTK_REAL dg[3][3][3], CCTK_REAL Gamma[3]) {
  for (int k = 0; k < 3; ++k) {
    CCTK_REAL s = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int l = 0; l < 3; ++l) {
          const CCTK_REAL Gkij = 0.5 * gi.a[k][l] * (dg[i][l][j] + dg[j][l][i] - dg[l][i][j]);
          s += gi.a[i][j] * Gkij;
        }
    Gamma[k] = s;
  }
}

extern "C" void FTARScalarX_InitialData(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_FTARScalarX_InitialData;
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(initial_data, "external")) return;
  grid.loop_all<0, 0, 0>(grid.nghostzones,
    [=] CCTK_HOST(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      if (CCTK_EQUALS(initial_data, "zero")) { phi(p.I)=0.0; Pi(p.I)=0.0; return; }
      using std::exp; using std::sqrt;
      const CCTK_REAL r = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
      const CCTK_REAL x = (r-gaussian_R0)/gaussian_sigma;
      const CCTK_REAL ph = gaussian_amplitude*exp(-0.5*x*x);
      const CCTK_REAL dph = -(r-gaussian_R0)/(gaussian_sigma*gaussian_sigma)*ph;
      CCTK_REAL q=0.0;
      if (CCTK_EQUALS(gaussian_momentum,"paper3_seed") || CCTK_EQUALS(gaussian_momentum,"paper3_ingoing")) q=-(dph+(r>1e-12?ph/r:0.0));
      else if (CCTK_EQUALS(gaussian_momentum,"true_ingoing")) q=+(dph+(r>1e-12?ph/r:0.0));
      else if (CCTK_EQUALS(gaussian_momentum,"simple_negative_characteristic") || CCTK_EQUALS(gaussian_momentum,"simple_ingoing")) q=-dph;
      phi(p.I)=ph; Pi(p.I)=q;
    });
}

extern "C" void FTARScalarX_SyncState(CCTK_ARGUMENTS) {}

extern "C" void FTARScalarX_SetTmunuStage(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_FTARScalarX_SetTmunuStage;
  DECLARE_CCTK_PARAMETERS;
  grid.loop_int_device<2,2,2>(grid.nghostzones,
    [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      Mat3 g{},gi{}; CCTK_REAL dg[3][3][3];
      physical_metric(gtDD00,gtDD01,gtDD02,gtDD11,gtDD12,gtDD22,chi,p,g,gi,dg);
      CCTK_REAL grad[3]={d1(phi,p,0),d1(phi,p,1),d1(phi,p,2)}, grad2=0.0;
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) grad2+=gi.a[i][j]*grad[i]*grad[j];
      const CCTK_REAL V=potential(phi(p.I),V_plateau,phi_star);
      const CCTK_REAL rho=0.5*(Pi(p.I)*Pi(p.I)+grad2)+V;
      CCTK_REAL S[3]; for(int i=0;i<3;++i) S[i]=-Pi(p.I)*grad[i];
      CCTK_REAL Sij[3][3];
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) Sij[i][j]=grad[i]*grad[j]+g.a[i][j]*(0.5*(Pi(p.I)*Pi(p.I)-grad2)-V);
      const CCTK_REAL alpha=evo_lapse(p.I), beta[3]={evo_shiftU0(p.I),evo_shiftU1(p.I),evo_shiftU2(p.I)};
      CCTK_REAL tt=alpha*alpha*rho; for(int i=0;i<3;++i) tt-=2.0*alpha*beta[i]*S[i]; for(int i=0;i<3;++i) for(int j=0;j<3;++j) tt+=beta[i]*beta[j]*Sij[i][j];
      CCTK_REAL ti[3]; for(int i=0;i<3;++i){ti[i]=-alpha*S[i];for(int j=0;j<3;++j)ti[i]+=beta[j]*Sij[i][j];}
      eTtt(p.I)=tt; eTtx(p.I)=ti[0]; eTty(p.I)=ti[1]; eTtz(p.I)=ti[2];
      eTxx(p.I)=Sij[0][0]; eTxy(p.I)=Sij[0][1]; eTxz(p.I)=Sij[0][2]; eTyy(p.I)=Sij[1][1]; eTyz(p.I)=Sij[1][2]; eTzz(p.I)=Sij[2][2];
    });
}

extern "C" void FTARScalarX_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_FTARScalarX_RHS;
  DECLARE_CCTK_PARAMETERS;
  grid.loop_int_device<2,2,2>(grid.nghostzones,
    [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      Mat3 g{},gi{}; CCTK_REAL dg[3][3][3]; physical_metric(gtDD00,gtDD01,gtDD02,gtDD11,gtDD12,gtDD22,chi,p,g,gi,dg);
      CCTK_REAL Gamma[3]; contracted_christoffel(gi,dg,Gamma);
      CCTK_REAL dp[3]={d1(phi,p,0),d1(phi,p,1),d1(phi,p,2)}, dPi[3]={d1(Pi,p,0),d1(Pi,p,1),d1(Pi,p,2)}, da[3]={d1(evo_lapse,p,0),d1(evo_lapse,p,1),d1(evo_lapse,p,2)};
      CCTK_REAL hess[3][3]{}; for(int i=0;i<3;++i){hess[i][i]=d2diag(phi,p,i);for(int j=i+1;j<3;++j)hess[i][j]=hess[j][i]=d2mixed(phi,p,i,j);} CCTK_REAL lap=0.0;
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) lap+=gi.a[i][j]*hess[i][j]; for(int k=0;k<3;++k) lap-=Gamma[k]*dp[k];
      CCTK_REAL agrad=0.0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) agrad+=gi.a[i][j]*da[i]*dp[j];
      const CCTK_REAL beta[3]={evo_shiftU0(p.I),evo_shiftU1(p.I),evo_shiftU2(p.I)}; CCTK_REAL advphi=0.0,advPi=0.0; for(int i=0;i<3;++i){advphi+=beta[i]*dp[i];advPi+=beta[i]*dPi[i];}
      const CCTK_REAL alpha=evo_lapse(p.I); phi_rhs(p.I)=advphi+alpha*Pi(p.I); Pi_rhs(p.I)=advPi+alpha*(lap+trK(p.I)*Pi(p.I)-dpotential(phi(p.I),V_plateau,phi_star))+agrad;
    });
}

extern "C" void FTARScalarX_AddToTmunu(CCTK_ARGUMENTS) {}

extern "C" void FTARScalarX_Diagnostics(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_FTARScalarX_Diagnostics;
  DECLARE_CCTK_PARAMETERS;
  grid.loop_int_device<2,2,2>(grid.nghostzones,
    [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
      scalar_V(p.I)=potential(phi(p.I),V_plateau,phi_star);
      scalar_grad2(p.I)=0.0;
      scalar_rho(p.I)=0.5*Pi(p.I)*Pi(p.I)+scalar_V(p.I);
    });
}

} // namespace FTARScalarX
