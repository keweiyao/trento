#ifndef RFT2D_H
#define RFT2D_H

#include <random>
#include <vector>
#include <fftw3.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <iostream>

namespace RFT {

class iCDF
{
	public:
		iCDF(int N_, double fluct_);
		~iCDF();
		double operator()(double input);
	private:
		const int N;
		const double fluct;
		double dx_gamma;
		double * y_gamma_cdf;
		double * x_gamma;
		gsl_interp_accel * acc;
		gsl_spline * spline;
};	

class rft2d
{
	public:
		rft2d(int const N1_=32, int const N2_=32, double L1=3.0, double L2=3.0, double Var_Phi=1.0, double lx=0.3, int seed = 0, double width_=0.5);
		~rft2d();
		void run();
		void convert_field()
		{
			for (int k1 = 0; k1 < N1; k1++)
			{
				for (int k2 = 0; k2 < N2; k2++)
				{
					phi_x[k1*N1+k2][0] = icdf(phi_x[k1*N1+k2][0]);
				}
			}
		};
		double calculate_proton_norm(int i1, int j1) const
		{
			double result = 0.0;
			int fi1, fj1;
			for (int i = -Ncut; i<=Ncut; i++)
			{
				for (int j = -Ncut; j<=Ncut; j++)
				{
					fi1 = (i1 + i + N1)%N1;
					fj1 = (j1 + j + N2)%N2;
					result += proton_clip[i+Ncut][j+Ncut]*phi_x[fi1*N1+fj1][0];
				}
			}
			return 1.0/result;
		};
		double calculate_fluct_norm(int i1, int j1, int i2, int j2, double bx, double by) const
		{
			double result = 0.0;
			int ibxhalf = int(0.5*bx/dxy);
			int ibyhalf = int(0.5*by/dxy);
			int fi1, fj1, fi2, fj2;
			for (int i = -Ncut; i<=Ncut; i++)
			{
				for (int j = -Ncut; j<=Ncut; j++)
				{
					fi1 = (i1 + ibxhalf + i + N1)%N1;
					fj1 = (j1 + ibyhalf + j + N2)%N2;
					fi2 = (i2 - ibxhalf + i + N1)%N1;
					fj2 = (j2 - ibyhalf + j + N2)%N2;
					result += TAB_clip[i+Ncut][j+Ncut]*phi_x[fi1*N1+fj1][0]*phi_x[fi2*N1+fj2][0];
				}
			}
			return result;
		};
		double get_field(int i, int j) const
		{
			int si = (i+N1)%N1;
			int sj = (j+N2)%N2;
			return phi_x[si*N1+sj][0];
		};
	private:
		// for thickness overlapping calculation
		double ** TAB_clip;
		double ** proton_clip;
		int const Ncut;
		double const dxy2, dxy, width;
		
		// for iCDF mapping
		iCDF icdf;
		//----------steps function call-----------------------------
		void real_space_white_noise(void);
		void apply_k_spcae_propagation(void);
		//----------One time grid init----------------------------
		int const N1, N2;
		double const L1, L2, dx1, dx2, dk1, dk2;
		//----------Kernel function info---------------
		double const lx, Var_x, Var_k, coeff_k;
		
		//---------One time gaussian random variable list-------------------
		std::mt19937 generator;
		std::normal_distribution<double> white_noise_generator;
		
		//---------FFTW3--------------------------------------------
		fftw_complex *phi_k, *phi_x;
		double ** C;
		
		fftw_plan plan_x2k, plan_k2x;
};
}

#endif
