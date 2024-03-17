#include <vector>
#include <fstream>
#include <cmath>
//#include <cblas.h>
#include <fftw3.h>
//#include<Accelerate/Accelerate.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

class CochlearModel_1D{

private:
    int N;
    double Lb;
    double L;
    double W;
    double H;
    double rho;
    double dx;
    vector<double> x;

    vector<double> ch_damp;

    vector<double> k1;
    double m1;
    vector<double> c1;
    vector<double> k2;
    double m2;
    vector<double> c2;
    vector<double> k3;
    vector<double> c3;
    vector<double> k4;
    vector<double> c4;

    vector<double> k1k3;
    vector<double> c1c3;
    vector<double> k2k3;
    vector<double> c2c3;

    double g;
    double b;
    vector<double> gamma;

    double dt;
    double beta;

    vector<double> Gohc(vector<double> uc);
    vector<double> dGohc(vector<double> uc, vector<double> vc);
    void get_g(vector<double> vb, vector<double> ub, vector<double> vt, vector<double> ut, vector<double>& gb, vector<double>& gt);

public:
    CochlearModel_1D(int num_segment, vector<double> gamma_given)
    {
        N = num_segment;
        Lb = 3.5;
        L = 0.1;
        W = 0.1;
        H = 0.1;
        rho = 1.0;
        dx = Lb/N;
    
        m1 = 3e-3;
        m2 = 0.5e-3;
        for (int ii = 0; ii < N; ii++){
            x.push_back(dx * ii);
//            ch_damp.push_back(2.8 * exp(-0.1*x[ii]));
            ch_damp.push_back(1.0);

            k1.push_back(2.2e8 * exp(-3 * x[ii]));
            c1.push_back(6 + 670 * exp(-1.5 * x[ii]) * ch_damp[ii]);
            k2.push_back(1.4e6 * exp(-3.3 * x[ii]));
            c2.push_back(4.4 * exp(-1.65 * x[ii]) * ch_damp[ii]);
            k3.push_back(2.0e6 * exp(-3 * x[ii]));
            c3.push_back(0.8 * exp(-0.6 * x[ii]) * ch_damp[ii]);
            k4.push_back(1.15e8 * exp(-3 * x[ii]));
            c4.push_back(440.0 * exp(-1.5 * x[ii]) * ch_damp[ii]);

            k1k3.push_back(k1[ii] + k3[ii]);
            c1c3.push_back(c1[ii] + c3[ii]);
            k2k3.push_back(k2[ii] + k3[ii]);
            c2c3.push_back(c2[ii] + c3[ii]);
            gamma.push_back(gamma_given[ii]);
        }
        g = 1;
        b = 0.4;
        dt = 10e-6;
        beta = 50e-7;
    }

   void solve_time_domain(vector<double> f, vector<vector<double> >& mat_vb, vector<vector<double> >& mat_ub, vector<vector<double> >& mat_p);
};