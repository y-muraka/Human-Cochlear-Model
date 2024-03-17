#include "CochlearModel_1D_fft.h"

using namespace std;

//extern "C" int dgetrf_(int *m, int *n, double *a, int * lda, int *ipiv, int *info);
//extern "C" int dgetri_(int *n, double *a, int *lda, int	*ipiv, double *work, int *lwork, int *info);


void save_datfile(string filename, vector<vector<double> > data)
{
    ofstream outFile;
    int size_x = data.size();
    int size_y = data.at(0).size();
    
    outFile.open(filename, std::ios::binary);
    for (int ii=0; ii < size_x; ii++)
        for (int jj=0; jj < size_y; jj++)
            outFile.write(reinterpret_cast<const char*>(&data[ii][jj]), sizeof(double));
    outFile.close();
}

void save_datfile(string filename, vector<double> data)
{
    ofstream outFile;
    int size_x = data.size();
    
    outFile.open(filename, std::ios::binary);
    for (int ii=0; ii < size_x; ii++)
            outFile.write(reinterpret_cast<const char*>(&data[ii]), sizeof(double));
    outFile.close();
}

vector<double> mat_to_mat1d(vector<vector<double> > mat)
{
    int size_x = mat.size();
    int size_y = mat.at(0).size();
    vector<double> mat1d(size_x*size_y);

    for (int ii = 0; ii < size_x; ii++)
        for (int jj = 0; jj < size_y; jj++)
            mat1d[ii*mat[0].size() + jj] = mat[ii][jj];

    return mat1d;
}

/*
vector<double> inv(vector<double> mat1d_input)
{
    vector<double> mat1d_output = mat1d_input;

    double* ptr_output = mat1d_output.data();

    int size = (int)sqrt(mat1d_output.size());
    int m = size;
    int n = size;
    int lda = size;
    int info;
    int* ipiv = new int[size];
    int lwork = size;
    double* work = new double[size];

    dgetrf_( &m, &n, ptr_output, &lda, ipiv, &info);
    dgetri_(&n, ptr_output, &lda, ipiv, work, &lwork, &info);

    delete[] ipiv;
    delete[] work;

    return mat1d_output;
}

void matvec_mul(vector<double> mat1d_in, vector<double> vec_in, vector<double>& vec_out)
{
    double alpha = 1.0;
    double beta = 0.0;

    int dim = vec_in.size();

    cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, alpha, mat1d_in.data(), dim, vec_in.data(), 1, beta, vec_out.data(), 1);
}
*/

/*
vector<double> CochlearModel_1D::Gohc(vector<double> uc)
{
    vector<double> output(N);

    for(int ii=0; ii < N; ii++){
        output[ii] = beta * tanh(uc[ii]/beta);
    }

    return output;
}
*/

/*
vector<double> CochlearModel_1D::dGohc(vector<double> uc, vector<double> vc)
{
    vector<double> output(N);

    for(int ii=0; ii < N; ii++){
        output[ii] = vc[ii] / pow(cosh(uc[ii]), 2.0);
    }

    return output;

}
*/

void CochlearModel_1D::get_g(vector<double> vb, vector<double> ub, vector<double> vt, vector<double> ut, vector<double>& gb, vector<double>& gt)
{
    double uc_lin, vc_lin;

    for (int ii=0; ii < N; ii++){
        
        gb[ii] = c1c3[ii]*vb[ii] + k1k3[ii]*ub[ii] - c3[ii]*vt[ii] - k3[ii]*ut[ii];
        gt[ii] = -c3[ii]*vb[ii] - k3[ii]*ub[ii] + c2c3[ii]*vt[ii] + k2k3[ii]*ut[ii];
        uc_lin = ub[ii] - ut[ii];
        vc_lin = vb[ii] - vt[ii];

        gb[ii] -= gamma[ii] * (c4[ii] * vc_lin + k4[ii] * uc_lin);
    }
}

void CochlearModel_1D::solve_time_domain(vector<double> vec_f, vector<vector<double> >& mat_vb, vector<vector<double> >& mat_ub, vector<vector<double> >& mat_p)
{
    int num_time = (int)round(vec_f.size()/2);
    double alpha2 = 4*rho*b/H/m1;
    vector<vector<double> > mat_vt(num_time, vector<double>(N,0));
    vector<vector<double> > mat_ut(num_time, vector<double>(N,0));

    struct timeval tv_start, tv_stop;
    double deltaT;
    double dx2 = pow(dx, 2.0);

    //fftw_init_threads();
    //fftw_plan_with_nthreads(1);

    vector<double> vec_gb(N);
    vector<double> vec_gt(N);
    vector<double> vec_k(N);
    vector<double> vec_khat(N);
    vector<double> vec_phat(N);

    vector<double> vec_dvb1(N), vec_dvb2(N), vec_dvb3(N), vec_dvb4(N);
    vector<double> vec_dvt1(N), vec_dvt2(N), vec_dvt3(N), vec_dvt4(N);
    vector<double> vec_vb1(N), vec_vb2(N), vec_vb3(N);
    vector<double> vec_vt1(N), vec_vt2(N), vec_vt3(N);
    vector<double> vec_ub1(N), vec_ub2(N), vec_ub3(N);
    vector<double> vec_ut1(N), vec_ut2(N), vec_ut3(N);
    vector<double> vec_p1(N), vec_p2(N), vec_p3(N), vec_p4(N);

    int mm;

    vector<double> vec_mwx(N);
    int ax;

    fftw_plan plan_forward, plan1_inverse, plan2_inverse, plan3_inverse, plan4_inverse;

    for (int kx = 1; kx < N+1; kx++){
        ax = M_PI*(2*kx-1)/4/N;
        vec_mwx[kx-1] = 4*(sin(ax)*sin(ax))/dx2;
    }

    plan_forward = fftw_plan_r2r_1d(N, vec_k.data(), vec_khat.data(), FFTW_REDFT10, FFTW_EXHAUSTIVE);
    plan1_inverse = fftw_plan_r2r_1d(N, vec_phat.data(), vec_p1.data(), FFTW_REDFT01, FFTW_EXHAUSTIVE);
    plan2_inverse = fftw_plan_r2r_1d(N, vec_phat.data(), vec_p2.data(), FFTW_REDFT01, FFTW_EXHAUSTIVE);
    plan3_inverse = fftw_plan_r2r_1d(N, vec_phat.data(), vec_p3.data(), FFTW_REDFT01, FFTW_EXHAUSTIVE);
    plan4_inverse = fftw_plan_r2r_1d(N, vec_phat.data(), vec_p4.data(), FFTW_REDFT01, FFTW_EXHAUSTIVE);

    gettimeofday(&tv_start, NULL);

    for (int ii = 0; ii < num_time-1; ii++){
        // RK4

        // (ii)
        get_g(mat_vb[ii], mat_ub[ii], mat_vt[ii], mat_ut[ii], vec_gb, vec_gt);

        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2] * 2/dx;

        // (iii)
        fftw_execute(plan_forward);
        for (mm = 0; mm < N; mm++)
            vec_phat[mm] = vec_khat[mm]/(vec_mwx[mm]-alpha2);
        fftw_execute(plan1_inverse);
        for (mm = 0; mm < N; mm++)
            vec_p1[mm] = vec_p1[mm]/2/N;

        /// (iv)
        for ( mm=0; mm < N; mm++){
            vec_dvb1[mm] = (vec_p1[mm] - vec_gb[mm])/m1;
            vec_ub1[mm] = mat_ub[ii][mm] + 0.5*dt*mat_vb[ii][mm];
            vec_vb1[mm] = mat_vb[ii][mm] + 0.5*dt*vec_dvb1[mm];

            vec_dvt1[mm] = -vec_gt[mm]/m2;
            vec_ut1[mm] = mat_ut[ii][mm] + 0.5*dt*mat_vt[ii][mm];
            vec_vt1[mm] = mat_vt[ii][mm] + 0.5*dt*vec_dvt1[mm];
        }


        // (ii)
        get_g(vec_vb1, vec_ub1, vec_vt1, vec_ut1, vec_gb, vec_gt);

        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2+1] * 2/dx;
        
        // (iii) BAD
        fftw_execute(plan_forward);
        for (mm = 0; mm < N; mm++)
            vec_phat[mm] = vec_khat[mm]/(vec_mwx[mm]-alpha2);
        fftw_execute(plan2_inverse);
        for (mm = 0; mm < N; mm++)
            vec_p2[mm] = vec_p2[mm]/2/N;

        /// (iv)

        for ( mm=0; mm < N; mm++){
            vec_dvb2[mm] = (vec_p2[mm] - vec_gb[mm])/m1;
            vec_ub2[mm] = vec_ub1[mm] + 0.5*dt*vec_vb1[mm];
            vec_vb2[mm] = vec_vb1[mm] + 0.5*dt*vec_dvb2[mm];

            vec_dvt2[mm] = -vec_gt[mm]/m2;
            vec_ut2[mm] = vec_ut1[mm] + 0.5*dt*vec_vt1[mm];
            vec_vt2[mm] = vec_vt1[mm] + 0.5*dt*vec_dvt2[mm];
        }

        // (ii)
        get_g(vec_vb2, vec_ub2, vec_vt2, vec_ut2, vec_gb, vec_gt);


        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2+1] * 2/dx;

        // (iii)
        fftw_execute(plan_forward);
        for (mm = 0; mm < N; mm++)
            vec_phat[mm] = vec_khat[mm]/(vec_mwx[mm]-alpha2);
        fftw_execute(plan3_inverse);
        for (mm = 0; mm < N; mm++)
            vec_p3[mm] = vec_p3[mm]/2/N;


        /// (iv)
        for ( mm=0; mm < N; mm++){
            vec_dvb3[mm] = (vec_p3[mm] - vec_gb[mm])/m1;
            vec_ub3[mm] = vec_ub2[mm] + dt*vec_vb2[mm];
            vec_vb3[mm] = vec_vb2[mm] + dt*vec_dvb3[mm];

            vec_dvt3[mm] = -vec_gt[mm]/m2;
            vec_ut3[mm] = vec_ut2[mm] + dt*vec_vt2[mm];
            vec_vt3[mm] = vec_vt2[mm] + dt*vec_dvt3[mm];

        }

        // (ii)
        get_g(vec_vb3, vec_ub3, vec_vt3, vec_ut3, vec_gb, vec_gt);

        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2+2] * 2/dx;

        // (iii)
        fftw_execute(plan_forward);
        for (mm = 0; mm < N; mm++)
            vec_phat[mm] = vec_khat[mm]/(vec_mwx[mm]-alpha2);
        fftw_execute(plan4_inverse);
        for (mm = 0; mm < N; mm++)
            vec_p4[mm] = vec_p4[mm]/2/N;


        /// (iv)
        for ( mm=0; mm < N; mm++){
            vec_dvb4[mm] = (vec_p4[mm] - vec_gb[mm])/m1;
            vec_dvt4[mm] = -vec_gt[mm] / m2;

            mat_ub[ii+1][mm] = mat_ub[ii][mm] + dt/6*(mat_vb[ii][mm] + 2*vec_vb1[mm] + 2*vec_vb2[mm] + vec_vb3[mm]);
            mat_vb[ii+1][mm] = mat_vb[ii][mm] + dt/6*(vec_dvb1[mm] + 2*vec_dvb2[mm] + 2*vec_dvb3[mm] + vec_dvb4[mm]);
            mat_ut[ii+1][mm] = mat_ut[ii][mm] + dt/6*(mat_vt[ii][mm] + 2*vec_vt1[mm] + 2*vec_vt2[mm] + vec_vt3[mm]);
            mat_vt[ii+1][mm] = mat_vt[ii][mm] + dt/6*(vec_dvt1[mm] + 2*vec_dvt2[mm] + 2*vec_dvt3[mm] + vec_dvt4[mm]);
        }
    }
    gettimeofday(&tv_stop, NULL);
    deltaT =  tv_stop.tv_sec  - tv_start.tv_sec +
    1e-6 * (tv_stop.tv_usec - tv_start.tv_usec);

    printf("Duration of stimulation: %f [sec] \n", num_time*dt);
    printf("Elapsed time: %f [sec] \n", deltaT);

    //fftw_cleanup_threads();
}

vector<double> get_f(double fp, int num_time)
{
    vector<double> vec_f(num_time, 0);
    double dt = 5e-6;
    double t;

    for (int ii = 0; ii < num_time; ii++){
        t = dt*ii;
        vec_f[ii] = sin(2*M_PI*fp*t);
    }

    return vec_f;
}

int main(void)
{
//    openblas_set_num_threads(16);
    int num_segment = 512;
    int num_time = 10000;
    int num_time2 = num_time * 2;
    double fp = 1000.0;

    vector<double> gamma(num_segment,1.0);

    CochlearModel_1D cm(num_segment, gamma);

    vector<vector<double> > mat_vb(num_time, vector<double>(num_segment));
    vector<vector<double> >  mat_ub(num_time, vector<double>(num_segment));
    vector<vector<double> > mat_p(num_time, vector<double>(num_segment));
    vector<double> vec_f = get_f(fp, num_time2);

    cm.solve_time_domain(vec_f, mat_vb, mat_ub, mat_p);

    save_datfile("mat_vb.dat", mat_vb);
    save_datfile("vec_f.dat", vec_f);
}
