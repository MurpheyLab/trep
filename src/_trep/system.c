#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "trep.h"

#define LU_tolerance 1.0e-20


#define F(i1)                      (*(double*)IDX1(sys->f, i1))
#define LAMBDA(i1)                 (*(double*)IDX1(sys->lambda, i1))
#define D(i1)                      (*(double*)IDX1(sys->D, i1))

#define M_LU(i1, i2)               (*(double*)IDX2(sys->M_lu, i1, i2))
#define A_PROJ_LU(i1, i2)          (*(double*)IDX2(sys->A_proj_lu, i1, i2))

#define AD(i1, i2)                 (*(double*)IDX2(sys->Ad, i1, i2))
#define AD_T(i1, i2)               (*(double*)IDX2(sys->AdT, i1, i2))
#define AK(i1, i2)                 (*(double*)IDX2(sys->Ak, i1, i2))
#define A_DT(i1, i2)               (*(double*)IDX2(sys->Adt, i1, i2))

#define F_DQ(i1, i2)               (*(double*)IDX2(sys->f_dq, i1, i2))
#define F_DDQ(i1, i2)              (*(double*)IDX2(sys->f_ddq, i1, i2))
#define F_DU(i1, i2)               (*(double*)IDX2(sys->f_du, i1, i2))
#define F_DK(i1, i2)               (*(double*)IDX2(sys->f_dk, i1, i2))

#define LAMBDA_DQ(i1, i2)          (*(double*)IDX2(sys->lambda_dq, i1, i2))
#define LAMBDA_DDQ(i1, i2)         (*(double*)IDX2(sys->lambda_ddq, i1, i2))
#define LAMBDA_DU(i1, i2)          (*(double*)IDX2(sys->lambda_du, i1, i2))
#define LAMBDA_DK(i1, i2)          (*(double*)IDX2(sys->lambda_dk, i1, i2))

#define D_DQ(i1, i2)               (*(double*)IDX2(sys->D_dq, i1, i2))
#define D_DDQ(i1, i2)              (*(double*)IDX2(sys->D_ddq, i1, i2))
#define D_DU(i1, i2)               (*(double*)IDX2(sys->D_du, i1, i2))
#define D_DK(i1, i2)               (*(double*)IDX2(sys->D_dk, i1, i2))

#define AD_DQ(i1, i2, i3)          (*(double*)IDX3(sys->Ad_dq, i1, i2, i3))
#define AK_DQ(i1, i2, i3)          (*(double*)IDX3(sys->Ak_dq, i1, i2, i3))
#define A_DT_DQ(i1, i2, i3)        (*(double*)IDX3(sys->Adt_dq, i1, i2, i3))

#define AD_DQDQ(i1, i2, i3, i4)    (*(double*)IDX4(sys->Ad_dqdq, i1, i2, i3, i4))
#define AK_DQDQ(i1, i2, i3, i4)    (*(double*)IDX4(sys->Ak_dqdq, i1, i2, i3, i4))
#define A_DT_DQDQ(i1, i2, i3, i4)  (*(double*)IDX4(sys->Adt_dqdq, i1, i2, i3, i4))

#define D_DQDQ(i1, i2, i3)         (*(double*)IDX3(sys->D_dqdq, i1, i2, i3))
#define D_DDQDQ(i1, i2, i3)        (*(double*)IDX3(sys->D_ddqdq, i1, i2, i3))
#define D_DDQDDQ(i1, i2, i3)       (*(double*)IDX3(sys->D_ddqddq, i1, i2, i3))
#define D_DKDQ(i1, i2, i3)         (*(double*)IDX3(sys->D_dkdq, i1, i2, i3))
#define D_DUDQ(i1, i2, i3)         (*(double*)IDX3(sys->D_dudq, i1, i2, i3))
#define D_DUDDQ(i1, i2, i3)        (*(double*)IDX3(sys->D_duddq, i1, i2, i3))
#define D_DUDU(i1, i2, i3)         (*(double*)IDX3(sys->D_dudu, i1, i2, i3))
#define F_DQDQ(i1, i2, i3)         (*(double*)IDX3(sys->f_dqdq, i1, i2, i3))
#define F_DDQDQ(i1, i2, i3)        (*(double*)IDX3(sys->f_ddqdq, i1, i2, i3))
#define F_DDQDDQ(i1, i2, i3)       (*(double*)IDX3(sys->f_ddqddq, i1, i2, i3))
#define F_DKDQ(i1, i2, i3)         (*(double*)IDX3(sys->f_dkdq, i1, i2, i3))
#define F_DUDQ(i1, i2, i3)         (*(double*)IDX3(sys->f_dudq, i1, i2, i3))
#define F_DUDDQ(i1, i2, i3)        (*(double*)IDX3(sys->f_duddq, i1, i2, i3))
#define F_DUDU(i1, i2, i3)         (*(double*)IDX3(sys->f_dudu, i1, i2, i3))
#define LAMBDA_DQDQ(i1, i2, i3)    (*(double*)IDX3(sys->lambda_dqdq, i1, i2, i3))
#define LAMBDA_DDQDQ(i1, i2, i3)   (*(double*)IDX3(sys->lambda_ddqdq, i1, i2, i3))
#define LAMBDA_DDQDDQ(i1, i2, i3)  (*(double*)IDX3(sys->lambda_ddqddq, i1, i2, i3))
#define LAMBDA_DKDQ(i1, i2, i3)    (*(double*)IDX3(sys->lambda_dkdq, i1, i2, i3))
#define LAMBDA_DUDQ(i1, i2, i3)    (*(double*)IDX3(sys->lambda_dudq, i1, i2, i3))
#define LAMBDA_DUDDQ(i1, i2, i3)   (*(double*)IDX3(sys->lambda_duddq, i1, i2, i3))
#define LAMBDA_DUDU(i1, i2, i3)    (*(double*)IDX3(sys->lambda_dudu, i1, i2, i3))

#define TEMP_ND(i1)                (*(double*)IDX1(sys->temp_nd, i1))

#define M_DQ(i1, i2, i3)           (*(double*)IDX3(sys->M_dq, i1, i2, i3))
#define M_DQDQ(i1, i2, i3, i4)     (*(double*)IDX4(sys->M_dqdq, i1, i2, i3, i4))


double System_total_energy(System *sys)
{
    int m;
    double E = 0.0;
    Frame *mass = NULL;
    Potential *potential = NULL;
    vec6 vb;
    
    E = 0.0;
    for(m = 0; m < System_MASSES(sys); m++) {
	mass = System_MASS(sys, m);
	
	unhat(vb, *Frame_vb(mass));
	E += 0.5*(mass->mass*(vb[0]*vb[0] + vb[1]*vb[1] + vb[2]*vb[2]) +
		  mass->Ixx*(vb[3]*vb[3]) +
		  mass->Iyy*(vb[4]*vb[4]) +
		  mass->Izz*(vb[5]*vb[5]));
    }
    for(m = 0; m < System_POTENTIALS(sys); m++) {
	potential = System_POTENTIAL(sys, m);
	E += potential->V(potential);
    }
    
    return E;
}

double System_L(System *sys)
{
    int m;
    double L = 0.0;
    Frame *mass = NULL;
    Potential *potential = NULL;
    vec6 vb;
    
    L = 0.0;
    for(m = 0; m < System_MASSES(sys); m++) {
	mass = System_MASS(sys, m);	
	unhat(vb, *Frame_vb(mass));
	L += 0.5*(mass->mass*(vb[0]*vb[0] + vb[1]*vb[1] + vb[2]*vb[2]) +
		  mass->Ixx*(vb[3]*vb[3]) +
		  mass->Iyy*(vb[4]*vb[4]) +
		  mass->Izz*(vb[5]*vb[5]));
    }
    for(m = 0; m < System_POTENTIALS(sys); m++) {
	potential = System_POTENTIAL(sys, m);
	L -= potential->V(potential);
    }
    
    return L;
}

double System_L_dq(System *sys, Config *q1)
{
    double L = 0.0;
    Frame *mass = NULL;
    Potential *potential = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(q1); m++) {
	mass = Config_MASS(q1, m);
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_dq(mass, q1));
	
	L += mass->mass*(vb[0]*vb_d1[0] + vb[1]*vb_d1[1] + vb[2]*vb_d1[2]) +
   	     mass->Ixx*(vb[3]*vb_d1[3]) +
	     mass->Iyy*(vb[4]*vb_d1[4]) +
	     mass->Izz*(vb[5]*vb_d1[5]);
    }
    for(m = 0; m < System_POTENTIALS(sys); m++) {
	potential = System_POTENTIAL(sys, m);
	L -= potential->V_dq(potential, q1);
    }
    
    return L;
}

double System_L_dqdq(System *sys, Config *q1, Config *q2)
{
    double L = 0.0;
    Frame *mass = NULL;
    Potential *potential = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    vec6 vb_d12;
    Config *mass_config = NULL;

    mass_config = q1;
    if(Config_MASSES(q2) < Config_MASSES(q1))
	mass_config = q2;
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);

	if(!Frame_USES_CONFIG(mass, q1) ||
	   !Frame_USES_CONFIG(mass, q2))
	    continue;

	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_dq(mass, q1));
	unhat(vb_d2, *Frame_vb_dq(mass, q2));
	unhat(vb_d12, *Frame_vb_dqdq(mass, q1, q2));	
	
	L += mass->mass*(vb_d2[0]*vb_d1[0] + vb_d2[1]*vb_d1[1] + vb_d2[2]*vb_d1[2]) +
   	     mass->Ixx*(vb_d2[3]*vb_d1[3]) +
	     mass->Iyy*(vb_d2[4]*vb_d1[4]) +
	     mass->Izz*(vb_d2[5]*vb_d1[5]);
	L += mass->mass*(vb[0]*vb_d12[0] + vb[1]*vb_d12[1] + vb[2]*vb_d12[2]) +
   	     mass->Ixx*(vb[3]*vb_d12[3]) +
	     mass->Iyy*(vb[4]*vb_d12[4]) +
	     mass->Izz*(vb[5]*vb_d12[5]);
    }
    for(m = 0; m < System_POTENTIALS(sys); m++) {
	potential = System_POTENTIAL(sys, m);
	L -= potential->V_dqdq(potential, q1, q2);
    }
    
    return L;
}

double System_L_dqdqdq(System *sys, Config *q1, Config *q2, Config *q3)
{
    double L = 0.0;
    Frame *mass = NULL;
    Potential *potential = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    vec6 vb_d3;
    vec6 vb_d12;
    vec6 vb_d13;
    vec6 vb_d23;
    vec6 vb_d123;
    Config *mass_config = NULL;

    // Iterate over the config with the smallest number of masses
    mass_config = q1;
    if(Config_MASSES(q2) < Config_MASSES(mass_config))
	mass_config = q2;
    if(Config_MASSES(q3) < Config_MASSES(mass_config))
	mass_config = q3;

    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);

	if(!Frame_USES_CONFIG(mass, q1) ||
	   !Frame_USES_CONFIG(mass, q2) ||
	   !Frame_USES_CONFIG(mass, q3))
	    continue;

	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_dq(mass, q1));
	unhat(vb_d2, *Frame_vb_dq(mass, q2));
	unhat(vb_d3, *Frame_vb_dq(mass, q3));
	unhat(vb_d12, *Frame_vb_dqdq(mass, q1, q2));	
	unhat(vb_d13, *Frame_vb_dqdq(mass, q1, q3));	
	unhat(vb_d23, *Frame_vb_dqdq(mass, q2, q3));	
	unhat(vb_d123, *Frame_vb_dqdqdq(mass, q1, q2, q3));	
	
	L += mass->mass*(vb_d1[0]*vb_d23[0] + vb_d1[1]*vb_d23[1] + vb_d1[2]*vb_d23[2]) +
   	     mass->Ixx*(vb_d1[3]*vb_d23[3]) +
	     mass->Iyy*(vb_d1[4]*vb_d23[4]) +
	     mass->Izz*(vb_d1[5]*vb_d23[5]);
	L += mass->mass*(vb_d2[0]*vb_d13[0] + vb_d2[1]*vb_d13[1] + vb_d2[2]*vb_d13[2]) +
   	     mass->Ixx*(vb_d2[3]*vb_d13[3]) +
	     mass->Iyy*(vb_d2[4]*vb_d13[4]) +
	     mass->Izz*(vb_d2[5]*vb_d13[5]);
	L += mass->mass*(vb_d3[0]*vb_d12[0] + vb_d3[1]*vb_d12[1] + vb_d3[2]*vb_d12[2]) +
   	     mass->Ixx*(vb_d3[3]*vb_d12[3]) +
	     mass->Iyy*(vb_d3[4]*vb_d12[4]) +
	     mass->Izz*(vb_d3[5]*vb_d12[5]);
	L += mass->mass*(vb[0]*vb_d123[0] + vb[1]*vb_d123[1] + vb[2]*vb_d123[2]) +
   	     mass->Ixx*(vb[3]*vb_d123[3]) +
	     mass->Iyy*(vb[4]*vb_d123[4]) +
	     mass->Izz*(vb[5]*vb_d123[5]);
    }
    for(m = 0; m < System_POTENTIALS(sys); m++) {
	potential = System_POTENTIAL(sys, m);
	L -= potential->V_dqdqdq(potential, q1, q2, q3);
    }
    
    return L;
}

double System_L_ddq(System *sys, Config *dq1)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(dq1); m++) {
	mass = Config_MASS(dq1, m);
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	
	L += mass->mass*(vb[0]*vb_d1[0] + vb[1]*vb_d1[1] + vb[2]*vb_d1[2]) +
	      mass->Ixx*(vb[3]*vb_d1[3]) +
	      mass->Iyy*(vb[4]*vb_d1[4]) +
	      mass->Izz*(vb[5]*vb_d1[5]);
    }
    
    return L;
}

double System_L_ddqdq(System *sys, Config *dq1, Config *q2)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    vec6 vb_d12;
    Config *mass_config = NULL;

    // Find the config with the fewest number of masses
    mass_config = dq1;
    if(Config_MASSES(q2) < Config_MASSES(mass_config))
	mass_config = q2;
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);

	if(!Frame_USES_CONFIG(mass, dq1) ||
	   !Frame_USES_CONFIG(mass, q2))
	    continue;
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	unhat(vb_d2, *Frame_vb_dq(mass, q2));
	unhat(vb_d12, *Frame_vb_ddqdq(mass, dq1, q2));
	
	L += mass->mass*(vb_d2[0]*vb_d1[0] + vb_d2[1]*vb_d1[1] + vb_d2[2]*vb_d1[2]) +
	      mass->Ixx*(vb_d2[3]*vb_d1[3]) +
	      mass->Iyy*(vb_d2[4]*vb_d1[4]) +
	      mass->Izz*(vb_d2[5]*vb_d1[5]);
	L += mass->mass*(vb[0]*vb_d12[0] + vb[1]*vb_d12[1] + vb[2]*vb_d12[2]) +
	      mass->Ixx*(vb[3]*vb_d12[3]) +
	      mass->Iyy*(vb[4]*vb_d12[4]) +
	      mass->Izz*(vb[5]*vb_d12[5]);
    }
    
    return L;
}

double System_L_ddqdqdq(System *sys, Config *dq1, Config *q2, Config *q3)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    vec6 vb_d3;
    vec6 vb_d12;
    vec6 vb_d13;
    vec6 vb_d23;
    vec6 vb_d123;
    Config *mass_config = NULL;
    
    mass_config = dq1;
    if(Config_MASSES(q2) < Config_MASSES(mass_config))
	mass_config = q2; 
    if(Config_MASSES(q3) < Config_MASSES(mass_config))
	mass_config = q3;
   
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);
	if(!Frame_USES_CONFIG(mass, dq1) ||
	   !Frame_USES_CONFIG(mass, q2) ||
	   !Frame_USES_CONFIG(mass, q3))
	    continue;
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	unhat(vb_d2, *Frame_vb_dq(mass, q2));
	unhat(vb_d3, *Frame_vb_dq(mass, q3));
	unhat(vb_d12, *Frame_vb_ddqdq(mass, dq1, q2));
	unhat(vb_d13, *Frame_vb_ddqdq(mass, dq1, q3));
	unhat(vb_d23, *Frame_vb_dqdq(mass, q2, q3));
	unhat(vb_d123, *Frame_vb_ddqdqdq(mass, dq1, q2, q3));
	
	L += mass->mass*(vb_d1[0]*vb_d23[0] + vb_d1[1]*vb_d23[1] + vb_d1[2]*vb_d23[2]) +
	      mass->Ixx*(vb_d1[3]*vb_d23[3]) +
	      mass->Iyy*(vb_d1[4]*vb_d23[4]) +
	      mass->Izz*(vb_d1[5]*vb_d23[5]);
	L += mass->mass*(vb_d2[0]*vb_d13[0] + vb_d2[1]*vb_d13[1] + vb_d2[2]*vb_d13[2]) +
	      mass->Ixx*(vb_d2[3]*vb_d13[3]) +
	      mass->Iyy*(vb_d2[4]*vb_d13[4]) +
	      mass->Izz*(vb_d2[5]*vb_d13[5]);
	L += mass->mass*(vb_d3[0]*vb_d12[0] + vb_d3[1]*vb_d12[1] + vb_d3[2]*vb_d12[2]) +
	      mass->Ixx*(vb_d3[3]*vb_d12[3]) +
	      mass->Iyy*(vb_d3[4]*vb_d12[4]) +
	      mass->Izz*(vb_d3[5]*vb_d12[5]);
	L += mass->mass*(vb[0]*vb_d123[0] + vb[1]*vb_d123[1] + vb[2]*vb_d123[2]) +
	      mass->Ixx*(vb[3]*vb_d123[3]) +
	      mass->Iyy*(vb[4]*vb_d123[4]) +
	      mass->Izz*(vb[5]*vb_d123[5]);
    }
    
    return L;
}

double System_L_ddqdqdqdq(System *sys, Config *dq1, Config *q2, Config *q3, Config *q4)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1, vb_d2, vb_d3, vb_d4;
    vec6 vb_d12, vb_d13, vb_d14, vb_d23, vb_d24, vb_d34;
    vec6 vb_d234, vb_d134, vb_d124, vb_d123; 
    vec6 vb_d1234;
    Config *mass_config = NULL;
    
    mass_config = dq1;
    if(Config_MASSES(q2) < Config_MASSES(mass_config))
	mass_config = q2; 
    if(Config_MASSES(q3) < Config_MASSES(mass_config))
	mass_config = q3;
    if(Config_MASSES(q4) < Config_MASSES(mass_config))
	mass_config = q4;
   
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);

	if(!Frame_USES_CONFIG(mass, dq1) ||
	   !Frame_USES_CONFIG(mass, q2) ||
	   !Frame_USES_CONFIG(mass, q3) ||
	   !Frame_USES_CONFIG(mass, q4))
	    continue;

	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	unhat(vb_d2, *Frame_vb_dq(mass, q2));
	unhat(vb_d3, *Frame_vb_dq(mass, q3));
	unhat(vb_d4, *Frame_vb_dq(mass, q4));
	unhat(vb_d12, *Frame_vb_ddqdq(mass, dq1, q2));
	unhat(vb_d13, *Frame_vb_ddqdq(mass, dq1, q3));
	unhat(vb_d14, *Frame_vb_ddqdq(mass, dq1, q4));
	unhat(vb_d23, *Frame_vb_dqdq(mass, q2, q3));
	unhat(vb_d24, *Frame_vb_dqdq(mass, q2, q4));
	unhat(vb_d34, *Frame_vb_dqdq(mass, q3, q4));
	unhat(vb_d234, *Frame_vb_dqdqdq(mass, q2, q3, q4));
	unhat(vb_d134, *Frame_vb_ddqdqdq(mass, dq1, q3, q4));
	unhat(vb_d124, *Frame_vb_ddqdqdq(mass, dq1, q2, q4));
	unhat(vb_d123, *Frame_vb_ddqdqdq(mass, dq1, q2, q3));
	unhat(vb_d1234, *Frame_vb_ddqdqdqdq(mass, dq1, q2, q3, q4));
	
	L += mass->mass*(vb_d1[0]*vb_d234[0] + vb_d1[1]*vb_d234[1] + vb_d1[2]*vb_d234[2]) +
	      mass->Ixx*(vb_d1[3]*vb_d234[3]) +
	      mass->Iyy*(vb_d1[4]*vb_d234[4]) +
	      mass->Izz*(vb_d1[5]*vb_d234[5]);
	L += mass->mass*(vb_d2[0]*vb_d134[0] + vb_d2[1]*vb_d134[1] + vb_d2[2]*vb_d134[2]) +
	      mass->Ixx*(vb_d2[3]*vb_d134[3]) +
	      mass->Iyy*(vb_d2[4]*vb_d134[4]) +
	      mass->Izz*(vb_d2[5]*vb_d134[5]);
	L += mass->mass*(vb_d3[0]*vb_d124[0] + vb_d3[1]*vb_d124[1] + vb_d3[2]*vb_d124[2]) +
	      mass->Ixx*(vb_d3[3]*vb_d124[3]) +
	      mass->Iyy*(vb_d3[4]*vb_d124[4]) +
	      mass->Izz*(vb_d3[5]*vb_d124[5]);
	L += mass->mass*(vb_d4[0]*vb_d123[0] + vb_d4[1]*vb_d123[1] + vb_d4[2]*vb_d123[2]) +
	      mass->Ixx*(vb_d4[3]*vb_d123[3]) +
	      mass->Iyy*(vb_d4[4]*vb_d123[4]) +
	      mass->Izz*(vb_d4[5]*vb_d123[5]);
	L += mass->mass*(vb_d12[0]*vb_d34[0] + vb_d12[1]*vb_d34[1] + vb_d12[2]*vb_d34[2]) +
	      mass->Ixx*(vb_d12[3]*vb_d34[3]) +
	      mass->Iyy*(vb_d12[4]*vb_d34[4]) +
	      mass->Izz*(vb_d12[5]*vb_d34[5]);
	L += mass->mass*(vb_d13[0]*vb_d24[0] + vb_d13[1]*vb_d24[1] + vb_d13[2]*vb_d24[2]) +
	      mass->Ixx*(vb_d13[3]*vb_d24[3]) +
	      mass->Iyy*(vb_d13[4]*vb_d24[4]) +
	      mass->Izz*(vb_d13[5]*vb_d24[5]);
	L += mass->mass*(vb_d14[0]*vb_d23[0] + vb_d14[1]*vb_d23[1] + vb_d14[2]*vb_d23[2]) +
	      mass->Ixx*(vb_d14[3]*vb_d23[3]) +
	      mass->Iyy*(vb_d14[4]*vb_d23[4]) +
	      mass->Izz*(vb_d14[5]*vb_d23[5]);
	L += mass->mass*(vb[0]*vb_d1234[0] + vb[1]*vb_d1234[1] + vb[2]*vb_d1234[2]) +
	      mass->Ixx*(vb[3]*vb_d1234[3]) +
	      mass->Iyy*(vb[4]*vb_d1234[4]) +
	      mass->Izz*(vb[5]*vb_d1234[5]);
    }
    
    return L;
}

double System_L_ddqddq(System *sys, Config *dq1, Config *dq2)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    Config *mass_config = NULL;
    
    mass_config = dq1;
    if(Config_MASSES(dq2) < Config_MASSES(mass_config))
	mass_config = dq2; 
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);

	if(!Frame_USES_CONFIG(mass, dq1) ||
	   !Frame_USES_CONFIG(mass, dq2))
	    continue;
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	unhat(vb_d2, *Frame_vb_ddq(mass, dq2));

	L += mass->mass*(vb_d1[0]*vb_d2[0] + vb_d1[1]*vb_d2[1] + vb_d1[2]*vb_d2[2]) +
	      mass->Ixx*(vb_d1[3]*vb_d2[3]) +
  	      mass->Iyy*(vb_d1[4]*vb_d2[4]) +
	      mass->Izz*(vb_d1[5]*vb_d2[5]);
    }
    
    return L;   
}

double System_L_ddqddqdq(System *sys, Config *dq1, Config *dq2, Config *q3)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    vec6 vb_d13;
    vec6 vb_d23;
    Config *mass_config = NULL;
    
    mass_config = dq1;
    if(Config_MASSES(dq2) < Config_MASSES(mass_config))
	mass_config = dq2; 
    if(Config_MASSES(q3) < Config_MASSES(mass_config))
	mass_config = q3; 
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);
	if(!Frame_USES_CONFIG(mass, dq1) ||
	   !Frame_USES_CONFIG(mass, dq2) ||
	   !Frame_USES_CONFIG(mass, q3))
	    continue;
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	unhat(vb_d2, *Frame_vb_ddq(mass, dq2));
	unhat(vb_d13, *Frame_vb_ddqdq(mass, dq1, q3));
	unhat(vb_d23, *Frame_vb_ddqdq(mass, dq2, q3));

	L += mass->mass*(vb_d1[0]*vb_d23[0] + vb_d1[1]*vb_d23[1] + vb_d1[2]*vb_d23[2]) +
	      mass->Ixx*(vb_d1[3]*vb_d23[3]) +
  	      mass->Iyy*(vb_d1[4]*vb_d23[4]) +
	      mass->Izz*(vb_d1[5]*vb_d23[5]);
	L += mass->mass*(vb_d2[0]*vb_d13[0] + vb_d2[1]*vb_d13[1] + vb_d2[2]*vb_d13[2]) +
	      mass->Ixx*(vb_d2[3]*vb_d13[3]) +
  	      mass->Iyy*(vb_d2[4]*vb_d13[4]) +
	      mass->Izz*(vb_d2[5]*vb_d13[5]);
    }
    
    return L;   
}

double System_L_ddqddqdqdq(System *sys, Config *dq1, Config *dq2, Config *q3, Config *q4)
{
    double L = 0.0;
    Frame *mass = NULL;
    int m;
    vec6 vb;
    vec6 vb_d1;
    vec6 vb_d2;
    vec6 vb_d13;
    vec6 vb_d14;
    vec6 vb_d24;
    vec6 vb_d23;
    vec6 vb_d134;
    vec6 vb_d234;
    Config *mass_config = NULL;
    
    mass_config = dq1;
    if(Config_MASSES(dq2) < Config_MASSES(mass_config))
	mass_config = dq2; 
    if(Config_MASSES(q3) < Config_MASSES(mass_config))
	mass_config = q3; 
    if(Config_MASSES(q4) < Config_MASSES(mass_config))
	mass_config = q4; 
    
    L = 0.0;
    for(m = 0; m < Config_MASSES(mass_config); m++) {
	mass = Config_MASS(mass_config, m);

	if(!Frame_USES_CONFIG(mass, dq1) ||
	   !Frame_USES_CONFIG(mass, dq2) ||
	   !Frame_USES_CONFIG(mass, q3) ||
	   !Frame_USES_CONFIG(mass, q4))
	    continue;
	
	unhat(vb, *Frame_vb(mass));
	unhat(vb_d1, *Frame_vb_ddq(mass, dq1));
	unhat(vb_d2, *Frame_vb_ddq(mass, dq2));
	unhat(vb_d13, *Frame_vb_ddqdq(mass, dq1, q3));
	unhat(vb_d14, *Frame_vb_ddqdq(mass, dq1, q4));
	unhat(vb_d23, *Frame_vb_ddqdq(mass, dq2, q3));
	unhat(vb_d24, *Frame_vb_ddqdq(mass, dq2, q4));
	unhat(vb_d134, *Frame_vb_ddqdqdq(mass, dq1, q3, q4));
	unhat(vb_d234, *Frame_vb_ddqdqdq(mass, dq2, q3, q4));

	L += mass->mass*(vb_d1[0]*vb_d234[0] + vb_d1[1]*vb_d234[1] + vb_d1[2]*vb_d234[2]) +
	      mass->Ixx*(vb_d1[3]*vb_d234[3]) +
  	      mass->Iyy*(vb_d1[4]*vb_d234[4]) +
	      mass->Izz*(vb_d1[5]*vb_d234[5]);
	L += mass->mass*(vb_d2[0]*vb_d134[0] + vb_d2[1]*vb_d134[1] + vb_d2[2]*vb_d134[2]) +
	      mass->Ixx*(vb_d2[3]*vb_d134[3]) +
  	      mass->Iyy*(vb_d2[4]*vb_d134[4]) +
	      mass->Izz*(vb_d2[5]*vb_d134[5]);
	L += mass->mass*(vb_d13[0]*vb_d24[0] + vb_d13[1]*vb_d24[1] + vb_d13[2]*vb_d24[2]) +
	      mass->Ixx*(vb_d13[3]*vb_d24[3]) +
  	      mass->Iyy*(vb_d13[4]*vb_d24[4]) +
	      mass->Izz*(vb_d13[5]*vb_d24[5]);
	L += mass->mass*(vb_d14[0]*vb_d23[0] + vb_d14[1]*vb_d23[1] + vb_d14[2]*vb_d23[2]) +
	      mass->Ixx*(vb_d14[3]*vb_d23[3]) +
  	      mass->Iyy*(vb_d14[4]*vb_d23[4]) +
	      mass->Izz*(vb_d14[5]*vb_d23[5]);
    }
    
    return L;   
}

double System_F(System *sys, Config *q)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f(F, q);
    }
    return result;
}

double System_F_dq(System *sys, Config *q, Config *q1)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_dq(F, q, q1);	    
    }
    return result;
}
    
double System_F_ddq(System *sys, Config *q, Config *dq1)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_ddq(F, q, dq1);	    
    }
    return result;
}

double System_F_du(System *sys, Config *q, Input *u1)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_du(F, q, u1);	    
    }
    return result;
}

double System_F_dqdq(System *sys, Config *q, Config *q1, Config *q2)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_dqdq(F, q, q1, q2);	    
    }
    return result;
}

double System_F_ddqdq(System *sys, Config *q, Config *dq1, Config *q2)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_ddqdq(F, q, dq1, q2);	    
    }
    return result;
}

double System_F_ddqddq(System *sys, Config *q, Config *dq1, Config *dq2)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_ddqddq(F, q, dq1, dq2);	    
    }
    return result;
}

double System_F_dudq(System *sys, Config *q, Input *u1, Config *q2)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_dudq(F, q, u1, q2);	    
    }
    return result;
}

double System_F_duddq(System *sys, Config *q, Input *u1, Config *dq2)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_duddq(F, q, u1, dq2);	    
    }
    return result;
}

double System_F_dudu(System *sys, Config *q, Input *u1, Input *u2)
{
    int i;
    Force *F;
    double result = 0.0;    
    for(i = 0; i < System_FORCES(sys); i++) {
	F = System_FORCE(sys, i);
	result += F->f_dudu(F, q, u1, u2);	    
    }
    return result;
}
    

/*
 * Calculate the inertia matrix inverse.  Returns 0 on
 * success, -1 on failure.
 */
static int calc_inertia_matrix_inverse(System *sys)
{
    int i, j;
    Config *qi, *qj;
    int nd = System_DYN_CONFIGS(sys);
    
    for(i = 0; i < nd; i++) {
	qi = System_DYN_CONFIG(sys, i);
	M_LU(i, i) = System_L_ddqddq(sys, qi, qi);
	for(j = 0; j < i; j++) {
	    qj = System_DYN_CONFIG(sys, j);
	    M_LU(i, j) = System_L_ddqddq(sys, qi, qj);
	    M_LU(j, i) = M_LU(i, j);
	}
    }
    // Calculate M inverse
    if(LU_decomp(sys->M_lu, nd, sys->M_lu_index, LU_tolerance))
	return -1;
    return 0;
}

/*
 * Calculate D for the current state.  Returns 0 on success, -1 on failure.
 */
static int calc_D(System *sys)
{
    int i1, i2;
    Config *q1, *q2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    
    for(i1 = 0; i1 < nd; i1++) {
	q1 = System_DYN_CONFIG(sys, i1);
	D(i1) = System_L_dq(sys, q1);
	for(i2 = 0; i2 < nk; i2++) {
	    q2 = System_KIN_CONFIG(sys, i2);	    
	    D(i1) -= System_L_ddqddq(sys, q1, q2)*q2->ddq;
	}
	for(i2 = 0; i2 < nd+nk; i2++) {
	    q2 = System_CONFIG(sys, i2);
	    D(i1) -= System_L_ddqdq(sys, q1, q2)*q2->dq;
	}
	D(i1) += System_F(sys, q1);
    }
    // An error might have occured when calling a force.
    if(PyErr_Occurred())
	return -1;
    return 0;    
}

/*
 * Calculate the constraint matrices required for the dynamics,
 * incuding the A_proj used everywhere.  Returns 0 on success, -1 on
 * failure.
 */
static int calc_constraints(System *sys)
{
    int i1, i2, i3;
    Constraint *constraint;
    Config *q2, *q3;
    
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i1 = 0; i1 < nc; i1++) {
	constraint = System_CONSTRAINT(sys, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    q2 = System_DYN_CONFIG(sys, i2);
	    AD(i1, i2) = constraint->h_dq(constraint, q2);
	    AD_T(i2, i1) = AD(i1, i2);
	}
	for(i2 = 0; i2 < nk; i2++) {
	    q2 = System_KIN_CONFIG(sys, i2);
	    AK(i1, i2) = constraint->h_dq(constraint, q2);
	}
	for(i2 = 0; i2 < nd+nk; i2++) {
	    q2 = System_CONFIG(sys, i2);
	    A_DT(i1, i2) = 0.0;
	    for(i3 = 0; i3 < nd+nk; i3++) {
		q3 = System_CONFIG(sys, i3);
		A_DT(i1, i2) += constraint->h_dqdq(constraint, q2, q3)*q3->dq;
	    }
	}
    }
    // An error might have occured when calling a constraint.
    if(PyErr_Occurred())
	return -1;
    
    // Calculate the A projection term.
    copy_np_matrix(sys->temp_ndnc, sys->AdT, nd, nc);
    LU_solve_mat(sys->M_lu, nd, sys->M_lu_index, sys->temp_ndnc, nc);
    mul_matmat_np_np_np(sys->A_proj_lu, nc, nc, sys->Ad, sys->temp_ndnc, nd);
    // Negate A_proj here so that we don't have to negate lambda and every derivative.
    for(i1 = 0; i1 < nc; i1++) { 
	for(i2 = 0; i2 < nc; i2++)
	    A_PROJ_LU(i1, i2) = -A_PROJ_LU(i1, i2);
    }
    if(LU_decomp(sys->A_proj_lu, nc, sys->A_proj_lu_index, LU_tolerance))
	return -1;
	
    return 0;    
}

static void calc_lambda(System *sys)
{
    int i1, i2;
    Config *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    // Solve for lambda
    copy_vector(&TEMP_ND(0), &D(0), nd);
    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
    mul_matvec_c_np_c(&LAMBDA(0), nc, sys->Ad, &TEMP_ND(0), nd);
     
    for(i1 = 0; i1 < nc; i1++) {
	for(i2 = 0; i2 < nk; i2++) {
	    q2 = System_KIN_CONFIG(sys, i2); 
	    LAMBDA(i1) += AK(i1, i2)*q2->ddq;
	}
	for(i2 = 0; i2 < nd+nk; i2++) {
	    q2 = System_CONFIG(sys, i2);
	    LAMBDA(i1) += A_DT(i1, i2)*q2->dq;
	}
    }
    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA(0));
}    

static void calc_f(System *sys)
{
    int i1;
    
    int nd = System_DYN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    mul_matvec_c_np_c(&F(0), nd, sys->AdT, &LAMBDA(0), nc);
    for(i1 = 0; i1 < nd; i1++)
        F(i1) +=  D(i1);
    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F(0));

    for(i1 = 0; i1 < nd; i1++)
	System_DYN_CONFIG(sys, i1)->ddq = F(i1);
}    

static int calc_dynamics(System *sys)
{
    if(sys->cache & SYSTEM_CACHE_DYNAMICS)
	return 0;
    if(calc_inertia_matrix_inverse(sys))
	return -1;
    if(calc_constraints(sys))
	return -1;
    if(calc_D(sys))
	return -1;
    
    calc_lambda(sys);
    calc_f(sys);
    sys->cache |= SYSTEM_CACHE_DYNAMICS;
    return 0;
}


/*
 * Calculate M_dq.
 */
static void calc_M_dq(System *sys)
{
    int i, i1, i2;
    Config *qi, *q1, *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);

    for(i = 0; i < nd+nk; i++) {
	qi = System_CONFIG(sys, i);
	for(i1 = 0; i1 < nd+nk; i1++) {
	    q1 = System_CONFIG(sys, i1);
	    for(i2 = i1; i2 < nd+nk; i2++) {
		q2 = System_CONFIG(sys, i2);
		M_DQ(i, i1, i2) = System_L_ddqddqdq(sys, q1, q2, qi);
		M_DQ(i, i2, i1) = M_DQ(i, i1, i2);
	    }
	}
    }
}	    
	
/*
 * Calculate derivatives of D needed for the first derivative
 * calculations. Returns -1 on failure and 0 on success.
 *
 * The full dynamics should be calculate before calling this function.
 */
static int calc_D_deriv1(System *sys)
{
    int i, i1, i2;
    Config *qi, *q1, *q2;
    Input *ui;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nu = System_INPUTS(sys);

    // Calculate D_dq
    // The outer loop is the derivative variable index
    for(i = 0; i < nd+nk; i++) {  
	qi = System_CONFIG(sys, i);
	for(i1 = 0; i1 < nd; i1++) {
	    q1 = System_DYN_CONFIG(sys, i1);
	    D_DQ(i, i1) = System_L_dqdq(sys, q1, qi);
	    for(i2 = 0; i2 < nk; i2++) {
		q2 = System_KIN_CONFIG(sys, i2);	    
		D_DQ(i, i1) -= M_DQ(i, i1, nd+i2)*q2->ddq;
	    }
	    for(i2 = 0; i2 < nd+nk; i2++) {
		q2 = System_CONFIG(sys, i2);
		D_DQ(i, i1) -= System_L_ddqdqdq(sys, q1, q2, qi)*q2->dq;
	    }
	    D_DQ(i, i1) += System_F_dq(sys, q1, qi);
	}
    }

    // Calculate D_ddq
    // The outer loop is the derivative variable index
    for(i = 0; i < nd+nk; i++) {
	qi = System_CONFIG(sys, i);
	
	for(i1 = 0; i1 < nd; i1++) {
	    q1 = System_DYN_CONFIG(sys, i1);
	    D_DDQ(i, i1) = System_L_ddqdq(sys, qi, q1);
	    D_DDQ(i, i1) -= System_L_ddqdq(sys, q1, qi);	    
	    for(i2 = 0; i2 < nd+nk; i2++) {
		q2 = System_CONFIG(sys, i2);
		D_DDQ(i, i1) -= M_DQ(i2, i1, i)*q2->dq;
	    }
	    D_DDQ(i, i1) += System_F_ddq(sys, q1, qi);
	}
    }

    
    // Calculate D_dk
    // The outer loop is the derivative variable index
    for(i = 0; i < nk; i++) {
	qi = System_KIN_CONFIG(sys, i);
	for(i1 = 0; i1 < nd; i1++) {
	    q1 = System_DYN_CONFIG(sys, i1);
	    D_DK(i, i1) = -System_L_ddqddq(sys, q1, qi);
	}
    }

    // Calculate D_du
    // The outer loop is the derivative variable index
    for(i = 0; i < nu; i++) {
	ui = System_INPUT(sys, i);
	for(i1 = 0; i1 < nd; i1++) {
	    q1 = System_DYN_CONFIG(sys, i1);
	    D_DU(i, i1) = System_F_du(sys, q1, ui);
	}
    }

    // An error might have occured when calling a force.
    if(PyErr_Occurred())
	return -1;
    return 0;    
}

/*
 * Calculate derivatives of the constraint matrices needed for the
 * first derivatives of the dynamics.  Returns -1 on failure and 0 on
 * success.
 */ 
static int calc_constraints_deriv1(System *sys)
{
    int i, i1, i2, i3;
    Constraint *constraint;
    Config *qi, *q2, *q3;
    
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i1 = 0; i1 < nc; i1++) {
	constraint = System_CONSTRAINT(sys, i1);
	// i is the derivative variable index
	for(i = 0; i < nd+nk; i++) {
	    qi = System_CONFIG(sys, i);
	    for(i2 = 0; i2 < nd; i2++) {
		q2 = System_DYN_CONFIG(sys, i2);
		AD_DQ(i, i1, i2) = constraint->h_dqdq(constraint, q2, qi);
	    }
	    for(i2 = 0; i2 < nk; i2++) {
		q2 = System_KIN_CONFIG(sys, i2);
		AK_DQ(i, i1, i2) = constraint->h_dqdq(constraint, q2, qi);
	    }
	    for(i2 = 0; i2 < nd+nk; i2++) {
		q2 = System_CONFIG(sys, i2);
		A_DT_DQ(i, i1, i2) = 0.0;
		for(i3 = 0; i3 < nd+nk; i3++) {
		    q3 = System_CONFIG(sys, i3);
		    A_DT_DQ(i, i1, i2) += constraint->h_dqdqdq(constraint, q2, qi, q3)*q3->dq;
		}
	    }
	}
    }
    // An error might have occured when calling a constraint.
    if(PyErr_Occurred())
	return -1;    
    return 0;    
}

/*
 * Calculate the derivative of lambda with respect to the
 * configuration variables.  
 */ 
static void calc_lambda_dq(System *sys)
{
    int i, i1, i2;
    // Config *qi, *q1, *q2;
    Config *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	//qi = System_CONFIG(sys, i);
	
	for(i1 = 0; i1 < nd; i1++) {
	    //q1 = System_CONFIG(sys, i1);
	    TEMP_ND(i1) = D_DQ(i, i1);
	    for(i2 = 0; i2 < nc; i2++) 
		TEMP_ND(i1) += AD_DQ(i, i2, i1) * LAMBDA(i2);
	    for(i2 = 0; i2 < nd; i2++) {
		//q2 = System_CONFIG(sys, i2);
		TEMP_ND(i1) -= M_DQ(i, i1, i2)*F(i2);
	    }
	}
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	mul_matvec_c_np_c(&LAMBDA_DQ(i, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	for(i1 = 0; i1 < nc; i1++) {
	    for(i2 = 0; i2 < nd+nk; i2++) {
		q2 = System_CONFIG(sys, i2);
		LAMBDA_DQ(i, i1) += A_DT_DQ(i, i1, i2)*q2->dq;
	    }
	    for(i2 = 0; i2 < nk; i2++) {
		q2 = System_KIN_CONFIG(sys, i2);
		LAMBDA_DQ(i, i1) += AK_DQ(i, i1, i2)*q2->ddq;
	    }
	    for(i2 = 0; i2 < nd; i2++) {
		//q2 = System_DYN_CONFIG(sys, i2);
		LAMBDA_DQ(i, i1) += AD_DQ(i, i1, i2)*F(i2);
	    }
	}
	LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DQ(i, 0));
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * configuration variables.  
 */ 
static void calc_f_dq(System *sys)
{
    int i, i1, i2;
    //Config *qi, *q1, *q2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	//qi = System_CONFIG(sys, i);
	for(i1 = 0; i1 < nd; i1++) {
	    //q1 = System_DYN_CONFIG(sys, i1);
	    F_DQ(i, i1) = D_DQ(i, i1);
	    for(i2 = 0; i2 < nc; i2++) {
		F_DQ(i, i1) += AD_DQ(i, i2, i1)*LAMBDA(i2);
		F_DQ(i, i1) += AD_T(i1, i2)*LAMBDA_DQ(i, i2);
	    }
	    for(i2 = 0; i2 < nd; i2++) {
		//q2 = System_DYN_CONFIG(sys, i2);
		F_DQ(i, i1) -= M_DQ(i, i1, i2)*F(i2);
	    }
	}
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DQ(i, 0));
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * configuration velocities.  
 */ 
static void calc_lambda_ddq(System *sys)
{
    int i, i1, i2;
    //Config *qi, *q2;
    Config *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	// qi = System_CONFIG(sys, i);
        
	copy_vector(&TEMP_ND(0), &D_DDQ(i, 0), nd);
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	mul_matvec_c_np_c(&LAMBDA_DDQ(i, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	for(i1 = 0; i1 < nc; i1++) {
	    for(i2 = 0; i2 < nd; i2++) {
		q2 = System_DYN_CONFIG(sys, i2);
		LAMBDA_DDQ(i, i1) += AD_DQ(i, i1, i2)*q2->dq;
	    }
	    for(i2 = 0; i2 < nk; i2++) {
		q2 = System_KIN_CONFIG(sys, i2);
		LAMBDA_DDQ(i, i1) += AK_DQ(i, i1, i2)*q2->dq;
	    }
	    LAMBDA_DDQ(i, i1) += A_DT(i1, i);
	}
	LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DDQ(i, 0));
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * configuration velocities.  
 */ 
static void calc_f_ddq(System *sys)
{
    int i, i1, i2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	for(i1 = 0; i1 < nd; i1++) {
	    F_DDQ(i, i1) = D_DDQ(i, i1);
	    for(i2 = 0; i2 < nc; i2++) 
		F_DDQ(i, i1) += AD_T(i1, i2)*LAMBDA_DDQ(i, i2);
	}
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DDQ(i, 0));
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * kinematic accelerations.  
 */ 
static void calc_lambda_dk(System *sys)
{
    int i, i1;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nk; i++) {
	copy_vector(&TEMP_ND(0), &D_DK(i, 0), nd);
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	mul_matvec_c_np_c(&LAMBDA_DK(i, 0), nc, sys->Ad, &TEMP_ND(0), nd);
	for(i1 = 0; i1 < nc; i1++) 
	    LAMBDA_DK(i, i1) += AK(i1, i);
	LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DK(i, 0));
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * kinematic accelerations.
 */ 
static void calc_f_dk(System *sys)
{
    int i, i1, i2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nk; i++) {
	for(i1 = 0; i1 < nd; i1++) {
	    F_DK(i, i1) = D_DK(i, i1);
	    for(i2 = 0; i2 < nc; i2++) 
		F_DK(i, i1) += AD_T(i1, i2)*LAMBDA_DK(i, i2);
	}
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DK(i, 0));
    }
}    

/*
 * Calculate the derivative of lambda with respect to the force
 * inputs.
 */ 
static void calc_lambda_du(System *sys)
{
    int i;
    //Input *ui;

    int nd = System_DYN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) {
	//ui = System_INPUT(sys, i);

	copy_vector(&TEMP_ND(0), &D_DU(i, 0), nd);
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	mul_matvec_c_np_c(&LAMBDA_DU(i, 0), nc, sys->Ad, &TEMP_ND(0), nd);
	LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DU(i, 0));
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the force
 * inputs.
 */ 
static void calc_f_du(System *sys)
{
    int i, i1, i2;
    int nd = System_DYN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);
    
    for(i = 0; i < nu; i++) {
	for(i1 = 0; i1 < nd; i1++) {
	    F_DU(i, i1) = D_DU(i, i1);
	    for(i2 = 0; i2 < nc; i2++) 
		F_DU(i, i1) += AD_T(i1, i2)*LAMBDA_DU(i, i2);
	}
	LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DU(i, 0));
    }
}    

static int calc_dynamics_deriv1(System *sys)
{
    if(sys->cache & SYSTEM_CACHE_DYNAMICS_DERIV1)
	return 0;
    if(!(sys->cache & SYSTEM_CACHE_DYNAMICS) && calc_dynamics(sys))
	    return -1;
    calc_M_dq(sys);
    if(calc_D_deriv1(sys))
	return -1;

    if(System_CONSTRAINTS(sys) > 0) {
	if(calc_constraints_deriv1(sys))
	    return -1;
	calc_lambda_dq(sys);
	calc_lambda_ddq(sys);
	calc_lambda_dk(sys);
	calc_lambda_du(sys);
    }
    
    calc_f_dq(sys);
    calc_f_ddq(sys);
    calc_f_dk(sys);
    calc_f_du(sys);
    
    sys->cache |= SYSTEM_CACHE_DYNAMICS_DERIV1;
    
    return 0;
}

/*
 * Calculate M_dqdq.
 */
static void calc_M_dqdq(System *sys)
{
    int i, j, i1, i2;
    Config *qi, *qj, *q1, *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);

    for(i = 0; i < nd+nk; i++) {
	qi = System_CONFIG(sys, i);
	for(j = i; j < nd+nk; j++) {
	    qj = System_CONFIG(sys, j);
	    for(i1 = 0; i1 < nd+nk; i1++) {
		q1 = System_CONFIG(sys, i1);
		for(i2 = i1; i2 < nd+nk; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    M_DQDQ(i, j, i1, i2) = System_L_ddqddqdqdq(sys, q1, q2, qi,qj);
		    M_DQDQ(i, j, i2, i1) = M_DQDQ(i, j, i1, i2);
		    M_DQDQ(j, i, i2, i1) = M_DQDQ(i, j, i1, i2);
		    M_DQDQ(j, i, i1, i2) = M_DQDQ(i, j, i1, i2);
		}
	    }
	}
    }
}	    
	

/*
 * Calculate derivatives of D needed for the second derivative
 * calculations. Returns -1 on failure and 0 on success.
 *
 * The first derivative of dynamics should be calculated before calling
 * this function.
 */
static int calc_D_deriv2(System *sys)
{
    int i, j, i1, i2;
    Config *qi, *qj, *q1, *q2;
    Input *ui, *uj;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nu = System_INPUTS(sys);
    
    // Calculate D_dqdq
    for(i = 0; i < nd+nk; i++) {  
	qi = System_CONFIG(sys, i);  // First derivative variable
	for(j = i; j < nd+nk; j++) {
	    qj = System_CONFIG(sys, j);   // Second derivative variable
	    for(i1 = 0; i1 < nd; i1++) {
		q1 = System_DYN_CONFIG(sys, i1);    // The outer loop is the derivative variable index

		D_DQDQ(i, j, i1) = System_L_dqdqdq(sys, q1, qi, qj);
		for(i2 = 0; i2 < nk; i2++) {
		    q2 = System_KIN_CONFIG(sys, i2);	    
		    D_DQDQ(i, j, i1) -= M_DQDQ(i, j, i1, nd+i2)*q2->ddq;
		}
		for(i2 = 0; i2 < nd+nk; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    D_DQDQ(i, j, i1) -= System_L_ddqdqdqdq(sys, q1, q2, qi, qj)*q2->dq;
		}
		D_DQDQ(i, j, i1) += System_F_dqdq(sys, q1, qi, qj);
	    }
	    // Take advantage of symmetry  i <= k
	    copy_vector(&D_DQDQ(j, i, 0), &D_DQDQ(i, j, 0), nd);
	}
    }

    // Calculate D_ddqdq
    for(i = 0; i < nd+nk; i++) {
	qi = System_CONFIG(sys, i);  // First derivative variable
	for(j = 0; j < nd+nk; j++) {
	    qj = System_CONFIG(sys, j);  // Second derivative variable
	
	    for(i1 = 0; i1 < nd; i1++) {
		q1 = System_DYN_CONFIG(sys, i1);
		D_DDQDQ(i, j, i1) = System_L_ddqdqdq(sys, qi, qj, q1);
		D_DDQDQ(i, j, i1) -= System_L_ddqdqdq(sys, q1, qi, qj);	    
		for(i2 = 0; i2 < nd+nk; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    D_DDQDQ(i, j, i1) -= M_DQDQ(i2, j, i, i1)*q2->dq;
		}
		D_DDQDQ(i, j, i1) += System_F_ddqdq(sys, q1, qi, qj);
	    }
	}
    }

    // Calculate D_ddqddq
    // The outer loop is the derivative variable index
    for(i = 0; i < nd+nk; i++) {
	qi = System_CONFIG(sys, i); // First derivative variable
	for(j = i; j < nd+nk; j++) {
	    qj = System_CONFIG(sys, j); // Second derivative variable
	
	    for(i1 = 0; i1 < nd; i1++) {
		q1 = System_DYN_CONFIG(sys, i1);
		D_DDQDDQ(i, j, i1) = M_DQ(i1, i, j);
		D_DDQDDQ(i, j, i1) -= M_DQ(i, i1, j);
		D_DDQDDQ(i, j, i1) -= M_DQ(j, i1, i);
		D_DDQDDQ(i, j, i1) += System_F_ddqddq(sys, q1, qi, qj);
	    }
	    // Take advantage of symmetry here
	    copy_vector(&D_DDQDDQ(j, i, 0), &D_DDQDDQ(i, j, 0), nd);
	}
    }

    // Calculate D_dkdq
    for(i = 0; i < nk; i++) {  // First Derivative Variable
	for(j = 0; j < nd+nk; j++) { // Second derivative variable
	    for(i1 = 0; i1 < nd; i1++) {
		D_DKDQ(i, j, i1) = -M_DQ(j, nd+i, i1);
	    }
	}
    }

    // Calculate D_dudq and D_duddq
    for(i = 0; i < nu; i++) {
	ui = System_INPUT(sys, i);  // First derivative variable
	for(j = 0; j < nd+nk; j++) {
	    qj = System_CONFIG(sys, j); // Second derivative variable
	    for(i1 = 0; i1 < nd; i1++) {
		q1 = System_DYN_CONFIG(sys, i1);
		D_DUDQ(i, j, i1) = System_F_dudq(sys, q1, ui, qj);
		D_DUDDQ(i, j, i1) = System_F_duddq(sys, q1, ui, qj);
	    }
	}
    }

    // Calculate D_dudu
    for(i = 0; i < nu; i++) {
	ui = System_INPUT(sys, i);  // First derivative variable
	for(j = i; j < nu; j++) {
	    uj = System_INPUT(sys, j); // Second derivative variable
	    for(i1 = 0; i1 < nd; i1++) {
		q1 = System_DYN_CONFIG(sys, i1);
		D_DUDU(i, j, i1) = System_F_dudu(sys, q1, ui, uj);
	    }
	    copy_vector(&D_DUDU(j, i, 0), &D_DUDU(i, j, 0), nd);            
	}
    }

    // An error might have occured when calling a force.
    if(PyErr_Occurred())
	return -1;
    return 0;    
}

/*
 * Calculate derivatives of the constraint matrices needed for the
 * second derivatives of the dynamics.  Returns -1 on failure and 0 on
 * success.
 */ 
static int calc_constraints_deriv2(System *sys)
{
    int i, j, i1, i2, i3;
    Constraint *constraint;
    Config *qi, *qj, *q2, *q3;
    
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i1 = 0; i1 < nc; i1++) {
	constraint = System_CONSTRAINT(sys, i1);
	for(i = 0; i < nd+nk; i++) {  
	    qi = System_CONFIG(sys, i);  // First derivative variable
	    for(j = i; j < nd+nk; j++) {
		qj = System_CONFIG(sys, j); // Second derivative variable
		for(i2 = 0; i2 < nd; i2++) {
		    q2 = System_DYN_CONFIG(sys, i2);
		    AD_DQDQ(i, j, i1, i2) = constraint->h_dqdqdq(constraint, q2, qi, qj);
		    AD_DQDQ(j, i, i1, i2) = AD_DQDQ(i, j, i1, i2);
		}
		for(i2 = 0; i2 < nk; i2++) {
		    q2 = System_KIN_CONFIG(sys, i2);
		    AK_DQDQ(i, j, i1, i2) = constraint->h_dqdqdq(constraint, q2, qi, qj);
		    AK_DQDQ(j, i, i1, i2) = AK_DQDQ(i, j, i1, i2);
		}
		/* Adt_dqdq is calculated in a wonky way here to take
		   advantage of symmetry in h_dqdqdqdq.  This gives a
		   15% speed up for the second derivative. */
		for(i2 = 0; i2 < nd+nk; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    A_DT_DQDQ(i, j, i1, i2) = constraint->h_dqdqdqdq(constraint, q2, q2, qi, qj)*q2->dq;
		    for(i3 = 0; i3 < i2; i3++) {
			q3 = System_CONFIG(sys, i3);
			double h = constraint->h_dqdqdqdq(constraint, q2, q3, qi, qj);
			A_DT_DQDQ(i, j, i1, i2) += h*q3->dq;
			A_DT_DQDQ(i, j, i1, i3) += h*q2->dq;
		    }
		    A_DT_DQDQ(j, i, i1, i2) = A_DT_DQDQ(i, j, i1, i2);
		}
	    }
	}
    }
    // An error might have occured when calling a constraint.
    if(PyErr_Occurred())
	return -1;    
    return 0;    
}

/*
 * Calculate the derivative of lambda with respect to the
 * configuration variables and configuration variables
 */ 
static void calc_lambda_dqdq(System *sys)
{
    int i, j, i1, i2;
    //Config *qi, *qj, *q1, *q2;
    Config *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	//qi = System_CONFIG(sys, i); // First derivative variable
	for(j = i; j < nd+nk; j++) {
	    //qj = System_CONFIG(sys, j); // Second derivative variable
	    
	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_CONFIG(sys, i1);
		TEMP_ND(i1) = D_DQDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) {
		    TEMP_ND(i1) += AD_DQDQ(i, j, i2, i1) * LAMBDA(i2);
		    TEMP_ND(i1) += AD_DQ(i, i2, i1) * LAMBDA_DQ(j, i2);
		    TEMP_ND(i1) += AD_DQ(j, i2, i1) * LAMBDA_DQ(i, i2);
		}
		for(i2 = 0; i2 < nd; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    TEMP_ND(i1) -= M_DQDQ(i, j, i1, i2)*F(i2);
		    TEMP_ND(i1) -= M_DQ(i, i1, i2)*F_DQ(j, i2);
		    TEMP_ND(i1) -= M_DQ(j, i1, i2)*F_DQ(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DQDQ(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	    for(i1 = 0; i1 < nc; i1++) {
		for(i2 = 0; i2 < nd+nk; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    LAMBDA_DQDQ(i, j, i1) += A_DT_DQDQ(i, j, i1, i2)*q2->dq;
		}
		for(i2 = 0; i2 < nk; i2++) {
		    q2 = System_KIN_CONFIG(sys, i2);
		    LAMBDA_DQDQ(i, j, i1) += AK_DQDQ(i, j, i1, i2)*q2->ddq;
		}
		for(i2 = 0; i2 < nd; i2++) {
		    q2 = System_DYN_CONFIG(sys, i2);
		    LAMBDA_DQDQ(i, j, i1) += AD_DQDQ(i, j, i1, i2)*F(i2);
		    LAMBDA_DQDQ(i, j, i1) += AD_DQ(i, i1, i2)*F_DQ(j, i2);
		    LAMBDA_DQDQ(i, j, i1) += AD_DQ(j, i1, i2)*F_DQ(i, i2);
		}
	    }
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DQDQ(i, j, 0));
	    copy_vector(&LAMBDA_DQDQ(j, i, 0), &LAMBDA_DQDQ(i, j, 0), nc);
	}
    }
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * configuration variables and configuration variables.
 */ 
static void calc_f_dqdq(System *sys)
{
    int i, j, i1, i2;
    //Config *qi, *qj, *q1, *q2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	//qi = System_CONFIG(sys, i); // First derivative variable
	for(j = i; j < nd+nk; j++) {
	    //qj = System_CONFIG(sys, j); // Second derivative variable
	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_DYN_CONFIG(sys, i1);  
		F_DQDQ(i, j, i1) = D_DQDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) {
		    F_DQDQ(i, j, i1) += AD_DQDQ(i, j, i2, i1)*LAMBDA(i2);
		    F_DQDQ(i, j, i1) += AD_DQ(j, i2, i1)*LAMBDA_DQ(i, i2);
		    F_DQDQ(i, j, i1) += AD_DQ(i, i2, i1)*LAMBDA_DQ(j, i2);
		    F_DQDQ(i, j, i1) += AD(i2, i1)*LAMBDA_DQDQ(i, j, i2);
		}
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_DYN_CONFIG(sys, i2);
		    F_DQDQ(i, j, i1) -= M_DQDQ(i, j, i1, i2)*F(i2);
		    F_DQDQ(i, j, i1) -= M_DQ(i, i1, i2)*F_DQ(j, i2);
		    F_DQDQ(i, j, i1) -= M_DQ(j, i1, i2)*F_DQ(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DQDQ(i, j, 0));
	    copy_vector(&F_DQDQ(j, i, 0), &F_DQDQ(i, j, 0), nd);
	}
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * configuration velocities and configuration variables.
 */ 
static void calc_lambda_ddqdq(System *sys)
{
    int i, j, i1, i2;
    //Config *qi, *qj, *q1, *q2;
    Config *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	// qi = System_CONFIG(sys, i); // First derivative variable (config velocity)
	for(j = 0; j < nd+nk; j++) { 
	    //qj = System_CONFIG(sys, j);  // second derivative variable (config variable)

	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_CONFIG(sys, i1);
		TEMP_ND(i1) = D_DDQDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) 
		    TEMP_ND(i1) += AD_DQ(j, i2, i1) * LAMBDA_DDQ(i, i2);
		for(i2 = 0; i2 < nd; i2++) {
		    q2 = System_CONFIG(sys, i2);
		    TEMP_ND(i1) -= M_DQ(j, i1, i2)*F_DDQ(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DDQDQ(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	    for(i1 = 0; i1 < nc; i1++) {
		for(i2 = 0; i2 < nd; i2++) {
		    q2 = System_DYN_CONFIG(sys, i2);
		    LAMBDA_DDQDQ(i, j, i1) += AD_DQDQ(i, j, i1, i2)*q2->dq;
		    LAMBDA_DDQDQ(i, j, i1) += AD_DQ(j, i1, i2)*F_DDQ(i, i2);
		}
		for(i2 = 0; i2 < nk; i2++) {
		    q2 = System_KIN_CONFIG(sys, i2);
		    LAMBDA_DDQDQ(i, j, i1) += AK_DQDQ(i, j, i1, i2)*q2->dq;
		}
		LAMBDA_DDQDQ(i, j, i1) += A_DT_DQ(j, i1, i);
	    }
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DDQDQ(i, j, 0));
	}
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * configuration velocities and configuration variables.
 */ 
static void calc_f_ddqdq(System *sys)
{
    int i, j, i1, i2;
    //Config *qj, *q1, *q2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) {
	for(j = 0; j < nd+nk; j++) {
	    //qj = System_CONFIG(sys, j);
	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_DYN_CONFIG(sys, i1);
		F_DDQDQ(i, j, i1) = D_DDQDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) {
		    F_DDQDQ(i, j, i1) += AD_DQ(j, i2, i1)*LAMBDA_DDQ(i, i2);
		    F_DDQDQ(i, j, i1) += AD(i2, i1)*LAMBDA_DDQDQ(i, j, i2);
		}
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_DYN_CONFIG(sys, i2);
		    F_DDQDQ(i, j, i1) -= M_DQ(j, i1, i2)*F_DDQ(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DDQDQ(i, j, 0));
	}
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * configuration velocities and configuration velocities.
 */ 
static void calc_lambda_ddqddq(System *sys)
{
    int i, j, i1;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) { // First derivative variable (config velocity)
	for(j = 0; j < nd+nk; j++) { // Second derivative variable (config velocity)
	    copy_vector(&TEMP_ND(0), &D_DDQDDQ(i, j, 0), nd);
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DDQDDQ(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	    for(i1 = 0; i1 < nc; i1++) {
		if(j < nd)
		    LAMBDA_DDQDDQ(i, j, i1) += AD_DQ(i, i1, j);
		else
		    LAMBDA_DDQDDQ(i, j, i1) += AK_DQ(i, i1, j-nd);
		if(i < nd)
		    LAMBDA_DDQDDQ(i, j, i1) += AD_DQ(j, i1, i);
		else
		    LAMBDA_DDQDDQ(i, j, i1) += AK_DQ(j, i1, i-nd);		
	    }
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DDQDDQ(i, j, 0));
	}
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * configuration velocities and configuration velocities.
 */ 
static void calc_f_ddqddq(System *sys)
{
    int i, j, i1, i2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nd+nk; i++) { // First derivative index
	for(j = 0; j < nd+nk; j++) { // Second derivative index
	    for(i1 = 0; i1 < nd; i1++) {
		F_DDQDDQ(i, j, i1) = D_DDQDDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++)
		    F_DDQDDQ(i, j, i1) += AD_T(i1, i2)*LAMBDA_DDQDDQ(i, j, i2);
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DDQDDQ(i, j, 0));
	    copy_vector(&F_DDQDDQ(j, i, 0), &F_DDQDDQ(i, j, 0), nd);
	}
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * kinematic accelerations and configuration variables.
 */ 
static void calc_lambda_dkdq(System *sys)
{
    int i, j, i1, i2;
    //Config *qj, *q1, *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nk; i++) { // First derivative variable (kinematic accel)
	for(j = 0; j < nd+nk; j++) { 
	    //qj = System_CONFIG(sys, j);  // Second derivative variable (config variable)

	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_CONFIG(sys, i1);
		TEMP_ND(i1) = D_DKDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) 
		    TEMP_ND(i1) += AD_DQ(j, i2, i1) * LAMBDA_DK(i, i2);
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_CONFIG(sys, i2);
		    TEMP_ND(i1) -= M_DQ(j, i1, i2)*F_DK(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DKDQ(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	    for(i1 = 0; i1 < nc; i1++) {
		LAMBDA_DKDQ(i, j, i1) += AK_DQ(j, i1, i);
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_DYN_CONFIG(sys, i2);
		    LAMBDA_DKDQ(i, j, i1) += AD_DQ(j, i1, i2)*F_DK(i, i2);
		}
	    }
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DKDQ(i, j, 0));
	}
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * kinematic acceleration and configuration variables.
 */ 
static void calc_f_dkdq(System *sys)
{
    int i, j, i1, i2;
    //Config *qj, *q1, *q2;
    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);

    for(i = 0; i < nk; i++) {
	for(j = 0; j < nd+nk; j++) {
	    //qj = System_CONFIG(sys, j);
	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_DYN_CONFIG(sys, i1);
		F_DKDQ(i, j, i1) = D_DKDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) {
		    F_DKDQ(i, j, i1) += AD_DQ(j, i2, i1)*LAMBDA_DK(i, i2);
		    F_DKDQ(i, j, i1) += AD(i2, i1)*LAMBDA_DKDQ(i, j, i2);
		}
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_DYN_CONFIG(sys, i2);
		    F_DKDQ(i, j, i1) -= M_DQ(j, i1, i2)*F_DK(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DKDQ(i, j, 0));
	}
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * force inputs and configuration variables.
 */ 
static void calc_lambda_dudq(System *sys)
{
    int i, j, i1, i2;
    //Config *qj, *q1, *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) {  // First derivative index (input)
	for(j = 0; j < nd+nk; j++) { 
	    //qj = System_CONFIG(sys, j);  // second derivative variable (config variable)

	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_CONFIG(sys, i1);
		TEMP_ND(i1) = D_DUDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) 
		    TEMP_ND(i1) += AD_DQ(j, i2, i1) * LAMBDA_DU(i, i2);
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_CONFIG(sys, i2);
		    TEMP_ND(i1) -= M_DQ(j, i1, i2)*F_DU(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DUDQ(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);

	    for(i1 = 0; i1 < nc; i1++) {
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_DYN_CONFIG(sys, i2);
		    LAMBDA_DUDQ(i, j, i1) += AD_DQ(j, i1, i2)*F_DU(i, i2);
		}
	    }
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DUDQ(i, j, 0));
	}
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * force inputs and configuration variables.
 */ 
static void calc_f_dudq(System *sys)
{
    int i, j, i1, i2;
    //Config *qj, *q1, *q2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) {
	for(j = 0; j < nd+nk; j++) {
	    //qj = System_CONFIG(sys, j);
	    for(i1 = 0; i1 < nd; i1++) {
		//q1 = System_DYN_CONFIG(sys, i1);
		F_DUDQ(i, j, i1) = D_DUDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) {
		    F_DUDQ(i, j, i1) += AD_DQ(j, i2, i1)*LAMBDA_DU(i, i2);
		    F_DUDQ(i, j, i1) += AD(i2, i1)*LAMBDA_DUDQ(i, j, i2);
		}
		for(i2 = 0; i2 < nd; i2++) {
		    //q2 = System_DYN_CONFIG(sys, i2);
		    F_DUDQ(i, j, i1) -= M_DQ(j, i1, i2)*F_DU(i, i2);
		}
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DUDQ(i, j, 0));
	}
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * inputs and configuration velocities.
 */ 
static void calc_lambda_duddq(System *sys)
{
    int i, j;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) { // First derivative variable (input)
	for(j = 0; j < nd+nk; j++) { // Second derivative variable (config velocity)
	    copy_vector(&TEMP_ND(0), &D_DUDDQ(i, j, 0), nd);
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DUDDQ(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DUDDQ(i, j, 0));
	}
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * force inputs and configuration velocities.
 */ 
static void calc_f_duddq(System *sys)
{
    int i, j, i1, i2;

    int nd = System_DYN_CONFIGS(sys);
    int nk = System_KIN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) {
	for(j = 0; j < nd+nk; j++) {
	    for(i1 = 0; i1 < nd; i1++) {
		F_DUDDQ(i, j, i1) = D_DUDDQ(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) 
		    F_DUDDQ(i, j, i1) += AD_T(i1, i2)*LAMBDA_DUDDQ(i, j, i2);
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DUDDQ(i, j, 0));
	}
    }
}    

/*
 * Calculate the derivative of lambda with respect to the
 * inputs and inputs.
 */ 
static void calc_lambda_dudu(System *sys)
{
    int i, j;

    int nd = System_DYN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) { // First derivative variable (input)
	for(j = 0; j < nu; j++) { // Second derivative variable (input)
	    copy_vector(&TEMP_ND(0), &D_DUDU(i, j, 0), nd);
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &TEMP_ND(0));
	    mul_matvec_c_np_c(&LAMBDA_DUDU(i, j, 0), nc, sys->Ad, &TEMP_ND(0), nd);
	    LU_solve_vec(sys->A_proj_lu, nc, sys->A_proj_lu_index, &LAMBDA_DUDU(i, j, 0));
	}
    }		
}

/*
 * Calculate the derivative of the dynamics with respect to the
 * force inputs and inputs.
 */ 
static void calc_f_dudu(System *sys)
{
    int i, j, i1, i2;

    int nd = System_DYN_CONFIGS(sys);
    int nc = System_CONSTRAINTS(sys);
    int nu = System_INPUTS(sys);

    for(i = 0; i < nu; i++) {
	for(j = 0; j < nu; j++) {
	    for(i1 = 0; i1 < nd; i1++) {
		F_DUDU(i, j, i1) = D_DUDU(i, j, i1);
		for(i2 = 0; i2 < nc; i2++) 
		    F_DUDU(i, j, i1) += AD_T(i1, i2)*LAMBDA_DUDU(i, j, i2);
	    }
	    LU_solve_vec(sys->M_lu, nd, sys->M_lu_index, &F_DUDU(i, j, 0));
	}
    }
}    


int calc_dynamics_deriv2(System *sys)
{
    if(sys->cache & SYSTEM_CACHE_DYNAMICS_DERIV2)
	return 0;
    if(!(sys->cache & SYSTEM_CACHE_DYNAMICS_DERIV1) && calc_dynamics_deriv1(sys))
	    return -1;
    calc_M_dqdq(sys);
    if(calc_D_deriv2(sys))
	return -1;

    if(System_CONSTRAINTS(sys) > 0) {
	if(calc_constraints_deriv2(sys))
	    return -1;
	calc_lambda_dqdq(sys);
	calc_lambda_ddqdq(sys);
	calc_lambda_ddqddq(sys);
	calc_lambda_dkdq(sys);
	calc_lambda_dudq(sys);
	calc_lambda_duddq(sys);
	calc_lambda_dudu(sys);
    }
	
    calc_f_dqdq(sys);
    calc_f_ddqdq(sys);
    calc_f_ddqddq(sys);
    calc_f_dkdq(sys);
    calc_f_dudq(sys);
    calc_f_duddq(sys);
    calc_f_dudu(sys);
    
    sys->cache |= SYSTEM_CACHE_DYNAMICS_DERIV2;
    return 0;
}


/***********************************************************************
 * Python API
 **********************************************************************/

static void dealloc(System *sys)
{
    Py_CLEAR(sys->world_frame);
    Py_CLEAR(sys->frames);
    Py_CLEAR(sys->configs);
    Py_CLEAR(sys->dyn_configs);
    Py_CLEAR(sys->kin_configs);
    Py_CLEAR(sys->potentials);
    Py_CLEAR(sys->forces);
    Py_CLEAR(sys->inputs);
    Py_CLEAR(sys->constraints);
    Py_CLEAR(sys->masses);

    Py_CLEAR(sys->f);
    Py_CLEAR(sys->lambda);
    Py_CLEAR(sys->D);

    Py_CLEAR(sys->M_lu); 
    Py_CLEAR(sys->M_lu_index);
    Py_CLEAR(sys->A_proj_lu); 
    Py_CLEAR(sys->A_proj_lu_index);
    Py_CLEAR(sys->temp_ndnc);
    
    Py_CLEAR(sys->Ad);
    Py_CLEAR(sys->AdT);
    Py_CLEAR(sys->Ak);
    Py_CLEAR(sys->Adt);
    Py_CLEAR(sys->Ad_dq);
    Py_CLEAR(sys->Ak_dq);
    Py_CLEAR(sys->Adt_dq);
    Py_CLEAR(sys->D_dq);
    Py_CLEAR(sys->D_ddq);
    Py_CLEAR(sys->D_du);
    Py_CLEAR(sys->D_dk);
    Py_CLEAR(sys->f_dq);
    Py_CLEAR(sys->f_ddq);
    Py_CLEAR(sys->f_du);
    Py_CLEAR(sys->f_dk);
    Py_CLEAR(sys->lambda_dq);
    Py_CLEAR(sys->lambda_ddq);
    Py_CLEAR(sys->lambda_du);
    Py_CLEAR(sys->lambda_dk);
    
    Py_CLEAR(sys->Ad_dqdq);
    Py_CLEAR(sys->Ak_dqdq);
    Py_CLEAR(sys->Adt_dqdq);

    Py_CLEAR(sys->D_dqdq);
    Py_CLEAR(sys->D_ddqdq);
    Py_CLEAR(sys->D_ddqddq);
    Py_CLEAR(sys->D_dkdq);
    Py_CLEAR(sys->D_dudq);
    Py_CLEAR(sys->D_duddq);
    Py_CLEAR(sys->D_dudu);

    Py_CLEAR(sys->f_dqdq);
    Py_CLEAR(sys->f_ddqdq);
    Py_CLEAR(sys->f_ddqddq);
    Py_CLEAR(sys->f_dkdq);
    Py_CLEAR(sys->f_dudq);
    Py_CLEAR(sys->f_duddq);
    Py_CLEAR(sys->f_dudu);
    
    Py_CLEAR(sys->lambda_dqdq);
    Py_CLEAR(sys->lambda_ddqdq);
    Py_CLEAR(sys->lambda_ddqddq);
    Py_CLEAR(sys->lambda_dkdq);
    Py_CLEAR(sys->lambda_dudq);
    Py_CLEAR(sys->lambda_duddq);
    Py_CLEAR(sys->lambda_dudu);

    Py_CLEAR(sys->temp_nd);

    Py_CLEAR(sys->M_dq);
    Py_CLEAR(sys->M_dqdq);
        
    sys->ob_type->tp_free((PyObject*)sys);
}

static int init(System *sys, PyObject *args, PyObject *kwds)
{   
    sys->time = 0.0;
    sys->cache = SYSTEM_CACHE_NONE;
    return 0;
}

static PyObject* total_energy(System* sys)
{
    return PyFloat_FromDouble(System_total_energy(sys));
}

static PyObject* L(System* sys)
{
    double result;
    result = System_L(sys);
    if(PyErr_Occurred())
	return NULL;
    else
	return PyFloat_FromDouble(result);
}

static PyObject* L_dq(System *sys, PyObject *args)
{
    Config *q1 = NULL;
    double result = 0;
    
    if(!PyArg_ParseTuple(args, "O", &q1))
        return NULL; 
    result = System_L_dq(sys, q1);
    if(PyErr_Occurred())
	return NULL;
    else
	return PyFloat_FromDouble(result);
}

static PyObject* L_dqdq(System *sys, PyObject *args)
{
    Config *q1 = NULL;
    Config *q2 = NULL;
    double result = 0.0;
    
    if(!PyArg_ParseTuple(args, "OO", &q1, &q2))
        return NULL; 
    result = System_L_dqdq(sys, q1, q2);
    if(PyErr_Occurred())
	return NULL;
    else
	return PyFloat_FromDouble(result);
}

static PyObject* L_dqdqdq(System *sys, PyObject *args)
{
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;
    double result = 0.0;

    if(!PyArg_ParseTuple(args, "OOO", &q1, &q2, &q3))
        return NULL;
    
    result = System_L_dqdqdq(sys, q1, q2, q3);
    if(PyErr_Occurred())
	return NULL;
    else
	return PyFloat_FromDouble(result);
}

static PyObject* L_ddq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;

    if(!PyArg_ParseTuple(args, "O", &dq1))
        return NULL; 
    return PyFloat_FromDouble(System_L_ddq(sys, dq1));
}

static PyObject* L_ddqdq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;
    Config *q2 = NULL;

    if(!PyArg_ParseTuple(args, "OO", &dq1, &q2))
        return NULL; 
    return PyFloat_FromDouble(System_L_ddqdq(sys, dq1, q2));
}

static PyObject* L_ddqdqdq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;

    if(!PyArg_ParseTuple(args, "OOO", &dq1, &q2, &q3))
        return NULL; 
    return PyFloat_FromDouble(System_L_ddqdqdq(sys, dq1, q2, q3));
}

static PyObject* L_ddqdqdqdq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;
    Config *q4 = NULL;

    if(!PyArg_ParseTuple(args, "OOOO", &dq1, &q2, &q3, &q4))
        return NULL; 
    return PyFloat_FromDouble(System_L_ddqdqdqdq(sys, dq1, q2, q3, q4));
}

static PyObject* L_ddqddq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;
    Config *dq2 = NULL;

    if (! PyArg_ParseTuple(args, "OO", &dq1, &dq2))
        return NULL; 
    return PyFloat_FromDouble(System_L_ddqddq(sys, dq1, dq2));
}

static PyObject* L_ddqddqdq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;
    Config *dq2 = NULL;
    Config *q3 = NULL;

    if (! PyArg_ParseTuple(args, "OOO", &dq1, &dq2, &q3))
        return NULL; 
    return PyFloat_FromDouble(System_L_ddqddqdq(sys, dq1, dq2, q3));
}

static PyObject* L_ddqddqdqdq(System *sys, PyObject *args)
{
    Config *dq1 = NULL;
    Config *dq2 = NULL;
    Config *q3 = NULL;
    Config *q4 = NULL;
    double result = 0;

    if (! PyArg_ParseTuple(args, "OOOO", &dq1, &dq2, &q3, &q4))
        return NULL;
    result = System_L_ddqddqdqdq(sys, dq1, dq2, q3, q4);
    if(PyErr_Occurred())
	return NULL;
    else
	return PyFloat_FromDouble(result);
}

static PyObject* update_cache(System *sys, PyObject *args)
{
    unsigned long flags;
    if(!PyArg_ParseTuple(args, "k", &flags))
        return NULL;
    if(flags & SYSTEM_CACHE_LG)
        build_lg_cache(sys);
    if(flags & SYSTEM_CACHE_G)
        build_g_cache(sys);
    if(flags & SYSTEM_CACHE_G_DQ)
        build_g_dq_cache(sys);
    if(flags & SYSTEM_CACHE_G_DQDQ)
        build_g_dqdq_cache(sys);
    if(flags & SYSTEM_CACHE_G_DQDQDQ)
        build_g_dqdqdq_cache(sys);
    if(flags & SYSTEM_CACHE_G_DQDQDQDQ)
        build_g_dqdqdqdq_cache(sys);
    if(flags & SYSTEM_CACHE_G_INV)
        build_g_inv_cache(sys);
    if(flags & SYSTEM_CACHE_G_INV_DQ)
        build_g_inv_dq_cache(sys);
    if(flags & SYSTEM_CACHE_G_INV_DQDQ)
        build_g_inv_dqdq_cache(sys);
    if(flags & SYSTEM_CACHE_VB)
        build_vb_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DQ)
        build_vb_dq_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DQDQ)
        build_vb_dqdq_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DQDQDQ)
        build_vb_dqdqdq_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DDQ)
        build_vb_ddq_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DDQDQ)
        build_vb_ddqdq_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DDQDQDQ)
        build_vb_ddqdqdq_cache(sys);
    if(flags & SYSTEM_CACHE_VB_DDQDQDQDQ)
        build_vb_ddqdqdqdq_cache(sys);
    if(flags & SYSTEM_CACHE_DYNAMICS)
        calc_dynamics(sys);
    if(flags & SYSTEM_CACHE_DYNAMICS_DERIV1)
        calc_dynamics_deriv1(sys);
    if(flags & SYSTEM_CACHE_DYNAMICS_DERIV2)
        calc_dynamics_deriv2(sys);
    Py_RETURN_NONE;
}

static PyMethodDef methods_list[] = {
    {"_total_energy", (PyCFunction)total_energy, METH_NOARGS, trep_internal_doc},
    {"_L", (PyCFunction)L, METH_NOARGS, trep_internal_doc},
    {"_L_dq", (PyCFunction)L_dq, METH_VARARGS, trep_internal_doc},
    {"_L_dqdq", (PyCFunction)L_dqdq, METH_VARARGS, trep_internal_doc},
    {"_L_dqdqdq", (PyCFunction)L_dqdqdq, METH_VARARGS, trep_internal_doc},
    {"_L_ddq", (PyCFunction)L_ddq, METH_VARARGS, trep_internal_doc},
    {"_L_ddqdq", (PyCFunction)L_ddqdq, METH_VARARGS, trep_internal_doc},
    {"_L_ddqdqdq", (PyCFunction)L_ddqdqdq, METH_VARARGS, trep_internal_doc},
    {"_L_ddqdqdqdq", (PyCFunction)L_ddqdqdqdq, METH_VARARGS, trep_internal_doc},
    {"_L_ddqddq", (PyCFunction)L_ddqddq, METH_VARARGS, trep_internal_doc},
    {"_L_ddqddqdq", (PyCFunction)L_ddqddqdq, METH_VARARGS, trep_internal_doc},
    {"_L_ddqddqdqdq", (PyCFunction)L_ddqddqdqdq, METH_VARARGS, trep_internal_doc},

    {"_update_cache", (PyCFunction)update_cache, METH_VARARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};

static PyGetSetDef getset_list[] = {
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_world_frame", T_OBJECT_EX, offsetof(System, world_frame), 0, trep_internal_doc},
    {"_frames", T_OBJECT_EX, offsetof(System, frames), 0, trep_internal_doc},
    {"_configs", T_OBJECT_EX, offsetof(System, configs), 0, trep_internal_doc},
    {"_dyn_configs", T_OBJECT_EX, offsetof(System, dyn_configs), 0, trep_internal_doc},
    {"_kin_configs", T_OBJECT_EX, offsetof(System, kin_configs), 0, trep_internal_doc},
    {"_potentials", T_OBJECT_EX, offsetof(System, potentials), 0, trep_internal_doc},
    {"_forces", T_OBJECT_EX, offsetof(System, forces), 0, trep_internal_doc},
    {"_inputs", T_OBJECT_EX, offsetof(System, inputs), 0, trep_internal_doc},
    {"_constraints", T_OBJECT_EX, offsetof(System, constraints), 0, trep_internal_doc},
    {"_masses", T_OBJECT_EX, offsetof(System, masses), 0, trep_internal_doc},
    {"_time", T_DOUBLE, offsetof(System, time), 0, trep_internal_doc},
    {"_cache", T_ULONG, offsetof(System, cache), 0, trep_internal_doc},

    {"_f", T_OBJECT_EX, offsetof(System, f), 0, trep_internal_doc},
    {"_lambda", T_OBJECT_EX, offsetof(System, lambda), 0, trep_internal_doc},
    {"_D", T_OBJECT_EX, offsetof(System, D), 0, trep_internal_doc},

    {"_M_lu", T_OBJECT_EX, offsetof(System, M_lu), 0, trep_internal_doc}, 
    {"_M_lu_index", T_OBJECT_EX, offsetof(System, M_lu_index), 0, trep_internal_doc},
    {"_A_proj_lu", T_OBJECT_EX, offsetof(System, A_proj_lu), 0, trep_internal_doc},
    {"_A_proj_lu_index", T_OBJECT_EX, offsetof(System, A_proj_lu_index), 0, trep_internal_doc},
    
    {"_Ad", T_OBJECT_EX, offsetof(System, Ad), 0, trep_internal_doc},
    {"_AdT", T_OBJECT_EX, offsetof(System, AdT), 0, trep_internal_doc},
    {"_Ak", T_OBJECT_EX, offsetof(System, Ak), 0, trep_internal_doc},
    {"_Adt", T_OBJECT_EX, offsetof(System, Adt), 0, trep_internal_doc},
    {"_Ad_dq", T_OBJECT_EX, offsetof(System, Ad_dq), 0, trep_internal_doc},
    {"_Ak_dq", T_OBJECT_EX, offsetof(System, Ak_dq), 0, trep_internal_doc},
    {"_Adt_dq", T_OBJECT_EX, offsetof(System, Adt_dq), 0, trep_internal_doc},
    
    {"_D_dq", T_OBJECT_EX, offsetof(System, D_dq), 0, trep_internal_doc},
    {"_D_ddq", T_OBJECT_EX, offsetof(System, D_ddq), 0, trep_internal_doc},
    {"_D_du", T_OBJECT_EX, offsetof(System, D_du), 0, trep_internal_doc},
    {"_D_dk", T_OBJECT_EX, offsetof(System, D_dk), 0, trep_internal_doc},
    {"_f_dq", T_OBJECT_EX, offsetof(System, f_dq), 0, trep_internal_doc},
    {"_f_ddq", T_OBJECT_EX, offsetof(System, f_ddq), 0, trep_internal_doc},
    {"_f_du", T_OBJECT_EX, offsetof(System, f_du), 0, trep_internal_doc},
    {"_f_dk", T_OBJECT_EX, offsetof(System, f_dk), 0, trep_internal_doc},
    {"_lambda_dq", T_OBJECT_EX, offsetof(System, lambda_dq), 0, trep_internal_doc},
    {"_lambda_ddq", T_OBJECT_EX, offsetof(System, lambda_ddq), 0, trep_internal_doc},
    {"_lambda_du", T_OBJECT_EX, offsetof(System, lambda_du), 0, trep_internal_doc},
    {"_lambda_dk", T_OBJECT_EX, offsetof(System, lambda_dk), 0, trep_internal_doc},

    {"_Ad_dqdq", T_OBJECT_EX, offsetof(System, Ad_dqdq), 0, trep_internal_doc},
    {"_Ak_dqdq", T_OBJECT_EX, offsetof(System, Ak_dqdq), 0, trep_internal_doc},
    {"_Adt_dqdq", T_OBJECT_EX, offsetof(System, Adt_dqdq), 0, trep_internal_doc},

    {"_D_dqdq", T_OBJECT_EX, offsetof(System, D_dqdq), 0, trep_internal_doc},
    {"_D_ddqdq", T_OBJECT_EX, offsetof(System, D_ddqdq), 0, trep_internal_doc},
    {"_D_ddqddq", T_OBJECT_EX, offsetof(System, D_ddqddq), 0, trep_internal_doc},
    {"_D_dkdq", T_OBJECT_EX, offsetof(System, D_dkdq), 0, trep_internal_doc},
    {"_D_dudq", T_OBJECT_EX, offsetof(System, D_dudq), 0, trep_internal_doc},
    {"_D_duddq", T_OBJECT_EX, offsetof(System, D_duddq), 0, trep_internal_doc},
    {"_D_dudu", T_OBJECT_EX, offsetof(System, D_dudu), 0, trep_internal_doc},

    {"_f_dqdq", T_OBJECT_EX, offsetof(System, f_dqdq), 0, trep_internal_doc},
    {"_f_ddqdq", T_OBJECT_EX, offsetof(System, f_ddqdq), 0, trep_internal_doc},
    {"_f_ddqddq", T_OBJECT_EX, offsetof(System, f_ddqddq), 0, trep_internal_doc},
    {"_f_dkdq", T_OBJECT_EX, offsetof(System, f_dkdq), 0, trep_internal_doc},
    {"_f_dudq", T_OBJECT_EX, offsetof(System, f_dudq), 0, trep_internal_doc},
    {"_f_duddq", T_OBJECT_EX, offsetof(System, f_duddq), 0, trep_internal_doc},
    {"_f_dudu", T_OBJECT_EX, offsetof(System, f_dudu), 0, trep_internal_doc},
    
    {"_lambda_dqdq", T_OBJECT_EX, offsetof(System, lambda_dqdq), 0, trep_internal_doc},
    {"_lambda_ddqdq", T_OBJECT_EX, offsetof(System, lambda_ddqdq), 0, trep_internal_doc},
    {"_lambda_ddqddq", T_OBJECT_EX, offsetof(System, lambda_ddqddq), 0, trep_internal_doc},
    {"_lambda_dkdq", T_OBJECT_EX, offsetof(System, lambda_dkdq), 0, trep_internal_doc},
    {"_lambda_dudq", T_OBJECT_EX, offsetof(System, lambda_dudq), 0, trep_internal_doc},
    {"_lambda_duddq", T_OBJECT_EX, offsetof(System, lambda_duddq), 0, trep_internal_doc},
    {"_lambda_dudu", T_OBJECT_EX, offsetof(System, lambda_dudu), 0, trep_internal_doc},

    {"_temp_nd", T_OBJECT_EX, offsetof(System, temp_nd), 0, trep_internal_doc},
    {"_temp_ndnc", T_OBJECT_EX, offsetof(System, temp_ndnc), 0, trep_internal_doc},
    
    {"_M_dq", T_OBJECT_EX, offsetof(System, M_dq), 0, trep_internal_doc},
    {"_M_dqdq", T_OBJECT_EX, offsetof(System, M_dqdq), 0, trep_internal_doc},
    
    {NULL}  /* Sentinel */
};

PyTypeObject SystemType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._System",           /*tp_name*/
    sizeof(System),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)dealloc,       /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    trep_internal_doc,         /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    methods_list,              /* tp_methods */
    members_list,              /* tp_members */
    getset_list,               /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};
