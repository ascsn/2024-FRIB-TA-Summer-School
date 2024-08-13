#include <iostream>
#include<math.h>

using namespace std;
int main(int argc, char **argv)
{
//%%%%%%%%%%%%%% Computation of the KD potential 1keV--200MeV 24<=A<=209

//%% Parameters of  the collision
int A,  Z, N;
double E,En,v1n,v2n,v3n,v4n, alpha,w1n,w2n,d1n,d2n,d3n,vnso1,vnso2,wnso1,wnso2;
double VR,rR,aR,WI,WD,rD,aD,VSO,WSO,rso,aso;

  if (argc != 4) {
      std::cerr << "Usage: " << argv[0] << " <target mass A> <Target proton number Z> <Projectile energy in Lab (MeV)>" << std::endl;
      return EXIT_FAILURE;
   }

A=atoi(argv[1]);
Z=atoi(argv[2]);
E=atoi(argv[3]);
N=A-Z;

//%% Parameters of KD
En=-11.2814+0.02646*A;

v1n=59.3-21*((N-Z)/A)-0.024*A;
v2n=0.007228-1.48*1e-6*A;
v3n=1.994*1e-5-2*1e-8*A;
v4n=7*1e-9;
alpha=(N-Z)/A;

w1n=12.195+0.0167*A;
w2n=73.55+0.0795*A;

d1n=16-16*(N-Z)/A;
d2n=0.018+0.003802/(1+exp((A-156)/8));
d3n=11.5;

vnso1=5.922+0.003*A;
vnso2=0.004;
wnso1=-3.1;
wnso2=160;


//%% potentials
VR=v1n*(1-v2n*(E-En)+v3n*pow((E-En),2.)-v4n*pow((E-En),3.));
rR=(1.3039-0.4054*pow(A,(-1./3.)))*pow(A,(1./3.));
aR=0.6778-1.487*1e-4*A;

WI=w1n*pow((E-En),2)/(pow((E-En),2)+pow(w2n,2));

WD=(16-16*alpha)*pow((E-En),2)/(pow((E-En),2)+pow(d3n,2))*exp(-d2n*(E-En));
rD=(1.3424-0.01585*pow(A,(1./3.)))*pow(A,(1./3.));
aD=0.5446-1.656*1e-4*A;

VSO=vnso1*exp(-vnso2*(E-En));
WSO=wnso1*pow((E-En),2)/(pow((E-En),2)+pow(wnso2,2));
rso=(1.1854-0.647*pow(A,(-1./3.)))*pow(A,(1./3.));
aso=0.59;
//sprintf('real vol pot')
cout<<"real vol pot"<<endl;
cout<<"VR="<<VR<< " rR="<<rR<<" aR="<<aR<<endl;
//sprintf('Imaginary vol pot')
cout<<"Imaginary vol pot"<<endl;
cout<< "WI="<<WI<<" rR="<<rR<< " aR="<<aR<<endl;
//sprintf('Imaginary surface pot')
cout<<"Imaginary surface pot"<<endl;
cout<<"WD="<<WD<<"  rD="<<rD<<"  aD="<<aD<<endl;
//sprintf('Real SO')
cout<<"Real spin-orbit potential" << endl;
cout<<"VSO="<<VSO<<"  rso="<<rso<<"  aso="<<aso<<endl;
//display('Imaginary SO')
//WSO,rso,aso

return 0;
}
