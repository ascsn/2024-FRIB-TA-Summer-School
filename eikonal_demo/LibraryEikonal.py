import numpy as np
import scipy
from scipy import linalg
from scipy.special import jv
from scipy.special import gamma
import scipy.integrate as spi
from scipy.special import gamma
from scipy.special import legendre
from scipy.special import lpmv
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy
from sympy.physics.quantum.cg import CG
import mpmath
import time
import math,cmath
from numba import jit,complex128


global amu,e2,hbarc,alpha
amu=931.5 #MeV/c^2
e2=1.43996518 #e^2/(4pi * epsilonnaught) MeV fm
hbarc=197.32698 #MeV*fm
alpha=1/137


##################################################################
#potential functions
#f woods saxon 1 and 2
#volume terms

def fvol(Vx, Rx, ax, r):
    return Vx / (1 + np.exp((r - Rx) / ax))

#surface terms
def fsurf(Vx, Rx, ax, r):
    return 4 * Vx * np.exp((r - Rx) / ax) / (1 + np.exp((r - Rx) / ax))**2

#optical potentials
 
def optical_potential(r, Vr, RR, aR, Wi, RI, aI, WD, RD, aD):
    return -fvol(Vr, RR, aR, r) - 1j * fvol(Wi, RI, aI, r) - 1j * fsurf(WD, RD, aD, r)

#optical potentials

def VSO(r, j,l,s,Vso, Wso, Rso, aso):
    radfac=np.exp((r-Rso)/aso)/(1+np.exp((r-Rso)/aso))**2/aso/r
    VSO=-(Vso+1j*Wso)*radfac*(j*(j+1)-l*(l+1)-s*(s+1))*((hbarc/139.6)**2.0)
    return VSO

#coulomb point sphere
def VC(Zt, Zp, Rc, r):
    return np.where(r < Rc, (Zt * Zp * e2) * ((3 / 2 - (r**2) / (2 * (Rc**2))) / Rc), (Zt * Zp * e2) * 1 / r)

def GaussianPot(Vr,sigma,r):
    return -Vr*np.exp(-(r/sigma)**2)

def PureCoul(Zt, Zp, r):
    return  (Zt * Zp * e2) * 1 / r
##################################################################
#transferred momentum
  
def q(k, theta):
    return 2 * k * np.sin(theta / 2)

# Define eikonal scattering amplitude
  
def fkfunc(b, chi, theta, k, eta,J_0):
    Xc = 2 * eta * np.log(k * b)
    return -1j * k * b * J_0 * np.exp(1j * Xc) * (np.exp(1j * chi) - 1)
#

def ElScatCS_eik(theta_mesh,bmesh,chi,k,eta):
    fkfunk_mesh=np.zeros(len(theta_mesh),dtype=complex)
    for ix in range(0,len(theta_mesh)):
        theta = theta_mesh[ix]
        besfun=   jv(0, q(k, theta) * bmesh)        
        fkfunk_mesh[ix]= np.trapz(fkfunc(bmesh,chi,theta,k,eta,besfun), bmesh)
    # Compute fcoulomb mesh
    fcoulomb_mesh = fc(theta_mesh, eta, k)
    # Compute ftotal mesh
    ftotal_mesh = fcoulomb_mesh + fkfunk_mesh
    crosssection_mesh = np.abs(ftotal_mesh) ** 2/100 # in barn
    return crosssection_mesh

def EikonalPhase(bmesh,z_mesh,Vr, RR, aR, Wi, RI, aI, WD, RD, aD,Zt, Zp, Rc,v):

    chi=np.zeros(len(bmesh),dtype=complex)
    hz=z_mesh[1]-z_mesh[0]
    for i in range(0,len(bmesh)):
        b=bmesh[i]
        r = np.sqrt(b**2 + z_mesh**2)
        Vpot = optical_potential(r, Vr, RR, aR, Wi, RI, aI, WD, RD, aD) + VC(Zt, Zp, Rc, r) - (Zt * Zp * e2) / r
        Vpot0=optical_potential(b, Vr, RR, aR, Wi, RI, aI, WD, RD, aD) + VC(Zt, Zp, Rc, b) - (Zt * Zp * e2) / b
        chi[i]=(Vpot0+np.sum(2*Vpot*hz)) * (-1 / (hbarc * v))
    return chi


#absorption cross section
def sig_abs_eik(chi,bmesh):
    return np.trapz(bmesh*(1 - abs(np.exp(1j * chi))**2), bmesh) * 2 * np.pi

def sig_el_eik(chi,bmesh):
    return np.trapz(bmesh*abs(1 -np.exp(1j * chi))**2, bmesh) * 2 * np.pi


def sig_tot_eik(chi,bmesh):
    return np.trapz(bmesh*2*(1 - np.exp(1j * chi)).real, bmesh) * 2 * np.pi

##################################################################
# Define coulomb phase shift
def sigma(eta,lmax):
    #Compute coulomb phase shift
    G = gamma(1+1j*eta)
    #Taking the argument of Gamma
    A=G.real
    B=G.imag
    if lmax==0:
        return np.arctan(B/A)
    if lmax>0:
        sig=np.zeros(lmax+1)
        sig[0]=np.arctan(B/A)
        for l in range(1,lmax+1):
            sig[l]=sig[l-1]+np.arctan(eta/l)
        return sig

# Define coulomb function

def fc(theta, eta, k):
    sigma_0 = sigma(eta, 0)    
    fcoulomb = -(eta* np.exp(2 * 1j * ((sigma_0) - eta * np.log(np.sin(theta / 2))))/ (2 * k * (np.sin(theta / 2)) ** 2))
    return fcoulomb
  
def rutherford_cs(theta,eta,k): # in barn
    return (eta / (2 * k) / (np.sin(theta / 2)) ** 2) ** 2/100
#################################################################
# setup kinematics & rescaling radii

def RescalingRadii(Ap,At,rR,rI,rD,rc):
    RR=rR*(Ap**(1/3)+At**(1/3))
    RI=rI*(Ap**(1/3)+At**(1/3))
    RD=rD*(Ap**(1/3)+At**(1/3))
    Rc=rc*(Ap**(1/3)+At**(1/3))
    return RR,RI,RD,Rc

def setup_mesh(bmax,Nb,zmin,zmax,Nz,thmax,hth):
    # Creating a mesh for the impact parameter
    bmesh = np.linspace(0, bmax, Nb)  # 0 to 50 fm
    # bmesh must have a small value approaching zero, otherwise a / 0 will cause an error
    if bmesh[0] == 0:
        bmesh[0] = 0.5
    # Creating a mesh for the integration in z
    z_mesh = np.linspace(zmin, zmax, Nz)
    Nth=int(thmax/hth)
    # Define theta mesh
    theta_mesh = np.linspace(0, thmax, Nth) * np.pi / 180
    theta_mesh[0] = 0.0001 if theta_mesh[0] == 0 else theta_mesh[0]    
    return bmesh,z_mesh,theta_mesh

def setup_kinematics(Ap,At,Ebeam,Zt,Zp):
    mp=Ap*amu
    mt=At*amu
    E= Ebeam*At/(Ap+At) # c-o-m energy
    mu=amu*(At*Ap)/(Ap+At) # reduced mass
    k = np.sqrt((2*mu*E)/(hbarc)**2  )  #wave number
    v= (hbarc*k)/mu #velocity
    eta=(Zt*Zp*e2 / (hbarc*v)) #sommerfeld parameter    
    return E,mu,k,v,eta
#################################################################
