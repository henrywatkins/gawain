''' physics-specific routines '''

import numpy as np
import sys

def debug(array, name):
    if not np.all(np.isfinite(array)):
        print("inf in "+name)
        sys.exit("error")
        

def EulerFluxX(u):

    """
    thermal_en = (u.en()-0.5*u.momMagSqr()/u.dens())
    pressure = u.adi_minus1()*thermal_en

    x_flux = np.array([u.momX(),
                      u.momX()*u.momX()/u.dens()+pressure,
                      u.momX()*u.momY()/u.dens(),
                      u.momX()*u.momZ()/u.dens(),
                      (u.en() + pressure)*u.momX()/u.dens()])
    """
    dens = u[0]
    momX = u[1]
    momY = u[2]
    momZ = u[3]
    en = u[4]
    adi_minus1 = 0.4
    momMagSqr = momX*momX + momY*momY + momZ*momZ

    thermal_en = (en-0.5*momMagSqr/dens)
    pressure = adi_minus1*thermal_en

    x_flux = np.array([momX,
                      momX*momX/dens+pressure,
                      momX*momY/dens,
                      momX*momZ/dens,
                      (en + pressure)*momX/dens])
    return x_flux

def EulerFluxY(u):
    """
    thermal_en = (u.en()-0.5*u.momMagSqr()/u.dens())
    pressure = u.adi_minus1()*thermal_en

    y_flux = np.array([u.momY(),
                      u.momY()*u.momX()/u.dens(),
                      u.momY()*u.momY()/u.dens()+pressure,
                      u.momY()*u.momZ()/u.dens(),
                      (u.en() + pressure)*u.momY()/u.dens()])
    """
    dens = u[0]
    momX = u[1]
    momY = u[2]
    momZ = u[3]
    en = u[4]
    adi_minus1 = 0.4
    momMagSqr = momX*momX + momY*momY + momZ*momZ

    thermal_en = (en-0.5*momMagSqr/dens)
    pressure = adi_minus1*thermal_en

    y_flux = np.array([momY,
                      momY*momX/dens,
                      momY*momY/dens+pressure,
                      momY*momZ/dens,
                      (en + pressure)*momY/dens])

    return y_flux

def EulerFluxZ(u):
    dens = u[0]
    momX = u[1]
    momY = u[2]
    momZ = u[3]
    en = u[4]
    adi_minus1 = 0.4
    momMagSqr = momX*momX + momY*momY + momZ*momZ

    thermal_en = (en-0.5*momMagSqr/dens)
    pressure = adi_minus1*thermal_en

    z_flux = np.array([momZ,
                      momZ*momX/dens,
                      momZ*momY/dens,
                      momZ*momZ/dens+pressure,
                      (en + pressure)*momZ/dens])
    return z_flux

class FluxCalculator:
    def __init__(self, Parameters):
        (self.dx, self.dy, self.dz) = Parameters.cell_sizes
        self.x_plus_flux = None
        self.x_minus_flux = None
        self.y_plus_flux = None
        self.y_minus_flux = None
        self.z_plus_flux = None
        self.z_minus_flux = None
        self.flux_functionX = EulerFluxX
        self.flux_functionY = EulerFluxY
        self.flux_functionZ = EulerFluxZ

    def _specific_fluxes(self, u, dt):
        pass

    def calculate_rhs(self, u, dt):
        self.x_plus_flux = self.flux_functionX(u.plusX())
        self.x_minus_flux = self.flux_functionX(u.minusX())

        self.y_plus_flux = self.flux_functionY(u.plusY())
        self.y_minus_flux = self.flux_functionY(u.minusY())

        self._specific_fluxes(u, dt)

        total_flux = -(self.y_plus_flux - self.y_minus_flux)/self.dy
        total_flux += -(self.x_plus_flux - self.x_minus_flux)/self.dx
        return total_flux



class HLLFluxer(FluxCalculator):
    """ A MUSCL-Hancock HLL solver using superbee limiter
    """
    def __init__(self, Parameters):
        super(HLLFluxer, self).__init__(Parameters)
        self.boundary_value = Parameters.boundary_value
    
    def smoothness(self, minus, mid, plus):
        d = 0.001
        numer = mid-minus
        numer+=(1.+2.*np.sign(numer))*d 
        denom = plus-mid
        denom+=(1.+2.*np.sign(denom))*d 
        return numer/denom
        
    def superbee(self, r):
        return np.maximum(np.zeros(r.shape), 
                          np.maximum(np.minimum(2*r, np.ones(r.shape)),
                          np.minimum(r, 2*np.ones(r.shape))))
    def vanleer(self, r):
        abs_r = np.absolute(r)
        return (r+abs_r)/(1.0+abs_r)
        
        
    def minmod(self, a, b):
        return np.where(a*b<=0.0, 0.0, np.where(np.absolute(a)<np.absolute(b), a, b))
        
    def calculate_sound_speed(self, u):
        adi_idx = 1.4
        dens = u[0]
        momX = u[1]
        momY = u[2]
        momZ = u[3]
        en = u[4]
        adi_minus1 = adi_idx-1.0
        momMagSqr = momX*momX + momY*momY + momZ*momZ

        thermal_en = (en-0.5*momMagSqr/dens)
        pressure = adi_minus1*thermal_en
        
        return np.sqrt(adi_idx*pressure/dens)
        
    def wave_speeds(self, Ul, Ur):
        
        vel = Ul[1]/Ul[0]
        speed = self.calculate_sound_speed(Ul)
        sl = vel - speed
        vel = Ur[1]/Ur[0]
        speed = self.calculate_sound_speed(Ur)
        sr = vel+speed
        
        return sl, sr #np.minimum(sl, sr), np.maximum(sl, sr)
        
    
    def hll_flux_X(self, Sl, Sr, Ul, Ur):
        
        fl = self.flux_functionX(Ul)
        fr = self.flux_functionX(Ur)
        
        fhll = (Sr*fl - Sl*fr + Sl*Sr*(Ur-Ul))/(Sr-Sl)
        
        return np.where(Sl>=0.0, fl, np.where(Sr<=0.0, fr, fhll))
        
    def MUSCL_Hancock_reconstruction(self, U_left, U_mid, U_right, dt):
        """" returns the states to the rights and lefts of the interfaces
        
        note: this means lefts holds the states for the left of the i+1/2 face,
        rights holds the states for the right of the i-1/2 face.
        """
        #reconstruct
        limited = self.minmod(U_right-U_mid, U_mid-U_left)#self.vanleer(self.smoothness(U_left,U_mid,U_right))
        rights = U_mid - 0.5*limited*(U_mid - U_left)
        lefts = U_mid + 0.5*limited*(U_right - U_mid)
       
        #evolve
        self.x_plus_flux = self.flux_functionX(lefts)
        self.x_minus_flux = self.flux_functionX(rights)
        
        rights = rights + 0.5*dt*(self.x_plus_flux-self.x_minus_flux)/self.dx 
        lefts = lefts + 0.5*dt*(self.x_plus_flux-self.x_minus_flux)/self.dx
        
        return lefts, rights
        
        
    def _specific_fluxes(self, u, dt):
        #MUSCL-Hancock reconstruction
        
        umid = u.centroid()
        uplus = u.plusX()
        uminus = u.minusX()
        
        lefts, rights = self.MUSCL_Hancock_reconstruction(uminus, umid, uplus,dt)
        
        
        #HLL flux calculation
        
        #minus flux calculation
        
        umid = rights
        uminus = np.roll(lefts, -1, axis=1)
        uminus[:,-1] = self.boundary_value[0][1]

        Sl, Sr = self.wave_speeds(uminus, umid)

        self.x_minus_flux = self.hll_flux_X(Sl, Sr, uminus, umid)
        
        
        #plus flux calculation
        
        umid = lefts
        uplus = np.roll(rights, 1, axis=1)
        uplus[:,0] = self.boundary_value[0][0]
        
        Sl, Sr = self.wave_speeds(umid, uplus)
        
        self.x_plus_flux = self.hll_flux_X(Sl, Sr, umid, uplus)
        
        

class LaxFriedrichsFluxer(FluxCalculator):
    def __init__(self, Parameters):
        super(LaxFriedrichsFluxer, self).__init__(Parameters)
    def _specific_fluxes(self, u, dt):

       u1 = u.centroid()

       mid_flux_x = self.flux_functionX(u1)

       u2 = u.minusX()
       self.x_minus_flux = 0.5*((self.x_minus_flux+mid_flux_x)-self.dx*(u1-u2)/dt)

       u2 = u.plusX()
       self.x_plus_flux = 0.5*((self.x_plus_flux+mid_flux_x)-self.dx*(u2-u1)/dt)
       
       mid_flux_y = self.flux_functionY(u1)
       
       u2 = u.minusY()
       self.y_minus_flux = 0.5*((self.y_minus_flux+mid_flux_y)-self.dy*(u1-u2)/dt)
       
       u2 = u.plusY()
       self.y_plus_flux = 0.5*((self.y_plus_flux+mid_flux_y)-self.dy*(u2-u1)/dt)


class LaxWendroffFluxer(FluxCalculator):
    """ using two-step richtmyer method
    """
    def __init__(self, Parameters):
        super(LaxWendroffFluxer, self).__init__(Parameters)
    def _specific_fluxes(self, u, dt):

       u1 = u.centroid()

       mid_flux_x = self.flux_functionX(u1)

       u2 = u.plusX()
       intermediate_plus_x = 0.5*((u1+u2)-dt*(self.x_plus_flux-mid_flux_x)/self.dx) 

       u2 = u.minusX()
       intermediate_minus_x = 0.5*((u1+u2)-dt*(mid_flux_x-self.x_minus_flux)/self.dx)

       self.x_plus_flux = self.flux_functionX(intermediate_plus_x)
       self.x_minus_flux = self.flux_functionX(intermediate_minus_x)
       
       mid_flux_y = self.flux_functionY(u1)

       u2 = u.plusY()
       intermediate_plus_y = 0.5*((u1+u2)-dt*(self.y_plus_flux-mid_flux_y)/self.dy)

       u2 = u.minusY()
       intermediate_minus_y = 0.5*((u1+u2)-dt*(mid_flux_y-self.y_minus_flux)/self.dy)

       self.y_plus_flux = self.flux_functionY(intermediate_plus_y)
       self.y_minus_flux = self.flux_functionY(intermediate_minus_y)

