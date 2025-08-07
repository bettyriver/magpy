#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:12:47 2022

backup of preblobby3d before fix the x, y bug....QAQ Oh My God!

@author: ymai0110
"""



from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas
from meta import Metadata
from reconstructLSF import reconstruct_lsf
import fit_two_gaussian_from_moffat as mofgauFit

class PreBlobby3D:
    
    def __init__(
            self, dirpath, fitsname, emipath, eminame, redshift_path, save_path,
            conti_path=None, wave_axis=0):
        """
        

        Parameters
        ----------
        dirpath : TYPE
            DESCRIPTION.
        fitsname : TYPE
            DESCRIPTION.
        redshift_path : TYPE
            DESCRIPTION.
        save_path : TYPE
            DESCRIPTION.
        wave_axis : TYPE, optional
            DESCRIPTION. The default is 0.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.dirpath = dirpath
        self.fitsname = fitsname
        self.fitsdata = fits.open(dirpath + fitsname)
        self.emidata = fits.open(emipath+eminame)
        self.ha_flux = np.transpose(self.emidata['Ha_F'].data,axes=(1,0)) # extension 47
        self.ha_err = np.transpose(self.emidata['Ha_FERR'].data,axes=(1,0)) # extension 48
        self.ha_sn = self.ha_flux/self.ha_err
        
        self.data1 = self.fitsdata[1].data
        # datacube is (wavelength, yaxis, xaxis), use transpose to have 
        # (wavelength, xaxis, yaxis)
        self.flux = np.transpose(self.data1,axes=(0,2,1))
        
        self.data2 = self.fitsdata[2].data
        self.var = np.transpose(self.data2,axes=(0,2,1))
        
        self.cubeshape = self.flux.shape
        self.magpiid = fitsname[5:15]
        self.wave_axis = wave_axis
        self.nw = self.cubeshape[wave_axis]
        self.save_path = save_path
        #if wave_axis == 0:
        #    self.nx , self.ny = self.cubeshape[1:]
        #elif wave_axis == 2:
        #    self.nx , self.ny = self.cubeshape[:2]
        #else:
        #    raise ValueError('Wave axis needs to be 0 or 2.')
            
        redshift_data =  pandas.read_csv(redshift_path)
        index = np.where(redshift_data['MAGPIID']==int(self.magpiid))[0][0]
        self.magpiredshift = redshift_data['Z'][index]
        
        self.HD1 = self.fitsdata[1].header
        HD1 = self.HD1
        self.nx = HD1['NAXIS1']
        x_del = HD1['CD1_1']
        x_crpix = HD1['CRPIX1']
        self.ny = HD1['NAXIS2']
        y_del = HD1['CD2_2']
        y_crpix = HD1['CRPIX2']

        wavelength_crpix = HD1['CRPIX3']
        wavelength_crval = HD1['CRVAL3']
        self.wavelength_del_Wav = HD1['CD3_3']
        self.nw = HD1['NAXIS3']
        
        self.Wavelengths = wavelength_crval + (np.arange(0,self.nw)+1-wavelength_crpix)*self.wavelength_del_Wav
        self.Wavelengths_deredshift = self.Wavelengths/(1+self.magpiredshift)
        # The data is assumed to be de-redshifted 
        # and centred about (0, 0) spatial coordinates.

        self.x_pix_range_sec = (np.arange(0,self.nx)+1-x_crpix)*x_del*3600
        self.x_pix_range = self.x_pix_range_sec - (self.x_pix_range_sec[0]+self.x_pix_range_sec[-1])/2

        self.y_pix_range_sec = (np.arange(0,self.ny)+1-y_crpix)*y_del*3600
        self.y_pix_range = self.y_pix_range_sec - (self.y_pix_range_sec[0]+self.y_pix_range_sec[-1])/2
        self.flux_scale_factor = 1
        
        # todo 
        #conti_fits = fits.open(conti_path)
        
            
    
    def plot_interg_flux(self,snlim,xlim=None, ylim=None, wavelim=None,
                         scale_flux=False,subtract_continuum=False):
        """
        

        Parameters
        ----------
        xlim : TYPE, optional
            DESCRIPTION. The default is None.
        ylim : TYPE, optional
            DESCRIPTION. The default is None.
        wavelim : TYPE, optional
            deredshifted wavelength range. The default is None.

        Returns
        -------
        None.

        """
        cutoutdata, cutoutvar, metadata = self.cutout_data(xlim=xlim, ylim=ylim,
                                                        wavelim=wavelim, 
                                                        scale_flux=scale_flux,
                                                        subtract_continuum=subtract_continuum)
        
        interg_flux = np.nansum(cutoutdata.reshape(int(metadata[0]),int(metadata[1]),int(metadata[2])),axis=2)
        
        ha_sn_cut = self.cutout(data = self.ha_sn,dim=2,xlim=xlim,ylim=ylim)
        
        
        ha_sn_low = ha_sn_cut < snlim
        ha_sn_high = ha_sn_cut >= snlim
        
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        
        
        
        if xlim is not None:
            x_values = np.arange(self.nx)
            x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
        if ylim is not None:
            y_values = np.arange(self.ny)
            y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
        
        xx = np.arange(self.nx)[x_mask]
        yy = np.arange(self.ny)[y_mask]
        
        xpos,ypos = np.meshgrid(xx,yy)
        xpos = xpos.T
        ypos = ypos.T
        
        sn_diagram = np.zeros(ha_sn_cut.shape)
        sn_diagram[ha_sn_high] = 1
        
        fig, axs = plt.subplots(1,3)
        print(xpos.shape)
        print(ha_sn_cut.shape)
        fig.suptitle(self.magpiid)
        axs[0].imshow(interg_flux)
        axs[0].set_title('interg_flux')
        
        
        im1 = axs[1].imshow(ha_sn_cut)
        axs[1].set_title('Halpha_sn')
        
        #fig.colorbar(im1)
        axs[2].imshow(sn_diagram)
        axs[2].set_title('Halpha_sn > ' + str(snlim))
        
        
        plt.show()
        return cutoutdata, cutoutvar, metadata
    
    def cutout(self,data,dim, xlim=None, ylim=None, wavelim=None):
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        if dim == 3:
            wavelength_mask = np.ones(self.nw,dtype=bool)
            
            
            
            if wavelim is not None:
                wavelength_mask = (self.Wavelengths_deredshift > wavelim[0]) & (self.Wavelengths_deredshift < wavelim[1])
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            if self.wave_axis == 0:
                cutoutdata = data[wavelength_mask,:,:][:, x_mask, :][:, :, y_mask]
                
            if self.wave_axis == 2:
                cutoutdata = data[:, x_mask, :][:, :, y_mask][wavelength_mask,:,:]
        if dim == 2:
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            cutoutdata = data[ x_mask, :][ :, y_mask]
        
        
        return cutoutdata
    
    def sn_cut(self,snlim,data,var):
        mask = (self.ha_sn < snlim) | (np.isnan(self.ha_sn))
        data[:,mask] = 0
        var[:,mask] = 0
        return data, var
    
    def cutout_data(self, snlim=None,xlim=None, ylim=None, wavelim=None, scale_flux=False,
                    subtract_continuum=False):
        
        flux_mask,var_mask = self.sn_cut(snlim=3, data=self.flux, var=self.var)
        
        
        flux_scale_factor = 1
        if scale_flux is True:
            if self.wave_axis == 0:
                flux_scale_factor = (np.nanmedian(self.flux[:,int(self.nx/2),int(self.ny/2)])*10)
            if self.wave_axis == 2:
                flux_scale_factor = (np.nanmedian(self.flux[int(self.nx/2),int(self.ny/2),:])*10)
        
        
        cutoutdata = self.cutout(data=flux_mask,dim=3,xlim=xlim,ylim=ylim,wavelim=wavelim)
        cutoutvar = self.cutout(data=var_mask,dim=3,xlim=xlim,ylim=ylim,wavelim=wavelim)
        cutoutdata = cutoutdata/flux_scale_factor
        cutoutvar = cutoutvar/flux_scale_factor**2
        
        
        wavelength_mask = np.ones(self.nw,dtype=bool)
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        
        
        if wavelim is not None:
            wavelength_mask = (self.Wavelengths_deredshift > wavelim[0]) & (self.Wavelengths_deredshift < wavelim[1])
        if xlim is not None:
            x_values = np.arange(self.nx)
            x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
        if ylim is not None:
            y_values = np.arange(self.ny)
            y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
        
        
        cutout_nx = np.sum(x_mask)
        cutout_ny = np.sum(y_mask)
        cutout_nw = np.sum(wavelength_mask)
        
        if self.wave_axis == 0:
            data_for_save = cutoutdata.reshape(cutout_nw,cutout_nx*cutout_ny).T
            var_for_save = cutoutvar.reshape(cutout_nw,cutout_nx*cutout_ny).T
        if self.wave_axis == 2:
            data_for_save = cutoutdata.reshape(cutout_nx*cutout_ny,cutout_nw)
            var_for_save = cutoutvar.reshape(cutout_nx*cutout_ny,cutout_nw)
        
        #if subtract_continuum == True:
        #    for i in range(data_for_save.shape[0]):
        #        data_for_save[i] = self.subtract_continuum_spaxel(data_for_save[i])
        
        
        

        metadata = np.array([cutout_nx,cutout_ny,cutout_nw,
                             -self.x_pix_range[x_mask][0],-self.x_pix_range[x_mask][-1],
                             self.y_pix_range[y_mask][0],self.y_pix_range[y_mask][-1],
                             self.Wavelengths_deredshift[wavelength_mask][0]-1/2*self.wavelength_del_Wav/(1+self.magpiredshift),
                             self.Wavelengths_deredshift[wavelength_mask][-1]+1/2*self.wavelength_del_Wav/(1+self.magpiredshift)])
        
        return data_for_save, var_for_save, metadata
        
    
    
    def subtract_continuum_spaxel(self, spectrum):
        """simple fit of continuum and subtract the continuum from a spectrum
        
        Parameters
        ----------
        spectrum : array-like
        
        
        Returns
        -------
        spectrum_subtracted : np.array
        
        """
        wave = np.arange(len(spectrum))
        wave_to_fit = np.concatenate((wave[:5],wave[-5:]))
        spec_to_fit = np.concatenate((spectrum[:5],spectrum[-5:]))
        coefficients= np.polyfit(wave_to_fit,spec_to_fit,deg=1)
        poly = np.poly1d(coefficients)
        straight_line_values = poly(wave)
        spectrum_subtracted = spectrum - straight_line_values
        return spectrum_subtracted
    
    def savedata(self,data,var,metadata):
        

        metafile = open(self.save_path+"metadata.txt","w")
        metafile.write("%d %d %d %.3f %.3f %.3f %.3f %.3f %.3f"%(metadata[0],metadata[1],
                                                                 metadata[2],metadata[3],
                                                                 metadata[4],metadata[5],
                                                                 metadata[6],metadata[7],
                                                                 metadata[8]))

        metafile.close()

        np.savetxt(self.save_path+"data.txt",np.nan_to_num(data))
        np.savetxt(self.save_path+"var.txt",np.nan_to_num(var))
    
    def plot_txt(self,xpix,ypix):
        metadata = Metadata(self.save_path+"metadata.txt")
        data = np.loadtxt(self.save_path+"data.txt")
        var = np.loadtxt(self.save_path+"var.txt")
        nx = metadata.naxis[0]
        ny = metadata.naxis[1]
        nw = metadata.naxis[2]
        data3d = data.reshape(nx,ny,nw)
        plt.plot(data3d[xpix,ypix,:])
        plt.show()

    def model_options(self,inc_path):
        modelfile = open(self.save_path+"MODEL_OPTIONS","w")
        # first line
        modelfile.write('# Specify model options\n')
        # LSFFWHM
        lsf_recon = reconstruct_lsf(wavelengths=6563*(1+self.magpiredshift), 
                        resolution_file=self.dirpath + self.fitsname)
        lsf_deredshift = lsf_recon/(1+self.magpiredshift)
        modelfile.write('LSFFWHM\t%.3f\n'%(lsf_deredshift))
        
        #PSF
        psfhdr = self.fitsdata[4].header
        # note that the datacubes have mistake, alpha is beta , beta is alpha
        beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
        alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
        
        weight1, weight2, fwhm1, fwhm2 = mofgauFit.mof_to_gauss(alpha=alpha, 
                                                                beta=beta)
        modelfile.write('PSFWEIGHT\t%f %f\n'%(weight1,weight2))
        modelfile.write('PSFFWHM\t%f %f\n'%(fwhm1,fwhm2))
        
        # inclination
        inc_data =  pandas.read_csv(inc_path)
        index = np.where(inc_data['MAGPIID']==int(self.magpiid))[0][0]
        semimaj = inc_data['semimaj'][index]
        semimin = inc_data['semimin'][index]
        inc = np.arccos(semimin/semimaj)
        
        modelfile.write('INC\t%f\n'%(inc))
        modelfile.write('LINE\t6562.81\n')
        modelfile.write('LINE\t6583.1\t6548.1\t0.333')
        
        modelfile.close()
        
