#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pre-process magpi data

@author: Yifan Mai
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas
from meta import Metadata
from reconstructLSF import reconstruct_lsf
import fit_two_gaussian_from_moffat as mofgauFit
from BPT import bptregion
import copy

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
        self.ha_flux = self.emidata['Ha_F'].data # extension 47
        self.ha_err = self.emidata['Ha_FERR'].data # extension 48
        self.ha_sn = self.ha_flux/self.ha_err
        
        self.data1 = self.fitsdata[1].data
        # datacube is (wavelength, yaxis, xaxis)
        self.flux = self.data1
        
        self.data2 = self.fitsdata[2].data
        self.var = self.data2
        
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
        self.x_del = HD1['CD1_1']
        x_crpix = HD1['CRPIX1']
        self.ny = HD1['NAXIS2']
        self.y_del = HD1['CD2_2']
        y_crpix = HD1['CRPIX2']

        wavelength_crpix = HD1['CRPIX3']
        wavelength_crval = HD1['CRVAL3']
        self.wavelength_del_Wav = HD1['CD3_3']
        self.nw = HD1['NAXIS3']
        
        self.Wavelengths = wavelength_crval + (np.arange(0,self.nw)+1-wavelength_crpix)*self.wavelength_del_Wav
        self.Wavelengths_deredshift = self.Wavelengths/(1+self.magpiredshift)
        # The data is assumed to be de-redshifted 
        # and centred about (0, 0) spatial coordinates.

        self.x_pix_range_sec = (np.arange(0,self.nx)+1-x_crpix)*self.x_del*3600
        self.x_pix_range = self.x_pix_range_sec - (self.x_pix_range_sec[0]+self.x_pix_range_sec[-1])/2

        self.y_pix_range_sec = (np.arange(0,self.ny)+1-y_crpix)*self.y_del*3600
        self.y_pix_range = self.y_pix_range_sec - (self.y_pix_range_sec[0]+self.y_pix_range_sec[-1])/2
        self.flux_scale_factor = 1
        
        # todo 
        self.conti_path = conti_path
        conti_fits = fits.open(conti_path)
        self.conti_spec = conti_fits[3].data - conti_fits[4].data
            
    
    def plot_interg_flux(self,snlim,xlim=None, ylim=None, **kwargs):
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
                                                        **kwargs)
        
        interg_flux = np.nansum(cutoutdata.reshape(int(metadata[0]),int(metadata[1]),int(metadata[2])),axis=2)
        
        ha_sn_cut = self.cutout(data = self.ha_sn,dim=2,xlim=xlim,ylim=ylim)
        
        
        ha_sn_low = ha_sn_cut < snlim
        ha_sn_high = ha_sn_cut >= snlim
        
        
        
        sn_diagram = np.zeros(ha_sn_cut.shape)
        sn_diagram[ha_sn_high] = 1
        
        fig, axs = plt.subplots(1,3)
        
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
                cutoutdata = data[wavelength_mask,:,:][:, y_mask, :][:, :, x_mask]
                
            # I think it's wrong... not need to think about it at this time...
            #if self.wave_axis == 2:
            #    cutoutdata = data[:, x_mask, :][:, :, y_mask][wavelength_mask,:,:]
        if dim == 2:
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            cutoutdata = data[ y_mask, :][ :, x_mask]
        
        
        return cutoutdata
    
    def sn_cut(self,snlim,data,var):
        mask = (self.ha_sn < snlim) | (np.isnan(self.ha_sn))
        data[:,mask] = 0
        var[:,mask] = 0
        return data, var
    
    def cutout_data(self, snlim=None,xlim=None, ylim=None, wavelim=None, scale_flux=False,
                    subtract_continuum=False,mask_center=False,mask_center_pix=None,mask_radius=None,
                    AGN_mask=False,AGN_mask_exp=None,comp_mask=False,niiha_mask=False):
        
        ''' mask_center, mask_center_pix, mask_radius is for mask a circular region at the center
        AGN_mask is mask AGN region based on bpt map
        AGN_mask_exp is expand the AGN mask region, int
        comp_mask is mask composite region based on bpt map
        niiha_mask mask log(nii/ha) > 0.1 , these pixels definitely AGN
        
        
        
        '''
        
        
        
        if subtract_continuum == True:
            if self.conti_path == None:
                print("No continuum available")
                return 0
            flux_subtracted = self.flux - self.conti_spec
        
        flux_mask,var_mask = self.sn_cut(snlim=3, data=flux_subtracted, var=self.var)
        
        if AGN_mask or comp_mask:
            
            ### read data ####
            dmap = self.emidata
            ha = dmap['Ha_F'].data
            ha_err = dmap['Ha_FERR'].data
            ha_snr = ha/ha_err
            
            nii = dmap['NII_6585_F'].data
            nii_err = dmap['NII_6585_FERR'].data
            nii_snr = nii/nii_err
            
            oiii = dmap['OIII_5008_F'].data
            oiii_err = dmap['OIII_5008_FERR'].data
            oiii_snr = oiii/oiii_err
            
            hb = dmap['Hb_F'].data
            hb_err = dmap['Hb_FERR'].data
            hb_snr = hb/hb_err
            
            crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
            crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
            indplot =  crit_err & crit_snr
            
            x = np.log10(nii/ha)
            y = np.log10(oiii/hb)
            
            ##constrction construction coordinates###
            nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
            ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
            xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
            
            x_type = np.full_like(xpos, np.nan)
            y_type = np.full_like(xpos, np.nan)
            x_type[indplot] = x[indplot]
            y_type[indplot] = y[indplot]
            AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
            
            if AGN_mask:
                ## mask AGN region
                flux_mask[:,AGN] = 0
                var_mask[:,AGN] = 0
            if comp_mask:
                ## mask composite region
                flux_mask[:,CP] = 0
                var_mask[:,CP] = 0
            if AGN_mask_exp is not None:
                yyy, xxx = np.meshgrid(np.arange(ha.shape[1]),np.arange(ha.shape[0]))
                AGN_E = copy.copy(AGN)
                for xx, yy in zip(xxx[AGN],yyy[AGN]):
                    AGN_E[xx-AGN_mask_exp:xx+AGN_mask_exp,yy-AGN_mask_exp:yy+AGN_mask_exp] = True
                flux_mask[:,AGN_E] = 0
                var_mask[:,AGN_E] = 0
        
        if niiha_mask:
            ### read data ####
            dmap = self.emidata
            ha = dmap['Ha_F'].data
            ha_err = dmap['Ha_FERR'].data
            ha_snr = ha/ha_err
            
            nii = dmap['NII_6585_F'].data
            nii_err = dmap['NII_6585_FERR'].data
            nii_snr = nii/nii_err
            
            crit_err = (ha_err > 0) & (nii_err > 0)
            crit_snr = (ha_snr > 3) &(nii_snr>3)
            indplot =  crit_err & crit_snr
            log_niiha = np.log10(nii/ha)
            
            niiha_mask_region = (log_niiha > 0.1) & (indplot)
            
            ## mask 
            flux_mask[:,niiha_mask_region] = 0
            var_mask[:,niiha_mask_region] = 0
        
        
        
        if mask_center==True:
            mask_outflow = np.full_like(self.ha_sn,False,dtype=bool)
            total_rows, total_cols = mask_outflow.shape
            #center_row, center_col = total_rows/2, total_cols/2
            integ_flux = np.nansum(flux_mask,axis=0)
            center_row, center_col = np.unravel_index(np.argmax(integ_flux),integ_flux.shape)
            
            if mask_center_pix is not None:
                center_row, center_col = mask_center_pix
            X, Y = np.ogrid[:total_rows, :total_cols]
            dist_from_center = np.sqrt((X - center_row)**2 + (Y-center_col)**2)
            circular_mask = (dist_from_center < mask_radius)
            flux_mask[:,circular_mask] = 0
            var_mask[:,circular_mask] = 0
            
        
        
        
        flux_scale_factor = 1
        if scale_flux is True:
            if self.wave_axis == 0:
                flux_scale_factor = (np.nanmedian(self.flux[:,int(self.ny/2),int(self.nx/2)])*10)
            if self.wave_axis == 2:
                flux_scale_factor = (np.nanmedian(self.flux[int(self.ny/2),int(self.nx/2),:])*10)
        
        
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
        
        
        

        metadata = np.array([cutout_ny,cutout_nx,cutout_nw,
                             -self.x_pix_range[x_mask][0]+1/2*self.x_del*3600,-self.x_pix_range[x_mask][-1]-1/2*self.x_del*3600,
                             self.y_pix_range[y_mask][0]-1/2*self.y_del*3600,self.y_pix_range[y_mask][-1]+1/2*self.y_del*3600,
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
        nx = metadata.naxis[1]
        ny = metadata.naxis[0]
        nw = metadata.naxis[2]
        data3d = data.reshape(ny,nx,nw)
        plt.plot(data3d[ypix,xpix,:])
        plt.show()

    def model_options(self,inc_path,flat_vdisp=True,psfimg=True,band='z',gaussian=2):
        modelfile = open(self.save_path+"MODEL_OPTIONS","w")
        # first line
        modelfile.write('# Specify model options\n')
        # LSFFWHM
        # reconstruct return sigma
        lsf_recon = reconstruct_lsf(wavelengths=6563*(1+self.magpiredshift), 
                        resolution_file=self.dirpath + self.fitsname)
        lsf_deredshift = lsf_recon/(1+self.magpiredshift)
        lsf_fwhm = 2*np.sqrt(2*np.log(2))*lsf_deredshift
        modelfile.write('LSFFWHM\t%.4f\n'%(lsf_fwhm))
        
        if psfimg==True:
            # default z-band
            img = self.fitsdata[4].data[3]
            
            if gaussian==2:
                weight1, weight2, fwhm1, fwhm2 = mofgauFit.psf_img_to_gauss(img)
                
                modelfile.write('PSFWEIGHT\t%f %f\n'%(weight1,weight2))
                modelfile.write('PSFFWHM\t%f %f\n'%(fwhm1,fwhm2))
            if gaussian==3:
                weight1, weight2, weight3, fwhm1, fwhm2 ,fwhm3= mofgauFit.psf_img_to_gauss_three(img)
                
                modelfile.write('PSFWEIGHT\t%f %f %f\n'%(weight1,weight2,weight3))
                modelfile.write('PSFFWHM\t%f %f %f\n'%(fwhm1,fwhm2,fwhm3))
                
                
                
                
        else:
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
        modelfile.write('LINE\t6583.1\t6548.1\t0.333\n')
        if flat_vdisp == True:
            modelfile.write('VDISPN_SIGMA\t1.000000e-09\n')
        
        modelfile.close()
        
    def dn4_options(self):
        modelfile = open(self.save_path+"OPTIONS","w")
        
        
        ha_sn_1d = self.ha_sn.flatten()
        npixel = sum(ha_sn_1d >= 3)
        if npixel <= 300:
            iterations = 5000
        elif npixel <= 400:
            iterations = 7000
        elif npixel <= 500:
            iterations = 9000
        elif npixel <= 800:
            iterations = 11000
        elif npixel <= 1000:
            iterations = 15000
        elif npixel <= 3000:
            iterations = 20000
        
        modelfile.write('# File containing parameters for DNest4\n')
        modelfile.write('# Put comments at the top, or at the end of the line.\n')
        modelfile.write('1	# Number of particles\n')
        modelfile.write('10000	# New level interval\n')
        modelfile.write('10000	# Save interval\n')
        modelfile.write('100	# Thread steps - how many steps each thread should do independently before communication\n')
        modelfile.write('0	# Maximum number of levels\n')
        modelfile.write('10	# Backtracking scale length (lambda in the paper)\n')
        modelfile.write('100	# Strength of effect to force histogram to equal push (beta in the paper)\n')
        modelfile.write('%d	# Maximum number of saves (0 = infinite)\n'%(iterations))
        modelfile.write('sample.txt	# Sample file\n')
        modelfile.write('sample_info.txt	# Sample info file\n')
        modelfile.write('levels.txt	# Sample file\n')
        modelfile.close()
        
        
    def try_sth(self,w):
        print('hello')
        
    def get_flux_scale_factor(self):
        flux_scale_factor_1 = (np.nanmedian(self.flux[:,int(self.ny/2),int(self.nx/2)])*10)
        return flux_scale_factor_1
    
    def get_fwhm_highweight(self):
        #PSF
        psfhdr = self.fitsdata[4].header
        # note that the datacubes have mistake, alpha is beta , beta is alpha
        beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
        alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
        
        weight1, weight2, fwhm1, fwhm2 = mofgauFit.mof_to_gauss(alpha=alpha, 
                                                                beta=beta)
        if weight1 > weight2:
            return fwhm1
        else:
            return fwhm2
        