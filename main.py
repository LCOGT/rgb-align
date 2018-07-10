from lco_alipy.ident import run
from lco_alipy.align import affineremap
from fits2image.conversions import fits_to_jpg
import click
from glob import glob
import os
from astropy.io import fits
import numpy as np
from astroscrappy import detect_cosmics

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def reproject_files(ref_image, images_to_align, tmpdir='temp/'):
    identifications = run(ref_image, images_to_align[1:3], visu=False)
    hdu = fits.open(ref_image)
    data = hdu[1].data
    outputshape = np.shape(data)

    for id in identifications:
        if id.ok:
            affineremap(id.ukn.filepath, id.trans, shape=(outputshape[1],outputshape[0]), outdir=tmpdir)

    aligned_images = sorted(glob(tmpdir+"/*_affineremap.fits"))

    img_list = [ref_image]+aligned_images

    return img_list

def sort_files_for_colour(file_list):
    colour_template = {'rp':'1','V':'2','B':'3'}
    colours = {v:k for k,v in colour_template.items()}
    for f in file_list:
        data, hdrs = fits.getdata(f, header=True)
        filtr = hdrs['filter']
        order = colour_template.get(filtr, None)
        if not order:
            logger.debug('{} is not a recognised colour filter'.format(filtr))
            return False
        colours[order] = f
    file_list = [colours[str(i)] for i in range(1,4)]
    assert len(file_list) == 3

    return file_list

def write_clean_data(filelist):
    '''
    Overwrite FITS files with cleaned and scaled data
    - Data is read into uncompressed FITS file to remove dependency on FPack
    '''
    img_list =[]
    for i, file_in in enumerate(filelist):
        data, hdrs = fits.getdata(file_in, header=True)
        filtr = hdrs['filter']
        path = os.path.split(file_in)[0]
        new_filename = os.path.join(path,"{}.fits".format(filtr))
        data = clean_data(data)
        hdu = fits.PrimaryHDU(data, header=hdrs)
        hdu.writeto(new_filename)
        img_list.append(new_filename)

    return img_list

def remove_cr(data):
    '''
    Removes high value pixels which are presumed to be cosmic ray hits.
    '''
    m, imdata = detect_cosmics(data, readnoise=20., gain=1.4, sigclip=5., sigfrac=.5, objlim=6.)
    return imdata

def clean_data(data):
    '''
    - Remove bogus (i.e. negative) pixels
    - Remove Cosmic Rays
    - Subtract the median sky value
    '''
    # Level out the colour balance in the frames
    logger.debug('--- Begin CR removal ---')
    median = np.median(data)
    data[data<0.]=median
    # Run astroScrappy to remove pesky cosmic rays
    data = remove_cr(data)
    logger.debug('Median=%s' % median)
    logger.debug('Max after median=%s' % data.max())
    return data

@click.command()
@click.option('--in_dir', '-i', help='Input folder')
@click.option("--name", "-n", help="Name of the output file")
def main(in_dir, name):
    path_match = "*.fits.fz"
    img_list = sorted(glob(os.path.join(in_dir, path_match)))
    # img_list = reproject_files(img_list[0], img_list, in_dir)
    # img_list = write_clean_data(img_list)
    img_list = sort_files_for_colour(img_list)
    fits_to_jpg(img_list, os.path.join(in_dir,name), width=1000, height=1000, color=True)
    return

if __name__ == '__main__':
    main()
