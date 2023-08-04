import os
import sys
from glob import glob
import tempfile
import subprocess

import click
import numpy as np
import requests
from fits_align.ident import make_transforms
from fits_align.align import affineremap
from fits2image.conversions import fits_to_jpg
from astropy.io import fits
from astroscrappy import detect_cosmics

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

ARCHIVE_API = 'https://archive-api.lco.global/'
THUMBNAIL_SIZE = (1000,1000)
TOKEN = '4e848389b809061d2bda397fca571cbc0e4a420f'
FRAMES = [
    ("NGC 7293",35749634, 2650312),
    ("IC 2944", 26820906, 2133595),
    ("NGC 104",34766066, 1681806),
    ("NGC 4631", 5520089, 950958),
    ("NGC 4565", 8048391,1436335),
    ("NGC 7479", 12397215, 1894235),
    ("NGC 7814", 0,1403749),
    ("NGC 2456", 0, 2262396),
    ("NGC 869", 34389498, 2237976),
    ("IC 5146", 36712342, 2336668),
    ("NGC 253", 34053511, 2256756)
]

USERNAME = 'egomez'
PROPOSAL = 'LCOEPO2014B-010'
HEADERS = {'Authorization': TOKEN}

def get_recent_requests(match=None):
    if match:
        matchstr = f"&name={match}"
    else:
        matchstr = ""
    url = f"https://observe.lco.global/api/requestgroups/?proposal={PROPOSAL}&state=COMPLETED&user={USERNAME}{matchstr}&limit=200"

    try:
        resp = requests.get(url,headers=HEADERS).json()
        reqnums = [sr['id'] for rq in resp['results'] for sr in rq['requests'] ]
    except Exception as e:
        logger.error(resp['results'])
        raise

    return reqnums

def frames_for_requestnum(reqnum):
    try:
        frames = requests.get(
            '{0}frames/?REQNUM={1}'.format(ARCHIVE_API, reqnum),
            headers=HEADERS
        ).json()['results']
    except Exception as e:
        logger.error(e)
        raise
    if any(f for f in frames if f['RLEVEL'] == 91):
        rlevel = 91
    elif any(f for f in frames if f['RLEVEL'] == 11):
        rlevel = 11
    else:
        rlevel = 0
    frames = [f for f in frames if f['RLEVEL'] == rlevel]
    return rvb_frames(frames)

def get_fits_data(url):
    with fits.open(url) as hdul:
        for hdu in hdul:
            if len(np.shape(hdu)) == 2:
                return hdu.data
            else:
                return None

def make_img_array(imdata, filename):
    rgb_cube = np.dstack(imdata).astype(np.uint8)
    im = Image.fromarray(rgb_cube)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.thumbnail(THUMBNAIL_SIZE, pimage.ANTIALIAS)
    im.save()
    return

def rvb_frames(frames):

    FILTERS = {
        'red': ['R', 'rp','ip'],
        'visual': ['V',],
        'blue': ['B','gp'],
    }

    selected_frames = []
    for color in ['red', 'visual', 'blue']:
        try:
            selected_frames.append(
                next(f['url'] for f in frames if f['FILTER'] in FILTERS[color])
            )
        except StopIteration:
            print('Filters for colour not found')
            return None
    print(f"📷 Found {len(selected_frames)} frames")
    return selected_frames

def reproject_files(ref_image, images_to_align, tmpdir):
    logger.info("Reprojecting data")
    identifications = make_transforms(ref_image, images_to_align[1:3])

    aligned_images = []
    for id in identifications:
        if id.ok:
            aligned_img = affineremap(id.ukn.filepath, id.trans, outdir=tmpdir)
            aligned_images.append(aligned_img)

    img_list = [ref_image]+aligned_images
    if len(img_list) != 3:
        return images_to_align
    return img_list

def sort_files_for_colour(file_list):
    colour_template = {'rp':'1','V':'2','B':'3'}
    colours = {v:k for k,v in colour_template.items()}
    for f in file_list:
        data, hdrs = fits.getdata(f, header=True)
        filtr = hdrs['filter']
        order = colour_template.get(filtr, None)
        if not order and filtr == 'R':
            order = '1'
        elif not order:
            logger.error('{} is not a recognised colour filter'.format(filtr))
            return False
        colours[order] = f
    file_list = [colours[str(i)] for i in range(1,4)]
    assert len(file_list) == 3

    return file_list

def write_clean_data(filelist, in_dir):
    '''
    Overwrite FITS files with cleaned and scaled data
    - Data is read into uncompressed FITS file to remove dependency on FPack
    '''
    img_list =[]
    for i, file_in in enumerate(filelist):
        data, hdrs = fits.getdata(file_in, header=True)
        filtr = hdrs['filter']
        new_filename = os.path.join(in_dir,"{}.fits".format(filtr))
        data = clean_data(data)
        hdu = fits.PrimaryHDU(data, header=hdrs)
        hdu.writeto(new_filename)
        img_list.append(new_filename)

    del hdu, data
    user = hdrs['userid']
    object = hdrs['OBJECT'].replace(' ','')
    reqnum = hdrs['REQNUM']
    width = hdrs['NAXIS1']
    height = hdrs['NAXIS2']
    filename = f"{object}_{user}_{reqnum}.png"
    return img_list, filename, width, height

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
    logger.info('--- Begin CR removal ---')
    # Run astroScrappy to remove pesky cosmic rays
    data = remove_cr(data)
    median = np.median(data)
    data[data<median]=median
    data = np.power(data, 1/2.4)
    logger.debug('Median=%s' % median)
    logger.debug('Max after median=%s' % data.max())
    return data

def make_colour_image(img_list, out_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                img_list = reproject_files(img_list[0], img_list, temp_dir)
            except:
                logger.error("Error aligning")
            img_list_n, filename, width, height = write_clean_data(img_list, temp_dir)
            filename_t = filename.replace('png','tif')
            imgs = sort_files_for_colour(img_list_n)
            # fits_to_jpg(imgs, os.path.join(out_dir,filename), width=width, height=height, color=True)
            subprocess.run(["stiff", "-VERBOSE_TYPE","QUIET","-OUTFILE_NAME", os.path.join(temp_dir,filename_t)]+imgs)
            subprocess.run(["convert",os.path.join(temp_dir,filename_t),os.path.join(out_dir,filename)])
        return filename


def read_reqnums(indir):
    reqnums = []
    files = glob(os.path.join(indir, "*.png"))
    reqnums = [int(f.split('_')[2].split('.')[0]) for f in files]
    return reqnums

@click.command()
@click.option('--in_dir', '-i', help='Input folder')
@click.option('--reqnum', '-r', help='Request Number', type=int)
@click.option('--out_dir', '-o', help='Output folder', required=True)
@click.option('--reqname','-n', help="Request name contains this string")
@click.option('--recent', is_flag=True, help='find all recent colour image requests')
def main(in_dir, reqnum, out_dir, reqname, recent):
    # for n,fid,reqnum in FRAMES:
    #     print(f"✨ Working on {n}")
    # try:
    if in_dir:
        path_match = "*.fits.fz"
        img_list = glob(os.path.join(in_dir, path_match))
        filename = make_colour_image(img_list, out_dir)
        print(f"🌠 Written file {filename}")
    else:
        if recent:
            requestnums = get_recent_requests(reqname)
        else:
            requestnums = [reqnum]
        # Filter out any from a previous run
        old_reqs = read_reqnums(out_dir)
        print(f"Found {len(old_reqs)} existing images")
        print(f"New {len(requestnums)}")
        requestnums = set(requestnums).difference(set(old_reqs))
        print(f"Remaining {len(requestnums)}")
        for rn in requestnums:
            img_list = frames_for_requestnum(rn)
            if not img_list:
                print(f"👾 No colour frames found for {rn}")
                continue
            filename = make_colour_image(img_list, out_dir)
            print(f"🌠 Written file {filename}")

    # except Exception as e:
    #     logger.critical(e)
    #     sys.exit(0)
    return

if __name__ == '__main__':
    main()
