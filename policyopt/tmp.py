import h5py
def split_h5_name(fullpath, sep='/'):
    '''
    From h5ls.c:
     * Example: ../dir1/foo/bar/baz
     *          \_________/\______/
     *             file       obj
     *
    '''
    sep_inds = [i for i, c in enumerate(fullpath) if c == sep]
    for sep_idx in sep_inds:
        filename, objname = fullpath[:sep_idx], fullpath[sep_idx:]
        if not filename: continue
        # Try to open the file. If it fails, try the next separation point.
        try: 
        	print 'Trying ' + filename
        	h5py.File(filename, 'r').close()
        except IOError: continue
        # It worked!
        return filename, objname
    raise IOError('Could not open HDF5 file/object %s' % fullpath)

filename, key = split_h5_name('training_logs/additiveStatePrior/conditionalThrashing_b20_kt50.h5/snapshots/iter0000800')
print filename, key