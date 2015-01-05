#!c:\users\lenovo\canopy\user\scripts\python.exe
"""
``remove_duplicate_images.py`` is a script that filters out duplicate images
from a dataset.

Needs to:

* Find duplicates (based on similarity: duplicates are over 0.9999 similar)
* Map all duplicates to one image ID
* Change the IDs of all the duplicate images to the representant ID
  * In the vtext-image-map.csv file
* Filter out duplicates from the im.ftrs.csv file

Be careful, this touches a lot of the files that map the dataset structure.

"""
import argparse
import logging
import os
import re
from safire.data.image_browser import ImageBrowser
from safire.data.loaders import MultimodalShardedDatasetLoader, IndexLoader

__author__ = 'Jan Hajic jr.'


profiid_regex = re.compile('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]')


def is_profiid(iid):
    return profiid_regex.match(iid)


##############################################################################


def main(args):
    logging.info('Executing remove_duplicate_images.py...')

    identity_treshold = 0.98
    profiid_identity_treshold = 0.9
    num_best = 200
    if args.profiids_only: # We only want to find a mapping from profiid to wpid
        num_best = 20

    # Load index
    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    iloader = IndexLoader(args.root, args.name)

    index = iloader.load_index(args.index_label)
    index.num_best = num_best # We're guessing there won't be more
    # than 50 duplicates

    dataset = dloader.load_img(args.index_label) # The dataset corresponds
                                                 # to the index.
    icorp = dataset.icorp # For image no. --> image ID string mapping

    image_browser = None
    if args.browser_map is not None:
        logging.info('Building image browser from map %s' % args.browser_map)
        with open(args.browser_map) as image_browser_handle:
            image_browser = ImageBrowser(args.root, image_browser_handle)


    logging.info('Searching for duplicates...')

    imnos_found_identical = set([]) # All image IDs that should be removed.
    imno_mapping = {} # Key: representant, Value: all IDs of identical images
    # (incl. representant)

    # Run similarity queries - populate id_mapping
    for imno in xrange(len(dataset)):

        current_iid = icorp.id2doc[imno]
        if args.profiids_only and not is_profiid(current_iid):
            logging.debug('Skipping non-profiid %s' % current_iid)
            continue

        if imno % 1000 == 0:
            logging.info('Querying for identicals: at imno. %d' % imno)

        # Speedup
        if imno in imnos_found_identical:
            logging.debug('Skipping imno %d, found in identical.' % imno)
            continue

        query = dataset.get_sparse(imno)
        qresults = index[query]

        current_identical_imnos = []
        found_itself = False

        # Profiids get a lower treshold and try to find a non-profiid repr.,
        # may get other special treatment.
        for qr_no, qr_sim in qresults:
            qr_iid = icorp.id2doc[qr_no]
            if is_profiid(qr_iid) != is_profiid(current_iid):
                treshold = profiid_identity_treshold # Across datasets
            else:
                treshold = identity_treshold
            if qr_sim > treshold:
                if qr_no == imno:
                    found_itself = True
                current_identical_imnos.append(qr_no)
            else:
                break # Sorted by similarity...

        # Sanity checks?
        if len(current_identical_imnos) == 0:
            logging.warn('No identical image found for %d, although at least %d itself should have been found.' % (imno, imno))
        if not found_itself:
            logging.warn('Query %d didn\'t find itself as one of the most similar.' % (imno))
            logging.warn('\tTotal: %d\n\tIDs found:\n%s' % (len(current_identical_imnos),
                                                            '\t'.join(map(str, current_identical_imnos))))
            logging.warn('\tMost similar imnos: %s' % ', '.join(map(str, ['/'.join([i,s]) for i,s in qresults ])))
            if image_browser:
                iid = icorp.id2doc[imno]
                duplicate_iids = [ icorp.id2doc[i] for i in current_identical_imnos ]
                image = image_browser.build_tiled_image([iid] + duplicate_iids)
                image.show()


        # If there are duplicates:
        if len(current_identical_imnos) > 1:
            logging.debug('Found duplicates for imno %d: %s' % (imno, str(current_identical_imnos)))

        # Current image number set as representant
        imno_mapping[imno] = current_identical_imnos
        imnos_found_identical = imnos_found_identical.union(set([n
                                                                 for n in current_identical_imnos
                                                                 if n != imno]))
        logging.debug('Imnos found identical status: %s' % str(imnos_found_identical))

        #if image_browser:
        #    iid = icorp.id2doc[imno]
        #    duplicate_iids = [icorp.id2doc[i] for i in current_identical_imnos]
        #    image = image_browser.build_tiled_image([iid] + duplicate_iids)
        #    image.show()



    logging.info('Finished querying index.')
    logging.debug('imno mapping:\n%s' % '\n'.join(map(str, imno_mapping.items())))

    logging.info('Total images that have duplicates: %d' % (
        len([ imno for imno in imno_mapping if len(imno_mapping[imno]) > 1 ])))

    logging.info('Translating imnos to image IDs in id_mapping, imnos_found_identical')
    logging.info('(%d Imnos available in corpus: %s)' % (len(icorp.id2doc), str(icorp.id2doc)))
    iids_found_identical = set([ icorp.id2doc[imno]
                                 for imno in imnos_found_identical ])
    iid_mapping = { icorp.id2doc[imno] : [ icorp.id2doc[i]
                                           for i in imno_mapping[imno] ]
                    for imno in imno_mapping }


    # Potentially: change representants, using some rule?
    if not args.ignore_profiids:
        logging.info('Making sure that representants are identified using web-pic '
                     'ID, not using profiid.')
        for iid in iid_mapping:
            current_iids = iid_mapping[iid]
            if is_profiid(iid):
                # Find first non-profiid, use it as representant
                non_profiids = [ i for i in iid_mapping[iid] if not is_profiid(i) ]
                if len(non_profiids) == 0:
                    logging.warn('Cannot find WP-id for profiid %s. (Total IIDs: %d)' % (iid, len(current_iids)))
                else:
                    new_representant = non_profiids[0]

                    iid_mapping[new_representant] = current_iids
                    del iid_mapping[iid]

                    if new_representant in iids_found_identical:
                        iids_found_identical.remove(new_representant)
                        iids_found_identical.add(iid)

    logging.debug('IID mapping:\n%s' % '\n'.join(map(str, iid_mapping.items())))

    logging.info('Creating reverse dict (IID --> representant IID)')
    reverse_iid_mapping = {}
    for iid in iid_mapping:
        for i in iid_mapping[iid]:
            reverse_iid_mapping[i] = iid


    if args.iid_map:
        logging.info('Writing output map to %s' % args.iid_map)
        with open(args.iid_map, 'w') as iid_map_handle:
            for iid in reverse_iid_mapping:
                representant_iid = reverse_iid_mapping[iid]
                output_line = '\t'.join([iid, representant_iid]) + '\n'
                iid_map_handle.write(output_line)


    # Read vtext-image-map

    if not args.no_transform_t2i:
        t2i_map_file = os.path.join(dloader.root, dloader.layout.textdoc2imdoc)
        t2i_output_map_file = os.path.join(dloader.root,
                                           'noduplicates.' + dloader.layout.textdoc2imdoc)
        logging.info('Processing t2i mapping: input file %s' % t2i_map_file)

        with open(t2i_map_file) as t2i_in_handle:
            with open(t2i_output_map_file, 'w') as t2i_out_handle:
                for t2i_record in t2i_in_handle:
                    text, iid = t2i_record.strip().split()
                    representant_iid = reverse_iid_mapping[iid]
                    t2i_output_record = '\t'.join([text, representant_iid])
                    t2i_out_handle.write(t2i_output_record + '\n')

    # Read im.ftrs.csv
    if not args.no_transform_imftrs:
        imftrs_file = os.path.join(dloader.root, dloader.layout.image_vectors)
        output_imftrs_file = os.path.join(dloader.root,
                                          'noduplicates.' + dloader.layout.image_vectors)
        logging.info('Processing image features: input file %s' % imftrs_file)
        logging.debug('Will skip the following IIDs:\n%s' % '\n'.join(map(str, iids_found_identical)))

        with open(imftrs_file) as imf_in_handle:
            with open(output_imftrs_file, 'w') as imf_out_handle:
                for line in imf_in_handle:
                    split_idx = line.index('\t')
                    iid = line[:split_idx]
                    if iid not in iids_found_identical:
                        imf_out_handle.write(line)
                    else:
                        logging.debug('Skipping image vector for iid %s.' % iid)



    logging.info('Exiting remove_duplicate_images.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory, passed to Loader.')
    parser.add_argument('-n', '--name', required=True,
                        help='The name passed to Loader.')

    parser.add_argument('-i', '--index_label', action='store',
                        help='Specify which index should be used for identity'
                             ' checking.')
    parser.add_argument('-o', '--output_name', action='store',
                        help='Specify under which name the filtered files '
                             'should be stored. [NOT IMPLEMENTED]')
    parser.add_argument('-m', '--iid_map', action='store',
                        help='The file to which the iid --> representant '
                             'mapping will be written (one pair per line, first'
                             ' column iid, second column representant IID).')
    parser.add_argument('-b', '--browser_map', action='store',
                        help='Specify a image ID -> image file map for browsing'
                             ' the identical images. (For testing.)')

    parser.add_argument('--ignore_profiids', action='store_true',
                        help='If set, will not attempt to prefer safire iids'
                             ' to profiid-based iids.')
    parser.add_argument('--profiids_only', action='store_true',
                        help='If set, will only search for identity among '
                             'profiids, speeding up the process.')

    parser.add_argument('--no_transform_t2i', action='store_true',
                        help='If set, will not process vtext-image-map.')
    parser.add_argument('--no_transform_imftrs', action='store_true',
                        help='If set, will not filter image vectors.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO logging messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG logging messages. (May get very '
                             'verbose.')

    return parser


if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    main(args)
