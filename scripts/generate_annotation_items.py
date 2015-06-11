#!/usr/bin/env python
"""
``generate_annotation_items.py`` is a script that generates annotation items
using a certain search strategy from SAFIRE text and image data.

The default search strategy chooses for each text K pictures such that their
similarity to the picture actually used for the text is varied.

The basic algorithm chooses the K/2 most similar images to the input image
and then chooses K/2 more uniformly distributed across the similarity spectrum.

"""
import argparse
import logging
import os
import operator
import random
import safire
from safire.data.loaders import IndexLoader, MultimodalShardedDatasetLoader

__author__ = 'Jan Hajic jr.'


def main(args):
    logging.info('Executing generate_annotation_items.py...')

    # Load index
    logging.info('Loading index with label %s.' % args.index_label)
    iloader = IndexLoader(args.root, args.name)
    index = iloader.load_index(args.index_label)

    index.num_best = None  # NOT working with this

    # Load dataset/corpus
    logging.info('Loading dataset and corpus with label %s.' % args.index_label)
    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    icorp = dloader.load_image_corpus(args.index_label)
    idata = dloader.load_img(args.index_label)

    # Load text-image map
    logging.info('Loading text-image map.')
    t2i_file = os.path.join(args.root, iloader.layout.textdoc2imdoc)
    with open(t2i_file) as t2i_handle:
        text2im, im2text = safire.utils.parse_csv_map(t2i_handle)

    # Load image ID-image file map
    logging.info('Loading iid-image file map.')
    i2f_file = os.path.join(args.root, iloader.layout.imgid2imgfile)
    with open(i2f_file) as i2f_handle:
        iid2imgf, imgf2iid = safire.utils.parse_csv_map(i2f_handle)

    # Parse vtlist
    logging.info('Processing vtlist.')
    vtlist = os.path.join(args.root, iloader.layout.vtlist)
    vtlines = []
    with open(vtlist) as vtlist_handle:
        for vl in vtlist_handle:
            vtlines.append(vl.strip())
    vtlines = vtlines[args.skip_first_vtl:]
    vtlines_cutoff = int(len(vtlines) * args.p_items)
    vtlines_cut = vtlines[:vtlines_cutoff]

    logging.info(' Total vtlines left: %d' % len(vtlines_cut))

    logging.info('Generating annotation items...')

    lines_to_prepend = []
    if args.continue_previous:
        logging.info('Will append new items to previously generated file.')
        prev_annot_inputs = os.path.join(args.root,
                                         iloader.layout.annotation_inputs)
        args.continue_indexing = True
        with open(prev_annot_inputs) as prev_handle:
            lines_to_prepend = [ l.strip() for l in prev_handle.readlines() ]

    if args.continue_indexing:
        logging.info('Will continue indexing from previously generated file.')
        prev_annot_inputs = os.path.join(args.root,
                                         iloader.layout.annotation_inputs)
        if not os.path.exists(prev_annot_inputs):
            raise ValueError('Cannot continue: annotation input file '
                             '%s not found in root %s' % (iloader.layout.annotation_inputs,
                                                          args.root))
        with open(prev_annot_inputs) as prev_handle:
            prev_lines = prev_handle.readlines()
        last_idx = int(prev_lines[-1].split()[0])
        args.first_index = last_idx + 1

    logging.info('Continuing indexing from %d.' % args.first_index)
    annot_item_idx = args.first_index

    # Create annotation items
    annot_items = []
    for text in vtlines_cut:

        iid = text2im[text]
        imno = icorp.doc2id[iid[0]] # Use the first available image only.
        query = idata.get_sparse(imno)

        qresults = index[query]
        results = zip(range(len(qresults)), qresults)
        sorted_results = sorted(results, key=operator.itemgetter(1), reverse=True)

        # Omit original picture?
        p_keep_best = random.uniform(0.0, 1.0)
        if p_keep_best < args.p_original:
            sorted_results = sorted_results[1:]

        # Select top and sampled pictures
        top_results = sorted_results[:args.n_top_imgs]

        sampled_results = []
        n_sampled = args.n_total_imgs - args.n_top_imgs
        if args.n_total_imgs - args.n_top_imgs > 0:
            if args.randomize_steps:
                sampled_results = safire.utils.random_steps(
                    sorted_results[args.n_top_imgs:], n_sampled)
            else:
                sampled_results = safire.utils.uniform_steps(
                    sorted_results[args.n_top_imgs:], n_sampled)

        all_results = top_results + sampled_results

        # Randomize order
        random.shuffle(all_results)

        # Retrieve image files
        imnos = map(operator.itemgetter(0), all_results)
        iids = [ icorp.id2doc[i] for i in imnos ]
        ifnames = [ iid2imgf[ii][0] for ii in iids]

        # Format output
        output_line = ' '.join([
            #str(annot_item_idx),
            str(args.label),
            str(args.priority),
            str(args.prefer_user),
            str(text),
            ';'.join(ifnames)
        ])

        # Print output
        annot_items.append(output_line)

        # Increment item indexer
        annot_item_idx += 1

        if annot_item_idx % 500 == 0:
            logging.info('At annot_item_idx %d' % annot_item_idx)

    # Generate duplicates
    logging.info('Generating duplicates.')

    n_duplicates = int(args.p_duplicates * len(annot_items))
    duplicates = random.sample(annot_items, n_duplicates)

    # Rewrite duplicate IDs!
    # for i, d in enumerate(duplicates):
    #     did = annot_item_idx + i
    #     d_with_correct_id = str(did) + ' ' + d.split()[1:]
    #     annot_items.append(d_with_correct_id)
    annot_items.extend(duplicates)

    # Make sure duplicates are evenly distributed
    random.shuffle(annot_items)

    # Add IDs to the shuffled list
    annot_items_with_id = []
    for i, item in enumerate(annot_items):
        item_id = i + args.first_index
        new_item = str(item_id) + ' ' + item
        annot_items_with_id.append(new_item)
    annot_items = annot_items_with_id

    logging.info('Substituting texts instead of vtexts...')
    annot_items_vt2t = []
    for item in annot_items:
        fields = item.split()
        fields[4] = fields[4].replace('.vt.', '.')
        annot_items_vt2t.append(' '.join(fields))
    annot_items = annot_items_vt2t

    if lines_to_prepend:
        logging.info('Prepending lines from previous annotation inputs file...')
        annot_items = lines_to_prepend + annot_items


    logging.info('Printing annotation items...')

    print '\n'.join(annot_items)

    logging.info('Exiting generate_annotation_items.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                         ' the directory which is the root of a dataset.'+
                         ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the'+
                        ' Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-i', '--index_label', help='The image index label.')

    parser.add_argument('-p', '--priority', type=int, default=0,
                        help='The basic item priority.')
    parser.add_argument('-l', '--label', default='basic',
                        help='The annotation item label.')
    parser.add_argument('-u', '--prefer_user', default='-',
                        help='To which user should the items be preferrably '
                             'assigned.')

    parser.add_argument('-f', '--first_index', default=0,
                        help='The first ID from which to start indexing items.')
    parser.add_argument('--continue_indexing', action='store_true',
                        help='If set, will read the starting annotation ID by '
                             'continuing the annotation_inputs.csv file in the'
                             ' root dir. If the file is not found, an exception'
                             ' is raised.')
    parser.add_argument('-c', '--continue_previous', action='store_true',
                        help='If set, will append new lines to the annotation '
                             'input file already in --root. (Behaves as if '
                             '--continue_indexing was also set.) Raises a'
                             ' ValueError if the annotation file does not '
                             'exist.')

    parser.add_argument('--n_top_imgs', type=int, default=6,
                        help='How many images should be the most similar ones.')
    parser.add_argument('--n_total_imgs', type=int, default=12,
                        help='How many images there should be in one item.')
    parser.add_argument('--randomize_steps', action='store_true',
                        help='If set, will randomly select step sizes in the'
                             ' sampled images instead of a fixed step size.')

    parser.add_argument('--p_items', type=float, default=0.1,
                        help='The proportion of images from the beginning of the'
                             ' vtlist that will be used for annotation.')
    parser.add_argument('--p_duplicates', type=float, default=0.25,
                        help='The proportion of duplicate items in the list. The'
                             'total number of items will be '
                             'n_items / (1 - p_duplicates).')
    parser.add_argument('--p_original', type=float, default=0.5,
                        help='The probability that the original picture will be'
                             ' included with the annotation item, if available.')
    parser.add_argument('--skip_first_vtl', type=int, default=0,
                        help='Start at this index in the input vtlist.')

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
