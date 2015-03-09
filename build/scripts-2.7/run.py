#!c:\users\lenovo\canopy\user\scripts\python.exe
"""
``run.py`` is a script that runs a trained text-image system.

The text and image pipelines are loaded layer by layer, then the joint layer
is loaded. The text-to-image joint sampling handle and the backward-pass handles
through the image pipeline are initialized.

::

run.py -t .NAV.pf0.2.pff.tfidf .RSM-2000 .RBM-1000 -i . .DA-1000 -j .RBM-1000

::

The labels for individual pipelines are assembled gradually from the left.
The first layer of the text pipeline will have the infix
``.NAV.pf0.2.pff.tfidf``, the second layer ``.NAV.pf0.2.pff.tfidf.RSM-2000``,
the third ``.NAV.pf0.2.pff.tfidf.RSM-2000.RBM-1000``. For an empty infix, use
``.``; the layers above the preprocessing layer should always have a non-empty
label.

The joint label is constructed by joining the text and image labels by a dash
(``-``) and then joining, also by dash, the joint label. (The text labels come
first.)

To run the baseline system (text similarity -> direct transfer to image ->
image similarity), use the ``--baseline`` flag. The top-level text label will
be used as a text index label.

"""
import argparse
import logging
import operator
import time
from safire.data.loaders import MultimodalShardedDatasetLoader, ModelLoader, \
    IndexLoader
from safire.learning.interfaces import MultimodalClampedSamplerModelHandle, \
    SafireTransformer
from safire.learning.interfaces.model_handle import BackwardModelHandle
from safire.utils.transcorp import dimension, get_transformers, \
    reset_vtcorp_input, bottom_corpus
from safire.utils.transformers import SimilarityTransformer

__author__ = 'Jan Hajic jr.'


def construct_pipeline_labels(labels):
    """Correctly concatenates the labels for one pipeline, to produce their full
    versions.

    :param labels: A list of partial labels (suffixes).

    :return: The list of full labels.
    """
    if labels == []:
        return []

    first_label = labels[0]
    if first_label == '.':
        first_label = ''
    output_labels = [first_label]
    for l in labels[1:]:
        output_labels.append(output_labels[-1] + l)
    return output_labels


def construct_joint_label(text_label, img_label, joint_label):
    """Constructs the label for the joint layer."""
    return text_label + '-' + img_label + '-' + joint_label


def emit_results(results):
    """Writes the query results. The output format is a CSV file with three
    columns: query text name, sorted image IDs and sorted similarities.
    The latter two columns are ``;``-separated."""
    output_lines = []
    for text_file in results:
        r = results[text_file]
        images, similarities = zip(*r)
        img_string = ';'.join(images)
        sim_string = ';'.join(['%.5f' % s for s in similarities])
        line = '\t'.join([text_file, img_string, sim_string])
        output_lines.append(line)
    output = '\n'.join(output_lines)
    return output


###############################################################################


def baseline_run(input_corpus, text_index, image_index, multimodal_dataset,
                 retrieve_num_best=10):
    """Runs the baseline system: find most similar text, get its image,
    find most similar.

    :type multimodal_dataset: safire.data.sharded_multimodal_dataset.ShardedMultimodalDatasest
    :param multimodal_dataset: The multimodal dataset that contains the appropriate
        text-image mapping.
    """
    text_num_best = 100
    text_index.num_best = text_num_best

    outputs = []

    # Retrieve most similar images to original images for most similar text
    n_processed = 0
    start_time = time.clock()

    for bow in input_corpus:
        n_processed += 1
        if n_processed % 100 == 0:
            logging.info('Processed %d items in %d s.' % (n_processed,
                                                          time.clock() - start_time))

        best_texts = text_index[bow]

        qresults = []

        for text, textsim in best_texts:
            # The index built from the vtextcorpus may have some texts
            # that have no images. We need to find the closest text that
            # actually has an associated image.
            try:
                iids = multimodal_dataset.textno2imno(text)
            except KeyError:
                continue

            for iid in iids:
                #logging.info('Text query %d: %s' % (text, str(iid)))
                ibow = multimodal_dataset.img.get_sparse(iid)
                iid_qresults = image_index[ibow]

                #logging.info('Extending query by %s' % str(iid_qresults))
                qresults.extend(iid_qresults)

        result_totals = {}
        #logging.info('QResults: %s' % str(qresults))
        for imno, sim in qresults:
            #logging.info('Imno: %s, sim: %s' % (str(imno), str(sim)))
            if imno in result_totals:
                result_totals[imno] += sim
            else:
                result_totals[imno] = sim

        result_totals_list = result_totals.items()
        sorted_qresults = sorted(result_totals_list, key=operator.itemgetter(1),
                                 reverse=True)

        output = sorted_qresults[:retrieve_num_best]
        outputs.append(output)

    # Convert to IIDs
    icorp = multimodal_dataset.img.icorp
    output_w_iids = []
    for o in outputs:
        o_w_iids = []
        for imno, sim in o:
            iid = icorp.id2doc[imno]
            o_w_iids.append((iid, sim))
        output_w_iids.append(o_w_iids)

    # Convert to tIDs
    input_vtcorp = bottom_corpus(input_corpus)
    tids = [input_vtcorp.id2doc[textno]
            for textno in xrange(len(input_corpus))]

    results = {}
    for i, tid in enumerate(tids):
        results[tid] = output_w_iids[i]

    return results


###############################################################################


def main(args):
    logging.info('Executing run.py...')

    # Construct full labels
    logging.info('Constructing labels.')
    text_labels = construct_pipeline_labels(args.text_labels)
    img_labels = construct_pipeline_labels(args.img_labels)
    joint_label = construct_joint_label(text_labels[-1], img_labels[-1],
                                        args.joint_label)

    # Initialize loaders
    logging.info(
        'Initializing loaders (root %s, name %s).' % (args.root, args.name))
    loader = MultimodalShardedDatasetLoader(args.root, args.name)
    mloader = ModelLoader(args.root, args.name)

    if not args.index_name:
        args.index_name = args.name
    logging.info('Initializing index loader (root %s, name %s).' % (
        args.root, args.index_name))
    iloader = IndexLoader(args.root, args.index_name)

    # Load index
    logging.info('Loading index with label %s' % args.index_label)
    if args.index_label not in img_labels:
        raise ValueError(
            'Index label (%s) does not correspond to any image processing label (%s).' % (
                args.index_label, ', '.join(img_labels)))
    index = iloader.load_index(args.index_label)
    if args.num_best > len(index):
        logging.warn(
            'num_best %d greater than index size %d, setting to index size.' % (
                args.num_best, len(index)))
        args.num_best = len(index)
    index.num_best = args.num_best

    # Load input corpora, derive modality dimensions
    logging.info('Loading input corpora, deriving modality dimensions.')
    logging.info('  Text label:  %s' % text_labels[-1])
    logging.info('  Image label: %s' % img_labels[-1])
    text_processing_corpus = loader.load_text_corpus(text_labels[-1])
    dim_text = dimension(text_processing_corpus)
    img_processing_corpus = loader.load_image_corpus(img_labels[-1])
    dim_img = dimension(img_processing_corpus)
    logging.info('  dim_text: %d, dim_img: %d' % (dim_text, dim_img))

    # At this point, the baseline system code branch branches off.
    if args.baseline:
        logging.info(
            'Running baseline model with text index %s.' % text_labels[-1])
        args.text_index_label = text_labels[-1]  # This is fixed.
        tiloader = IndexLoader(args.root,
                               args.name)  # Text index from training.
        text_index = tiloader.load_text_index(args.text_index_label)
        reset_vtcorp_input(text_processing_corpus, args.input)

        multimodal_dataset = loader.load(text_infix=args.text_index_label,
                                         img_infix=args.index_label)

        results = baseline_run(text_processing_corpus,
                               text_index=text_index, image_index=index,
                               multimodal_dataset=multimodal_dataset)
        results_report = emit_results(results)
        print results_report

        logging.info('Exiting run.py, baseline.')
        return

    # Load joint transformer
    logging.info('Loading joint transformer with label %s' % joint_label)
    joint_transformer = mloader.load_transformer(joint_label)

    # Construct sampling handle & transformer
    logging.info('Constructing joint sampling handle and transformer.')
    joint_sampling_handle = MultimodalClampedSamplerModelHandle.clone(
        joint_transformer.model_handle,
        dim_text=dim_text,
        dim_img=dim_img,
        k=10
    )
    joint_sampling_transformer = SafireTransformer(joint_sampling_handle)

    # Apply image sampling handle
    logging.info('Creating joint sampling corpus.')
    joint_sampling_corpus = joint_sampling_transformer[text_processing_corpus]

    # Construct backward handles & transformers, link them
    logging.info('Constructing backward handles & transformers.')
    index_level = img_labels.index(args.index_label)
    #index_level = args.image_index_levelno
    logging.info('  Index level: %d' % index_level)
    transformers = get_transformers(img_processing_corpus)
    logging.info(' Available transformers: %s' % str(transformers))
    # How to stop at the correct level when there are more transformers
    # in the pipeline?
    # ...the quick fix: an argument that shifts the blame on the user.

    # Transformers now ordered from 0-th level. Need to get handles in reverse.
    backward_image_corpus = joint_sampling_corpus
    for t_idx in xrange(len(transformers) - 1, index_level, -1):
        current_handle = transformers[t_idx].model_handle
        backward_handle = BackwardModelHandle.clone(current_handle)
        backward_transformer = SafireTransformer(backward_handle)

        # Linking the corpus transformations:
        backward_image_corpus = backward_transformer[backward_image_corpus]

    logging.info('Full pipeline construction finished.')
    full_pipeline = backward_image_corpus  # Renaming, for clarity

    # Link input vtlist
    logging.info('Setting vtcorp input to %s' % args.input)
    reset_vtcorp_input(full_pipeline, args.input)

    # Add Similarity transformation.
    similarity = SimilarityTransformer(index=index)
    full_pipeline = similarity[full_pipeline]

    # At this point, the transformation pipeline is set up.
    # We now need to transform the inputs, query the image index

    # Run transformations on text inputs
    logging.info('Running transformation on text inputs.')
    outputs = [img_features for img_features in full_pipeline]

    # Query images
    logging.info('Querying image index.')
    query_results = [index[query] for query in outputs]

    # Map results to image files
    logging.info('Mapping results to image files...')
    icorp = bottom_corpus(img_processing_corpus)
    fnamed_query_results = []
    for query_result in query_results:
        fq = [(icorp.id2doc[iid], sim) for iid, sim in query_result]
        fnamed_query_results.append(fq)

    logging.info('Mapping query documents to text files...')
    vtcorp = bottom_corpus(text_processing_corpus)
    text_files = [vtcorp.id2doc[i] for i in xrange(len(query_results))]

    logging.info('Building results data structure.')
    results = {t: q for t, q in zip(text_files, fnamed_query_results)}

    # Emit results to CSV
    logging.info('Emitting results.')
    output = emit_results(results)
    print output

    logging.info('Exiting run.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory, passed to Loader.')
    parser.add_argument('-n', '--name', required=True,
                        help='The name passed to Loader.')

    parser.add_argument('--input', required=True,
                        help='The *.vtlist file (with full path) that specifies'
                             ' the vtext files to use in testing.')
    parser.add_argument('--num_best', type=int, default=None,
                        help='How many top images should be in the output. If '
                             'left to None (default), will not sort the results'
                             ' and output similarities for all images (this may'
                             ' be a very large number of outputs - thousands '
                             'per query document).')

    parser.add_argument('-t', '--text_labels', required=True, nargs='+',
                        help='The text label suffixes for individual layers of'
                             ' the text processing pipeline.')
    parser.add_argument('-i', '--img_labels', required=True, nargs='+',
                        help='The image label suffixes for individual layers of'
                             ' the image processing pipeline.')
    parser.add_argument('-j', '--joint_label', required=False, default='',
                        help='The joint layer label suffix. Required unless '
                             'the --baseline system is requested.')

    parser.add_argument('-b', '--baseline', action='store_true',
                        help='If given, will run the baseline retrieval system:'
                             ' search for closest text, get images originally '
                             'associated with closest text, get images closest'
                             ' to these originals.')
    parser.add_argument('--text_index_label', action='store',
                        help='The text index label for the baseline system. Has'
                             ' to match the top-level text transformation '
                             'label. (The text index name always matches the'
                             ' model name - we are only using training texts.)')

    parser.add_argument('--index_name',
                        help='If the index label used for getting retrieval '
                             'results should be different from the name used'
                             ' for loading the models, supply it through this'
                             ' argument and a separate loader will be created'
                             ' for the index. This is to facilliate retrieving'
                             ' evaluation images that were not in the training'
                             ' set and thus are not a part of the index with'
                             ' the --name name.')
    parser.add_argument('-x', '--index_label', default='',
                        help='The image index label. Has to correspond to '
                             'an image label at a level in the processing '
                             'pipeline to which it is possible to backtrack.')
    parser.add_argument('--image_index_levelno', type=int, default=None,
                        help='Will use the top transformer of the i.i.l.-th '
                             'image_label from the command line as the cutoff'
                             ' for the backward activaiton.')

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
