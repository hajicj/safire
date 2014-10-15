#!/usr/bin/env python
"""
``annotation_stats.py`` is a script that processes annotation results and
compiles statistics about the annotations so far.

In total:
- # annotated items
- # annotated appropriate, average
- # annotated inappropriate, average
- are original images tagged as appropriate?

For each annotator:
- same as in total
- agreement with other annotators

If the ``--t2i`` option is given, will generate a vtext-image map to
the given file.

* By default, it only outputs text-image pairs for appropriate
  images on which annotators agreed.
* If the ``--relaxed`` option is given, will
  output all images that were tagged by at least one annotator (incl. items
  which were seen by only one annotator).
* If the ``--with_solo`` option is given,
  will output the images tagged only by one annotator, but from duplicate items,
  will only output images tagged by both.

"""
import argparse
from collections import defaultdict
import copy
import logging
import operator
import datetime
import itertools
from matplotlib import pyplot as plt
from safire.utils import squish_array
from safire.utils.matutils import avg_pairwise_mutual_precision, avg, precision, \
    f_score, kappa, recall

__author__ = 'Jan Hajic jr.'

#: Annotation outputs column names.
ao_column_names = [ 'id',
                 'user',
                 'time',
                 'skipped',
                 'appropriate',
                 'inappropriate' ]

ao_array_names = [ 'images', 'appropriate', 'inappropriate', 'unannotated' ]
ao_array_delimiter = ';'

#: Annotation inputs column names.
ai_column_names = ['id',
                   'label',
                   'priority',
                   'prefer_user',
                   'text',
                   'images']
ai_array_names = [ 'images', 'combined' ]
ai_array_delimiter = ';'

#: Combined output column names and ordering
c_column_names = [ 'id', 'label', 'priority',
                   'prefer_user', 'text', 'images', 'user', 'time', 'skipped',
                   'appropriate', 'inappropriate', 'combined', 'unannotated']
c_array_names = [ 'images', 'appropriate', 'inappropriate', 'combined',
                  'unannotated' ]
c_array_delimiter = ';'

report_delimiter = '\t'

##############################################################################


def parse_annot_item(line, column_names, array_names, array_delimiter=';',
                     delimiter='\t'):
    """
    Parses one annotation item line into the item dict.

    :param line:

    :return: dict with ``column_names`` keys
    """
    item = {}

    # Prepare item structure
    for name in column_names:
        if name in array_names:
            item[name] = []
        else:
            item[name] = None

    for value, name in zip(line.split(delimiter), column_names):
        if name in array_names:
            values = value.split(array_delimiter)
            item[name] = values
        else:
            item[name] = value
    return item


def parse_annotations(filename):
    """
    Parses the annotation file and returns the annotations data structure.

    :param filename: The ``annotation_outputs.csv`` file. The expected format
        is tab-separated CSV, with columns ID, Label, Preferred user, etc.

    :return: An array of dicts.
    """
    with open(filename) as input_handle:
        annotations = [ parse_annot_item(l.strip(), ao_column_names,
                                         ao_array_names, ao_array_delimiter,
                                         delimiter='\t')
                        for l in input_handle ]

    return annotations

def parse_annotation_inputs(filename):
    with open(filename) as input_handle:
        annotations = [ parse_annot_item(l.strip(), ai_column_names,
                                         ai_array_names, ai_array_delimiter,
                                         delimiter=' ')
                        for l in input_handle ]

    return annotations


def format_annotations(annotations, columns=c_column_names, delimiter='\t',
                       array_delimiter=c_array_delimiter):
    """Generates the string to write to an annotations file.
    If the ``columns`` parameter is ``None``, will write all columns.

    Will follow the column ordering.
    """
    if not columns:
        return None # No output

    output_lines = []
    for a in annotations:
        fields = []
        for c in columns:
            if c not in a:
                raise ValueError('Column %s not found in annotations (available: %s)' % (
                    c, str(a.keys())))
            if c in c_array_names:
                field = array_delimiter.join(map(str, a[c]))
                fields.append(field)
            else:
                fields.append(str(a[c]))
        line = delimiter.join(fields)
        output_lines.append(line)

    return output_lines


###############################################################################

# Database operations on annotations


def add_column(annotations, column, name):
    """Adds a column of the given name to the annotations. Non-destructive
    (returns a deep copy)."""
    if not annotations:
        return annotations
    if name in annotations[0]:
        raise ValueError('Trying to duplicate name %s' % name)
    if len(annotations) != len(column):
        raise ValueError('Annotations and new column length don not match (%d vs %d)' % (len(annotations), len(column)))

    new_annots = []
    for annot, value in zip(annotations, column):
        new_annot = copy.deepcopy(annot)
        new_annot[name] = value
        new_annots.append(new_annot)
    return new_annots


def do_join_items_by(i1, i2, name):
    """Joins two dicts by the given member."""
    if i1[name] != i2[name]:
        raise ValueError('Cannot join items by %s: values not equal! (1: %s, 2: %s)' % (name, i1[name], i2[name]))
    output = { key : value for key,value in i1.items }
    for key, value in i2.items:
        output[key] = value
    return output


def inner_join_by(a1, a2, name):
    """Inner-joins the given arrays of dicts by the column with the given
    name."""
    a1_index = group_by(a1, name)
    a2_index = group_by(a2, name)

    logging.debug('Available %s in a1: %s' % (name, str(a1_index.keys())))
    logging.debug('Available %s in a2: %s' % (name, str(a2_index.keys())))

    storage = defaultdict(dict)

    for item in a1:
        if item[name] in a2_index:
            storage[item[name]].update(item)

    for item in a2:
        if item[name] in a1_index:
            storage[item[name]].update(item)

    output = storage.values()
    return output


def group_by(annotations, name):
    """
    Splits the annotations into item arrays with the same name.

    :param annotations: The annotations array.

    :param name: One of the names in ``column_names``.

    :return: A dict of annotations arrays. The keys are all values that occur
        in the ``name`` column, the values are the complete annotation items
        (including the ``name`` column!)
    """
    # Empty
    if not annotations:
        return annotations
    column_names = annotations[0].keys()

    if not name in column_names:
        raise ValueError('Invalid column name: %s (available: %s)',
                         (name, str(column_names)))

    output_dict = {}
    for item in annotations:
        value = item[name]
        if value not in output_dict:
            output_dict[value] = []
        output_dict[value].append(item)

    return output_dict


def intersect(a1, a2, name, unique_key='id'):
    """Returns two lists of annotations consisting of the items from ``a1`` and
    ``a2`` that have the same value in the column ``name``. If there is more
    than one item with the same value in the ``name`` column in one of the
    annotations, a cartesian product of each from a1 -- each from a2 is
    generated.

    :param unique_key: If a pair of items in a1, a2 share the unique key, they
        are considered the same item and not considered in the intersection.
        This controls self-intersection behavior, when for instance measuring
        self-agreement.
    """
    grouped_a1 = group_by(a1, name)
    grouped_a2 = group_by(a2, name)

    output_a1 = []
    output_a2 = []

    for value in grouped_a1:
        if value in grouped_a2:
            for item1 in grouped_a1[value]:
                for item2 in grouped_a2[value]:
                    if item1[unique_key] != item2[unique_key]:
                        output_a1.append(item1)
                        output_a2.append(item2)

    return output_a1, output_a2

###############################################################################

# Functional operations

def apply_on_columns(annotations, columns, operation, **kwargs):
    """Creates an array that maps the given operation on the given columns."""
    output = []
    for annot in annotations:
        cols = [ annot[col] for col in columns ]
        result = operation(*cols, **kwargs)
        output.append(result)
    return output


def apply_on_annotations(annotations_array, column, operation, **kwargs):
    """Apply an operation on the same column from multiple annotations."""
    output = []
    for items in zip(*annotations_array):
        cols = [ item[column] for item in items ]
        result = operation(*cols, **kwargs)
        output.append(result)
    return output


def grid_on_intersection(grouped_annotations, intersect_column, op_column,
                         operation, skip_identical=True, record_counts=True,
                         **op_kwargs):
    """Applies the given operation on each pair of groups in the grouped
    annotation (on the given column).

    :param grouped_annotations: A grouped annotations dict. Grid members will
        be pairs of groups.

    :param intersect_column: The column according to which to compute the
        intersect of two annotation groups.

    :param op_column: The column on which to apply the ``operation``.

    :param skip_identical: If set (default), will not include the grid
        diagonal.

    :param operation: The operation which will be applied_on_annotations on the
        ``op_column`` of the intersecting annotations for each grid member.

    :param record_counts: If set (default), will assume the ``column`` is an
        array column and will also record counts for items in each intersection.
        The counts will be recorded in the same structure as the results of
        the operation, in tuples ``(count_g1, count_g2)``.

    :returns: A dictionary of operation results on the given ``column``s of
        the intersection of each pair of annotation groups. To find the result
        for (g1, g2), go through ``results[g1][g2]``. **The result is a list
        of results for all members of the intersect between g1 and g2.**
    """
    total_intersection_size = 0

    counts = {}
    results = {}
    for g1 in sorted(grouped_annotations.keys()):
        g1_results = {}
        g1_counts = {}
        for g2 in sorted(grouped_annotations.keys()):
            if skip_identical and g1 == g2:
                continue
            intersection = intersect(grouped_annotations[g1],
                                     grouped_annotations[g2],
                                     intersect_column)

            # Update counters
            if record_counts:
                intersection_size = len(intersection[0])
                total_intersection_size += intersection_size

                g1_items_count = count_by_len(intersection[0], op_column)
                g2_items_count = count_by_len(intersection[1], op_column)

                g1_counts[g2] = (g1_items_count, g2_items_count)

            # Apply operation
            result = apply_on_annotations([intersection[0], intersection[1]],
                                          op_column, operation, **op_kwargs)
            g1_results[g2] = result

        counts[g1] = g1_counts
        results[g1] = g1_results

        if not record_counts:
            counts = None

    return results, counts


def apply_on_grid(grid, operation, **operation_kwargs):
    """Applies the given operation on each grid member. The grid is assumed
    to be the output of a ``grid_on_`` method (currently implemented: only
    grid_on_intersect): a dict of dicts of lists. The operation is applied
    to each grid member. The result is again a grid.
    """
    output_grid = {}
    for g1 in grid:
        current_output = {}
        current_member = grid[g1]
        for g2 in current_member:
            logging.debug('Applying operation on grid member %s/%s: %s' % (
                str(g1), str(g2), str(current_member[g2])
            ))
            current_output[g2] = operation(current_member[g2],
                                           **operation_kwargs)
        output_grid[g1] = current_output
    return output_grid


def filter_grid(grid, condition, **condition_kwargs):
    """Filters the given grid: only if the given condition holds,
    will retain the given grid member in output. Non-destructive."""
    output_grid = {}
    for g1 in grid:
        current_output = {}
        current_member = grid[g1]
        for g2 in current_member:
            if condition(current_member[g2], **condition_kwargs):
                current_output[g2] = current_member[g2]
                # logging.debug('Filter_grid retaining member %s/%s: %s' % (
                #     str(g1), str(g2), str(current_member[g2])
                #                                                          ))
            else:
                logging.debug('Will not retain member %s/%s: %s' % (
                    str(g1), str(g2), str(current_member[g2])
                ))
        if len(current_output) > 0:
            logging.debug('Will retain grid for top-level member %s' % str(g1))
            output_grid[g1] = current_output
    return output_grid


def apply_on_grid_axis(grid, operation, **operation_kwargs):
    """Applies the given operation on the list of values of each top-level
    grid member (grid row)."""
    output = {}
    for g1 in grid:
        output[g1] = operation(grid[g1].values(), **operation_kwargs)
    return output


def apply_on_grids(grids, operation, **operation_kwargs):
    """Analogous to apply_on_annotations - applies the given operation
    to members of grids indexed by the same label pairs. Outputs, again,
    a grid with the results of the ``operation``."""
    output = {}
    for g1 in grids[0]:
        current_output = {}
        for g2 in grids[0][g1]:
            items = [ g[g1][g2] for g in grids ]
            result = operation(*items, **operation_kwargs)
            current_output[g2] = result
        if len(current_output) > 0:
            output[g1] = current_output
    return output


def grid_axis_weighed_average(grid, weights_grid):
    """Computes the weighed average for the top-level grid member (grid row)."""
    output = {}
    for g1 in grid:
        vw_pairs = [ (grid[g1][g2], weights_grid[g1][g2])
                      for g2 in grid[g1].keys() ]
        values, weights = zip(*vw_pairs)
        output[g1] = avg(values, weights)
    return output

##############################################################################

# Reporting and Evaluation


def annot_stats(annotations):
    """
    Computes all the statistics from the given annotations.

    :param annotations: An annotations array.

    :return: A dict of annotation statistics.
    """
    # Total annotations
    total_items = len(annotations)

    # Total/average appropriate
    total_appropriate = count_by_len(annotations, 'appropriate')
    avg_appropriate = float(total_appropriate) / float(total_items)

    # Total/average inappropriate
    total_inappropriate = count_by_len(annotations, 'inappropriate')
    avg_inappropriate = float(total_inappropriate) / float(total_items)

    # Total annotated
    total_images = total_appropriate + total_inappropriate
    avg_images = avg_appropriate + avg_inappropriate

    return total_items, total_images, avg_images, total_appropriate, \
            avg_appropriate, total_inappropriate, avg_inappropriate


def annot_stats_report(in_stats, latex=True):

    total_items, total_images, avg_images, total_appropriate,\
        avg_appropriate, total_inappropriate, avg_inappropriate = in_stats

    table_line = '%d\t%.3f\t%d\t%.3f\t%d\t%.3f' % (total_items, avg_images,
    total_appropriate, avg_appropriate, total_inappropriate, avg_inappropriate)

    #if latex:
        #table_line = ' & '.join(table_line.split()) + ' \\\\'
        ##table_line += "\n\\hline"

    return table_line


def count_by_len(annotations, name, array_names=ao_array_names+ai_array_names):
    """Counts annotations in the given type.

    :param annotations:
    :param name:
    :return:
    """
    if name not in array_names:
        logging.warn('Name %s not in array names (%s)' % (name, str(array_names)))
    total = 0
    for item in annotations:
        total += len(item[name])
    return total


def counts_by_len(annotations, name, array_names=ao_array_names+ai_array_names):
    """Gives annotation array field lengths in the given column."""
    if name not in array_names:
        logging.warn('Name %s not in array names (%s)' % (name, str(array_names)))
    lengths = []
    for item in annotations:
        lengths.append(len(item[name]))
    return lengths


def avg_annot_precision(prediction, true, name):
    """Computes the average precision of the ``prediction`` annotations of
    column ``name`` against the ``true`` annotations.
    """
    precisions = apply_on_annotations([prediction, true], name, precision)
    if len(precisions) == 0:
        #logging.warn('avg_annot_precision: No prediction!')
        return 0.0
    average_precision = avg(precisions)
    return average_precision


def avg_annot_fscore(prediction, true, name):
    fscores = apply_on_annotations([prediction, true], name, f_score)
    if len(fscores) == 0:
        return 0.0
    average_fsc = avg(fscores)
    return average_fsc


def group_mutual_precision(grouped_annotations, name):
    """Measures the mutual precision in each group in the column with the given
    name. Intended mainly for measuring items grouped by text.

    :returns: A dict with keys corresponding to ``grouped_annotations`` groups
        and values corresponding to mutual precision."""
    outputs = {}
    for gkey in grouped_annotations:
        item_fields = [ g[name] for g in grouped_annotations[gkey]]
        outputs[gkey] = avg_pairwise_mutual_precision(item_fields)
    return outputs

##############################################################################

# Miscellaneous functions

def timeframe(annotations):
    """Outputs the minimum and maximum time of annotations as a tuple."""
    if not annotations:
        return None
    if not 'time' in annotations[0]:
        raise ValueError('Cannot get timeframe from annotations without time.')

    times = [int(t) for t in map(operator.itemgetter('time'), annotations)]
    min_time = min(times)
    max_time = max(times)
    return (min_time, max_time)

def human_timeframe(annotations):
    """Gets the timeframe in a human-readable format."""
    from_t, to_t = timeframe(annotations)

    from_d = datetime.datetime.fromtimestamp(from_t)
    from_h = from_d.strftime('%Y-%m-%d %H:%M:%S')

    to_d = datetime.datetime.fromtimestamp(to_t)
    to_h = to_d.strftime('%Y-%m-%d %H:%M:%S')

    return from_h, to_h

def plot_column_in_time(annotations, name, **pyplot_kwargs):
    """Plots the development of a certain column in time.

    Assumes the column is numerical."""
    times = [ int(a['time']) for a in annotations ]
    values = [ a[name] for a in annotations ]

    plt.figure(figsize=(8,4))
    if 'title' in pyplot_kwargs:
        plt.title(pyplot_kwargs['title'])
    plt.plot(times, values, 'ro')
    plt.axis([min(times), max(times), 0, 12])
    plt.show()


##############################################################################

# Agreement reporting

def report_agreement(avg_result_over_users, avg_results,
                     column, counts, counts_per_user, max_agreement,
                     min_agreement, pivot_agreement, results_counts,
                     w_avg_result_over_users, w_avg_results_per_user,
                     averages_only=False, print_descriptor=False, latex=False):
    """Prints the agreement information."""
    print '\n\n---------    Inter-annotator scores for %s    ----------\n' % column
    print 'Average agreement: %f/%f' % (w_avg_result_over_users,
                                        avg_result_over_users)
    print 'Maximum/Minimum agreement: %f / %f' % (max_agreement,
                                                  min_agreement)

    if print_descriptor:
        print '\n\n-----------------------------------------------------------'
        print 'Columns:\n-------\n\n' \
              'username; avg. agreement; % of median agr.; # of duplicates total'
        print '\tother u.name; agr. w. other; # of dupl. w. ot.; # of imgs. you/ot.'
        print '\tother u.name; agr. w. other; # of dupl. w. ot.; # of imgs. you/ot.'
        print '\tetc.'
        print '-----------------------------------------------------------\n\n'

    print 'Individual user reports:\n'

    for u1 in avg_results:
        if not averages_only:
            print ''
        out = '%s\t%f\t%f\t%d' % (u1[:3],
                                    w_avg_results_per_user[u1],
                                    w_avg_results_per_user[u1]/pivot_agreement,
                                    counts_per_user[u1])  # Add % of maximum res.
        if latex:
            out = ' & '.join(out.split()) + ' \\\\'
        print out

        if not averages_only:
            print '\\hline'
            for u2 in avg_results[u1]:
                out ='\t%s\t%f\t%d\t%d/%d' % (u2[:3], avg_results[u1][u2],
                                                 results_counts[u1][u2],
                                                 counts[u1][u2][0],
                                                 counts[u1][u2][1])
                if latex:
                    out = ' & '.join(out.split()) + ' \\\\'
                print out


def compute_agreement(annotations, column, evaluation_fn,
                      print_averages_only=False, no_print=False, latex=False,
                     **evaluation_fn_kwargs):
    """Reports the average evaluation_fn for all pairs of annotation
    groups with non-empty intersection."""
    results, counts = grid_on_intersection(annotations, 'text',
                                           column, evaluation_fn,
                                           skip_identical=False,
                                           **evaluation_fn_kwargs)
    results = filter_grid(results, lambda x: len(x) > 0)
    results_counts = apply_on_grid(results, len)
    avg_results = apply_on_grid(results, avg)

    avg_results_per_user = apply_on_grid_axis(avg_results, avg)
    w_avg_results_per_user = grid_axis_weighed_average(avg_results,
                                                       results_counts)
    counts_per_user = apply_on_grid_axis(results_counts, sum)

    # Get weighed average results instead of plain average
    # - weighing by no. of annotation items

    results_list = [ w_avg_results_per_user[u]
                     for u in sorted(w_avg_results_per_user.keys())]
    counts_list = [ counts_per_user[u]
                    for u in sorted(w_avg_results_per_user.keys())]
    avg_result_over_users = avg(results_list)
    w_avg_result_over_users = avg(results_list, counts_list)

    max_agreement = max(results_list)
    min_agreement = min(results_list)
    pivot_agreement = sorted(results_list)[len(results_list)/2 + 1]

    # Return weighed averages per user
    if not no_print:
        report_agreement(avg_result_over_users, avg_results,
                         column, counts, counts_per_user, max_agreement,
                         min_agreement, pivot_agreement, results_counts,
                         w_avg_result_over_users, w_avg_results_per_user,
                         averages_only=print_averages_only, latex=latex)

    return w_avg_results_per_user

##############################################################################


def reward_coefficients(average_agreement_by_user):
    """Computes reward coefficients based on the users' average agreement."""
    pivot = sorted(average_agreement_by_user.values())[len(average_agreement_by_user.values())/2 + 1]
    output = { u : average_agreement_by_user[u] / pivot
               for u in average_agreement_by_user}
    return output


def compute_time_reward(annotations):
    """Computes the reward for time spent annotating."""
    minutes_worked = 0
    times = map(int, sorted(map(operator.itemgetter('time'), annotations)))
    start = min(times)
    end = max(times)

    tidx = 0
    logging.debug('Maximum minutes: %d' % len(range(start - 1, end + 59, 60)))
    logging.debug('Total times: %d' % len(times))
    for minute_start in xrange(start - 1, end + 59, 60):
        logging.debug('minute starting: %d' % minute_start)
        if times[tidx] < minute_start:
            logging.debug('Worked in minute starting at %d: time %d' % (minute_start, times[tidx]))
            minutes_worked += 1
            while times[tidx] < minute_start:
                logging.debug('Moving tidx to %d, time at tidx %d < %d' % (tidx,
                                                    times[tidx], minute_start))
                tidx += 1
                if tidx >= len(times):
                    break
            if tidx > len(times):
                break

    reward_per_minute = 1.0
    return minutes_worked * reward_per_minute


def compute_item_reward(annotations, coef_appropriate=1.0,
                        coef_inappropriate=1.0):
    """Computes the reward for individual annotated images."""
    total_item_reward = 0.0
    item_rewards = [ 1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25 ]
    for item in annotations:
        app = frozenset(item['appropriate'])
        inapp = frozenset(item['inappropriate'])
        for r, i in zip(item_rewards, item['combined']):
            if i in app:
                total_item_reward += r * coef_appropriate
            elif i in inapp:
                total_item_reward += r * coef_inappropriate
    return total_item_reward

def compute_reward(annotations, coef_appropriate=1.0, coef_inappropriate=1.0):
    """Computes how much an annotator should be paid for the given set of
    annotations.

    The formula::

        R = time_reward + item_rewards * agreement_coefficient

    where::

        time_reward = # of minutes in which at least one annotation was sent
        item_rewards = sum of individual item rewards
        item_reward = dot([1 for _ in annotated_pictures], reward_vector)
        reward_vector = [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0., 0., 0., 0., 0.]
        agreement_coefficient = apppropriate + inappropriate agr. coef. / 2
        appropriate agr. coef. = avg. app. precision vs all users / 3rd best avg. prec.
        inappropriate agr. coef. = dtto for inapp. precision

    :returns: The total reward.
    """
    time_reward = compute_time_reward(annotations)
    item_reward = compute_item_reward(annotations,
                                      coef_appropriate=coef_appropriate,
                                      coef_inappropriate=coef_inappropriate)
    reward = time_reward + item_reward
    return reward


def compute_user_orig_acc(annotations, originals, column='appropriate',
                          with_counts=False):
    """Computes how often are original illustrative images annotated
        as the given column.

        :param with_counts: Will also return the hit count."""
    total_hits = 0.0
    total_with_original = 0.00001
    total_annotated = 0.00001
    for a in annotations:
        if a['text'] not in originals:
            continue
        original = originals[a['text']]
        if original in a['images']:
            total_with_original += 1.0
            total_annotated += len(a[column])
            if original in a[column]:
                total_hits += 1.0

    rec = total_hits / total_with_original
    prec = total_hits / total_annotated
    f_sc = (rec * prec) / (rec + prec + 0.00001)

    if with_counts:
        return rec, prec, f_sc, total_hits
    else:
        return rec, prec, f_sc

###############################################################################

def find_overlapping(annotations, group='text', column='appropriate',
                     output_column='overlapping'):
    """Will add an ``output_column`` that contains images that overlap in the
    given ``column`` when annotations are grouped by ``group_by``.
    """

    grouped = group_by(annotations, group)
    overlapping = {}

    # Find overlapping items
    for g in grouped:
        cg = grouped[g]
        if len(cg) <= 1:
            continue
        i1_set = set(cg[0][column])
        intersection = i1_set.intersection(*map(list, map(operator.itemgetter(column), cg[1:])))
        overlapping[g] = list(intersection)
        #print 'Appended to overlapping %s: %s' % (g, str(list(intersection)))

    logging.debug('Total overlapping: %d' % len(overlapping))

    # Generate output column
    o_column = []
    for a in annotations:
        g = a[group]
        if g not in overlapping:
            o_column.append([])
        else:
            logging.debug('Appending to key %s, overlapping members: %s' % (g, str(overlapping[g])))
            o_column.append(overlapping[g])

    output_annotations = add_column(annotations, o_column, output_column)

    return output_annotations

def generate_t2i_map(annotations, column='overlapping'):
    """Generates the text-image map string from the given annotations."""
    output_lines = []

    grouped = group_by(annotations, 'text')
    for g in grouped:
        logging.debug('\nProcessing key %s' % g)
        cg = grouped[g]
        logging.debug('\n'.join([str(c[column]) for c in cg]))
        if len(cg[0][column]) == 0:
            logging.debug('Continuing')
            continue
        else:
            logging.debug('Adding items:\n' + '\t\n'.join([str(c[column]) for c in cg]))
            all_items = [ i for cgl in cg for i in cgl[column] ]
            all_items_deduplicated = set(all_items)
            logging.debug('Adding items: %s' % str(all_items_deduplicated))
            for i in all_items_deduplicated:
                if i == '': # HACK: when both annotators agree that no image fits
                    continue
                output_lines.append('\t'.join([g, i]))

    return output_lines


###############################################################################


def main(args):
    logging.info('Executing annotation_stats.py...')

    output_annotations = parse_annotations(args.annot_output)
    input_annotations = parse_annotation_inputs(args.annot_input)
    annotations = inner_join_by(output_annotations, input_annotations, 'id')

    # Obtaining item counts:
    appropriate_counts = counts_by_len(annotations, 'appropriate')
    annotations = add_column(annotations, appropriate_counts, 'app_count')
    inappropriate_counts = counts_by_len(annotations, 'inappropriate')
    annotations = add_column(annotations, inappropriate_counts, 'inapp_count')
    total_counts = apply_on_columns(annotations, ['app_count', 'inapp_count'],
                                    lambda x,y: x+y)
    annotations = add_column(annotations, total_counts, 'total_count')

    # Filling in combined and unannotated columns.
    combined = apply_on_columns(annotations, ['appropriate', 'inappropriate'],
                                lambda x,y: x+y)
    annotations = add_column(annotations, combined, 'combined')

    unannotated = apply_on_columns(annotations, ['images','combined'],
                                   lambda x,y: [i for i in x if i not in y])
    annotations = add_column(annotations, unannotated, 'unannotated')

    # Filtering items
    if args.labels:
        print '\n\n--------- Filtering by labels: %s ---------\n' % ', '.join(args.labels)
        annotations = filter(lambda x: x['label'] in args.labels, annotations)

    if args.filter_suspicious_inappropriate:
        print '\n\n----- Suspicious item filtering: inappropriate imgs. -----\n'
        filtered_annotations = filter(lambda x: len(x['inappropriate']) >= args.filter_suspicious_inappropriate,
                             annotations)
        annotations = filter(lambda x: len(x['inappropriate']) < args.filter_suspicious_inappropriate,
                             annotations)
        print 'Total filtered: %d' % len(filtered_annotations)

    if args.filter_suspicious:
        print '\n\n----- Suspicious item filtering: combined imgs. -----\n'
        filtered_annotations = filter(lambda x: len(x['combined']) >= args.filter_suspicious,
                             annotations)
        annotations = filter(lambda x: len(x['combined']) < args.filter_suspicious,
                             annotations)
        print 'Total filtered: %d' % len(filtered_annotations)

    annotations_by_user = group_by(annotations, 'user')

    if args.total:
        print '\n---------- Totals report ---------\n'

        tline = 'Total:                   \t' +  annot_stats_report(annot_stats(annotations))
        if args.latex:
            tline = ' & '.join(tline.split()) + ' \\\\'
            tline += "\n\\hline"
        print tline

        for user in annotations_by_user:
            report = annot_stats_report(annot_stats(annotations_by_user[user]))
            out = '%s\t%s' % (user[:3], report)
            if args.latex:
                out = ' & '.join(out.split()) + ' \\\\'
            print out

    # Agreement reports

    if args.global_agreement_report:
        print '\n\n------------- Agreement report -----------------\n'

        annotations_by_text = group_by(annotations, 'text')
        duplicate_annotations = { t : annotations_by_text[t] for t in annotations_by_text
                                  if len(annotations_by_text[t]) > 1 }
        print 'Total duplicates: %d' % len(duplicate_annotations)

        # Appropriate items global agreement report

        appropriate_pmp = group_mutual_precision(duplicate_annotations, 'appropriate')
        avg_appropriate_pmp = avg(appropriate_pmp.values())
        print 'Appropriate average p.m.p.: %f' % avg_appropriate_pmp

        no_agreement_appropriate = [ annotations_by_text[t] for t in appropriate_pmp if appropriate_pmp[t] == 0.0]
        print 'Texts with no agreement on appropriate: %d' % len(no_agreement_appropriate)
        perfect_agreement_appropriate = [ annotations_by_text[t] for t in appropriate_pmp if appropriate_pmp[t] == 1.0]
        print 'Texts with perfect agreement on appropriate: %d' % len(perfect_agreement_appropriate)

        # Inappropriate items global agreement report

        inappropriate_pmp = group_mutual_precision(duplicate_annotations, 'inappropriate')
        avg_inappropriate_pmp = avg(inappropriate_pmp.values())
        print 'Inppropriate average p.m.p.: %f' % avg_inappropriate_pmp

        no_agreement_inappropriate = [ annotations_by_text[t] for t in inappropriate_pmp if inappropriate_pmp[t] == 0.0]
        print 'Texts with no agreement on inappropriate: %d' % len(no_agreement_inappropriate)
        perfect_agreement_inappropriate = [ annotations_by_text[t] for t in inappropriate_pmp if inappropriate_pmp[t] == 1.0]
        print 'Texts with perfect agreement on inappropriate: %d' % len(perfect_agreement_inappropriate)

    # Report agreement by user
    agreement_columns = [ 'appropriate', 'inappropriate']
    # Compute f-score of user 1 predicting user 2's labels
    if args.f_score:
        print '\n\nReporting: f-scores'
        eval_fn = f_score
        eval_fn_kwargs = { 'w_rec' : 0.5, 'w_prec' : 1.5 }
        for column in agreement_columns:
            compute_agreement(annotations_by_user, column, eval_fn,
                              print_averages_only=args.agreement_averages_only,
                              latex=args.latex,
                              **eval_fn_kwargs)

    if args.precision:
        print '\n\nReporting: precisions'
        eval_fn = precision
        eval_fn_kwargs = {}
        for column in agreement_columns:
            compute_agreement(annotations_by_user, column, eval_fn,
                              print_averages_only=args.agreement_averages_only,
                              latex=args.latex,
                              **eval_fn_kwargs)
    if args.recall:
        print '\n\nReporting: recall'
        eval_fn = recall
        eval_fn_kwargs = {}
        for column in agreement_columns:
            compute_agreement(annotations_by_user, column, eval_fn,
                              print_averages_only=args.agreement_averages_only,
                              latex=args.latex,
                              **eval_fn_kwargs)

    # "Suspiciously many pictures" items report
    if args.suspicious:
        print '\n\n--------- Suspicion report -------------\n'

        suspicion_level = 10

        print 'Reporting itmes with more than %d images tagged.' % suspicion_level

        suspiciously_many = [ item for item in annotations if item['app_count'] + item['inapp_count'] >= suspicion_level]
        suspiciously_many_by_user = group_by(suspiciously_many, 'user')

        print 'Total suspiciously prolific items: %d\n' % len(suspiciously_many)
        #print '\nColumns:\n\tusername; # susp.; % susp.'

        for user in suspiciously_many_by_user:
            suspicious_proportion = (1.0 * len(suspiciously_many_by_user[user])) / (1.0 * len(annotations_by_user[user]))
            out = '\t%25s\t%d\t%f' % (user,
                                          len(suspiciously_many_by_user[user]),
                                          suspicious_proportion)
            if args.latex:
                out = ' & '.join(out.split()) + ' \\\\'
            print out

    # Plot for each user the annotated inappropriate item counts.
    if args.plot_counts:
        for user in annotations_by_user:
            plot_column_in_time(annotations_by_user[user], 'inapp_count',
                                title=user)
            plot_column_in_time(annotations_by_user[user], 'total_count',
                                title=user)

    if args.rewards:
        print '\n\n------------ Rewards -------------\n'
        # print 'Columns:\n\tusername; time; item; total; app.coef.; inapp.coef.'

        if args.reward_metric == 'precision':
            reward_metric = precision
        elif args.reward_metric == 'f-score':
            reward_metric = f_score
        elif args.reward_metric == 'recall':
            reward_metric = recall

        appropriate_agreements = compute_agreement(annotations_by_user,
                                                   'appropriate',
                                                   reward_metric,
                                                   no_print=True)
        appropriate_rewards = reward_coefficients(appropriate_agreements)
        inappropriate_agreements = compute_agreement(annotations_by_user,
                                                   'inappropriate',
                                                   reward_metric,
                                                   no_print=True)
        inappropriate_rewards = reward_coefficients(inappropriate_agreements)

        total_rewards = 0.0
        total_time_rewards = 0.0
        total_item_rewards = 0.0

        for user in annotations_by_user:
            if user not in appropriate_agreements:
                reward = 0.0
            else:
                time_reward = compute_time_reward(annotations_by_user[user])
                item_reward = compute_item_reward(annotations_by_user[user],
                                    coef_appropriate=appropriate_rewards[user],
                                    coef_inappropriate=inappropriate_rewards[user])
                total_reward = time_reward + item_reward

                total_time_rewards += time_reward
                total_item_rewards += item_reward
                total_rewards += total_reward
                out = '\t%25s\t%.1f\t%.1f\t%.1f\t%.3f\t%.3f' % (user,
                                                time_reward,
                                                item_reward,
                                                total_reward,
                                                appropriate_rewards[user],
                                                inappropriate_rewards[user])
                if args.latex:
                    out = ' & '.join(out.split()) + ' \\\\'
                print out

        print '\nTotal rewards:\t%.1f\t%.1f\t%.1f' % (total_time_rewards,
                                                      total_item_rewards,
                                                      total_rewards)


    if args.dump_duplicate_texts:
        duplicate_texts = sorted(duplicate_annotations.keys())
        duplicate_ids = []
        for t in duplicate_texts:
            ids = map(operator.itemgetter('id'), duplicate_annotations[t])
            duplicate_ids.append(ids)

        duplicate_lines = [ t + '\t' + ','.join([str(i) for i in ids])
                            for t, ids in zip(duplicate_texts, duplicate_ids) ]

        with open(args.dump_duplicate_texts, 'w') as output_handle:
            output_handle.write('\n'.join(duplicate_lines))

    if args.original:
        print '\n\n------ Original image recall report --------\n'

        # print '\nColumns:'
        # print 'username; % orig. in app.; # orig. in app.; % and # orig. in inapp.; dtto for unannotated\n\n'

        if not args.text_imfile_map:
            raise ValueError('Have to supply --text_imfile_map to compute original!')

        with open(args.text_imfile_map) as t2i_handle:
            originals = { text : image
                          for text, image in itertools.imap(lambda x: x.split(),
                                                            t2i_handle)
            }
        user_orig_accs = { u : compute_user_orig_acc(annotations_by_user[u],
                                                     originals,
                                                     with_counts=True)
                           for u in annotations_by_user }
        user_orig_inapp_accs = { u : compute_user_orig_acc(annotations_by_user[u],
                                                           originals,
                                                           column='inappropriate',
                                                           with_counts=True)
                                         for u in annotations_by_user }
        user_orig_untagged_accs = { u : compute_user_orig_acc(annotations_by_user[u],
                                                           originals,
                                                           column='unannotated',
                                                           with_counts=True)
                                         for u in annotations_by_user }

        for u in user_orig_accs:
            out = '%s\t%.3f\t%d\t%.3f\t%d\t%.3f\t%d' % (u[:3],
                                      user_orig_accs[u][0],
                                      user_orig_accs[u][3],
                                      user_orig_inapp_accs[u][0],
                                      user_orig_inapp_accs[u][3],
                                      user_orig_untagged_accs[u][0],
                                      user_orig_untagged_accs[u][3],
            )
            if args.latex:
                out = ' & '.join(out.split()) + ' \\\\'
            print out
        recs, precs, f_scores, counts = zip(*user_orig_accs.values())
        avg_rec = avg(recs, counts)
        print '\nWeighed average original recall: %.3f (total hits: %d)' % (
            avg_rec, sum(counts)
        )

    if args.t2i:

        print '\n\nExporting t2i map...'

        annotations = find_overlapping(annotations, group='text',
                                       column='appropriate',
                                       output_column='t2i_output')
        print 'Total annotations: %d' % len(annotations)
        t2i_string = generate_t2i_map(annotations, 't2i_output')
        with open(args.t2i, 'w') as t2i_handle:
            t2i_handle.write('\n'.join(t2i_string) + '\n')


    logging.info('Exiting annotation_stats.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--root', action='store',
                        help='Annotation data dir root. This works for '
                             'retrieving the \'real\' images that were '
                             'originally assigned to the text.')
    parser.add_argument('-a', '--annot_output', action='store',
                        help='The annotations output file to process (full path).')
    parser.add_argument('-i', '--annot_input', action='store',
                        help='The annotations input file that was used to '
                             'distribute items to annotators. Used to group '
                             'annotations by text. (Full path.)')
    parser.add_argument('-m', '--text_imfile_map', action='store',
                        help='The text-image file map file. Used when computing'
                             ' how much annotators identified the original'
                             ' images.')

    parser.add_argument('-l', '--labels', action='store', nargs='+',
                        help='Only consider annotations with this label.')

    parser.add_argument('--total', action='store_true',
                        help='If set, will compute total amounts of tagged '
                             'images and items.')

    parser.add_argument('--f_score', action='store_true',
                        help='If set, will compute f-score agreement for all'
                             ' pairs of users.')
    parser.add_argument('--precision', action='store_true',
                        help='If set, will compute precision agreement for all'
                             ' pairs of users.')
    parser.add_argument('--recall', action='store_true',
                        help='If set, will compute recall agreement for all'
                             ' pairs of users.')

    parser.add_argument('--agreement_averages_only', action='store_true',
                        help='If set, will not print detailed user-vs-user'
                             'agreement stats.')
    parser.add_argument('--plot_counts', action='store_true',
                        help='If set, will plot annotated items counts for '
                             'annotators.')
    parser.add_argument('--dump_duplicate_texts', action='store',
                        help='Will dump duplicate text IDs into this file, '
                             'one per line, together with corresponding item '
                             'IDs comma-separated in the second column.')
    parser.add_argument('--filter_suspicious', action='store', type=int,
                        help='Will filter out items with suspiciously many '
                             'images tagged. The argument is the threshold for'
                             ' marking an item as suspicious.')
    parser.add_argument('--filter_suspicious_inappropriate',
                        action='store', type=int,
                        help='Will filter out items with suspiciously many '
                             'images tagged as inappropriate.. The argument is '
                             'the threshold for marking an item as suspicious.')

    parser.add_argument('--original', action='store_true',
                        help='Will report how often annotators identified the '
                             'original images used for the texts.')
    parser.add_argument('--suspicious', action='store_true',
                        help='Will report items with a suspicious amount of '
                             'annotated images.')
    parser.add_argument('--rewards', action='store_true',
                        help='Compute rewards for users.')
    parser.add_argument('--reward_metric', action='store', default='precision',
                        help='Determine which metric should be used to compute'
                             ' reward coefficients. Accepts \'precision\', '
                             '\'recall\' and \'f-score\'. Defaults to precision.')
    parser.add_argument('--global_agreement_report', action='store_true',
                        help='Compute global agreement stats.')

    parser.add_argument('--t2i', action='store',
                        help='Generate a vtext-image map to this file. By '
                             'default in a strict regime.')
    parser.add_argument('--relaxed', action='store_true',
                        help='If set, will also put images to t2i map that were'
                             ' only tagged by one annotator. [NOT IMPLEMENTED]')
    parser.add_argument('--with_solo', action='store_true',
                        help='Will run in un-relaxed regime, but will also '
                             'output images from items that only one annotator'
                             ' has seen. [NOT IMPLEMENTED]')

    parser.add_argument('--latex', action='store_true',
                        help='If set, will output tables with LaTeX delimiters.')

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
