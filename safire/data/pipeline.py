"""
This module contains classes that ...
"""
import logging
import safire.utils.transcorp

__author__ = "Jan Hajic jr."


class Pipeline(object):
    """This class is a wrapper around a graph of blocks that provides its
    outward interface. The rule is that there is one output per pipeline.
    Multiple pipelines can share blocks, however.

    The Pipeline instance adds only a minimal overhead around the blocks:
    a dictionary that refers to individual blocks by their configuration names,
    a dependency graph that describes the topology of the pipeline,
    a topologically sorted version of the graph and several methods that improve
    the API to the pipeline blocks' operations. (These will mostly call their
    ``safire.utils.transcorp`` counterparts on the pipeline's output block.)
    """

    def __init__(self, output, objects, depgraph, sorted_depgraph):
        """
        :param output: The output block.

        :param objects: The dictionary of all blocks. Keys are block names
            from the corresponding configuration.

        :param depgraph: A dependency graph for individual blocks. Keys are
            object names, values are lists of object names.

        :param sorted_depgraph: The sorted version of the dependency graph,
            implemented as an OrderedDict. The sort order is the initialization
            order: if object X comes before object Y in the sorted graph, that
            means X can be initialized before Y, and so X does not depend on Y.
            (Because pipelines are a partial ordering, Y might not depend on
            X either.)
        """
        self.objects = objects
        self.output = output
        self.depgraph = depgraph
        self.sorted_depgraph = sorted_depgraph

    def get_object(self, name):
        """Retrieves the pipeline object with the given name."""
        return self.objects[name]

    def __getitem__(self, item):
        return self.output[item]

    def __len__(self):
        return len(self.output)

    def __iter__(self):
        for item in self.output:
            yield item
        # Should add batch mode iteration.

    def log_corpus_stack(self, with_length=True):
        safire.utils.transcorp.log_corpus_stack(self.output,
                                                with_length=with_length)

    def dimension(self):
        return safire.utils.transcorp.dimension(self.output)

    def report_memory_usage(self, names=None, human_readable=True):
        """Logs memory usage for each of the pipeline objects. If ``names``
        are given, will only log memory usage for the given names.

        Logs memory usage in order of initialization (presumably from smallest
        to largest). If ``names`` are given, logs instead in the given order.

        If ``human_readable`` is given (default: yes), will output the in-memory
        sizes in B, kB, MB and GB.
        """
        if names is None:
            names = self.sorted_depgraph.keys()

        reports = ['Memory report:']
        total_size = 0
        for name in names:
            memdict = safire.utils.obj_memory_usage(self.objects[name])
            obj_size = sum(memdict.values())
            total_size += obj_size
            memdict['!TOTAL'] = obj_size
            mem_report = safire.utils.memory_usage_report(
                            memdict,
                            human_readable=human_readable,
                            linesep='\n\t')

            obj_report = '--------\t{0}\t--------\n'.format(name) + \
                         'Obj type: {0}\n'.format(type(self.objects[name])) + \
                         mem_report
            obj_report += '\n'
            reports.append(obj_report)

        if human_readable:
            total_size = safire.utils.pformat_nbytes(total_size)

        reports.append('Total size of reported objects: {0}'.format(total_size))
        output_report = '\n'.join(reports)
        print output_report

        return output_report







