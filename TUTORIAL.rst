
Safire Tutorial
===============


This tutorial will walk you through the Safire ``scripts`` suite. You will
learn how to:

* set up text and image corpora/datasets, with various preprocessing steps,
* train individual network layers,
* train multimodal text-image layers,
* build similarity indexes for image retrieval,
* stitch together layers into a processing pipeline and run the image retrieval
  task

We will not be talking about *how* the Safire library works. That is described in the technical documentation in the ``doc/build/html`` folder.

We assume that:

* You have set your PYTHONPATH so that your Python can find the safire module,
* You have set your PATH to include the scripts/ directory of the safire    
  package. 
  
All scripts have a -v and --debug parameter for INFO prints and DEBUG prints.

A little on datasets
--------------------

We need, unfortunately, to describe a couple of concepts on how Safire manages data. (Technically, you don't have to read this to run the tutorial, but some processing steps may seem unclear to you if you skip this section.)

The *root directory* is the directory in which the rest of the dataset lives. It has to adher to a specific structure and naming conventions, so that the Safire experiment management components can automatically take care of behind-the-scenes loading and saving. (If you really want to see the gritty details, look at the safire.data.layouts.DataDirLayout class and the safire.data.loaders module.)

The *dataset name* defines a set of text documents, images and a mapping between texts and images. These two arguments, root and name, are enough to fully specify for the Safire experiment management components what data you will be using. 

(For the curious: this set is defined by some *master files* in the root directory. The details on how to construct master files and where the actual data lies we leave out for now. The point of this setup is that you may have a *lot* of the raw data and you don't want to use all of it at once, or you want to use different subsets, or your raw data lies somewhere completely different and you do not want to copy it, you want to share it among different datasets.)  

For the ``safire-mini`` dataset included with the library, we have:

* The root directory: ./data/safire-mini
* The dataset name: safire-mini 

Let's build a pipeline!
=======================

Throughout the tutorial, we assume that you are running the commands from the safire library root directory (the one with this file). If you are not, the only thing that should be necessary to change will be the path to the root directory.

Raw data to corpora
-------------------

While we do have the raw data available through the root and name, we first need to convert it to a format that can be read by the deep learning models further up the line. The first thing we need to do is convert the text data to numbers (word counts). Run::

dataset2corpus -r ../data/mini-safire -n mini-safire -v

This will build a text corpus. The ``-v`` flag turns on the INFO messages that tell us what is going on. To mimic the thesis, we can add a couple of preprocessing steps::

dataset2corpus -r ../data/mini-safire -n mini-safire -v --pos NAV --top_k 1010 --discard_top 10 --pfilter 0.2 --pfilter_fullfreq --filter_capital --tfidf -l .POS.top1010.pf0.2.pff.fc.tfidf

This command will, besides building a (sparse) vector space from lemmas:

* --pos NAV          ... only retain Nouns, Adjectives and Verbs,
* --pfilter 0.2      ... only use lemmas that were in the first 20 % of 
                         sentences in each doc,
* --pfilter_fullfreq ... but count their frequencies from the whole document,
* --filter_capital   ... do not use lemmas that start with a capital letter,
* --tfidf            ... apply the Tf-Idf transformation,                      
* --top_k 1010       ... only use the 1010 most frequent words from the dataset
                         (after the application of Tf-Idf)
* --discard_top 10   ... and discard the 10 most frequent of these.

However, the most important argument is the last one::

 -l .NAV.top1010.pf0.2.pff.tfidf

This argument -- the l stands for *label* - tells Safire under which name to store the particular transformation of the dataset you defined through the other arguments. A lot goes on under the hood, but from your perspective, all you need to remember is the label and what it means.

You don't have to use what seems like such a tedious string, but if you are doing a lot of experiments, you will soon get lost in names like "my-transform" and "my-other-transform" and "bad-idea". (Generating these names automatically from the command-line parameters is something that will definitely feature in future versions of Safire.)

We also need to convert the raw image data to a Safire corpus. We will use two other preprocessing steps:

* --uniform_covariance ... to scale each feature to uniform covariance,
* --tanh 1.0           ... to transform each feature by the tanh function, with
                           a multiplicative coefficient of 1.0

To tell Safire that this will be an *image* corpus, we need to use the --images flag.

The command to run is::

dataset2corpus -r ../data/mini-safire -n mini-safire --images --uniform_covariance --tanh 1.0 -l .UCov.tanh

(Notice the pattern with the label?) Actually, regarding the label, we lied a bit: you also need to know whether you are using the image, or the text modality with the given label. You could of course run these same transforms on the text data as well (although it doesn't work the other way; image features really can't get filtered by part of speech tags).  

Training models
---------------

Now for the interesting part. We will train one layer above each modality and then join them into a multimodal model.

To train a 100-dimensional representation using a Restricted Boltzmann Machine on the previously build text corpus,  run::

pretrain.py -r ../data/mini-safire -n mini-safire -t .NAV.top1010.pf0.2.pff.fc.tfidf -m RestrictedBoltzmannMachine --n_out 100 --batch_size 1 --n_epochs 5 -v -l .NAV.top1010.pf0.2.pff.fc.tfidf.RBM-100

Notice that the text corpus infix is given by the -t flag. To similarly train a Denoising Autoencoder, but on images, use the image corpus label with the -i flag::

pretrain.py -r ../data/mini-safire -n mini-safire -i .UCov.tanh -m DenoisingAutoencoder --n_out 100 --batch_size 1 --n_epochs 5 -v -l .UCov.tanh.DA-100



The crown jewel: joint training
-------------------------------

Finally, to train the joint layer, run::

pretrain_multimodal -r ../data/mini-safire -n mini-safire -i .UCov.tanh.DA-100 -t .NAV.top1010.pf0.2.pff.fc.tfidf.RBM-100 -j .RBM-200 -m RestrictedBoltzmannMachine --batch_size 1 -v --n_out 200  


Similarity index
----------------

We now have the models and infrastructure in place to transform text to images using the stack of pretrained layers with the joint layer. However, we also need to build a similarity index which to query. We will run::

icorp2index -r ../data/mini-safire -n mini-safire -l .UCov.tanh -v


Putting it all together: the text --> image pipeline
----------------------------------------------------

To finally run the whole system, use::

run.py -r ../data/mini-safire -n mini-safire --num_best 10 -t .NAV.top1010.pff.fc.tfidf .RBM-100 -i . .UCov.tanh .DA-100 -j .RBM-200 -x .UCov.tanh -v --input ../data/mini-safire/mini-safire.vtlist > outputs.tmp

This step is a little complicated, because we have to specify:

* -t         ... Which components of the text pipeline will be used,
* -i         ... Which components of the image pipeline,
* -j         ... Which joint model,
* -x         ... Whcih image retrieval similarity index,
* --num_best ... How many most similar images to return per query text

Finally, we have to specify where to find the query documents. This is the 
--input parameter: it points straight to a *.vtlist file (one of the four master files for each root/name combination) that in turn contains the names of transformed and tagged documents... etc.

The output of the query system will be redirected to the otuputs.tmp file.


Evaluating the results
----------------------

Supposing you have the correct images for the texts in the --input *.vtlist file from the previous step, you can now measure the performance of the system: whether it was able to recover the original image.


What's next?
============

The other scripts have diverse roles:

* Data exploration/visualization: img_index_explorer.py, annot_stats.py, 
  dataset_stats.py, text_preprocessing_explorer.py

* filter_by_t2i.py, remove_duplicate_images.py, rename_iids.py: 'Nonstandard'  
  dataset management -- scripts for splitting of sub-datasets, filtering out
  duplicates, etc.
  
Run them with the ``-h`` option to get a list of available commands. Almost all the scripts use ``-r`` and ``-n`` for the root and name of the datasets and all have the ``-v`` option for verbose (``INFO`` level of Python's standard library ``logging``) output and ``--debug`` for detailed output (``DEBUG`` level of ``logging``).

Dataset visualization
----------------------

Two scripts support visualization what is going on during and around preprocessing and training and in the data themselves:

* dataset_stats.py, which has options for plotting multiple views of the data - 
  heatmaps, histograms, averages...
 
* The pretrain.py script has options --plot_transformation, --plot_weights, 
  --plot_on_init and --plot_every for plotting various information about
  ongoing training.        