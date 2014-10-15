#!/usr/bin/env python
"""
Maps image IDs from an ImagenetCorpus to image files for exploring
Safire model performance.
"""
import os
try:
    import Image
    import ImageFont
    import ImageDraw
except ImportError:
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
import math
import safire

__author__ = 'Jan Hajic jr.'


class ImageBrowser(object):
    """
    Provides functionality for looking directly at the images used to build the
    ImagenetCorpus.
    """

    def __init__(self, root, ids2files_map, icorp=None):
        """
        :type root: str
        :param root: Path to the directory relative to which files in the
            ids2files_map are given. Used in retrieving images. Best practice:
            use an absolute path.

        :type ids2files_map: file
        :param ids2files_map: A file-like object from which the ID-to-file
            mapping will be read. Expected format is two tab-separated columns,
            with the IDs in the first column and files in the second column.

        :type icorp: safire.data.imagenet_corpus.ImagenetCorpus
        :param icorp: A corpus that maps image numbers to image IDs. Useful
            if you wish to index the ImageBrowser directly by similarity query
            results, provided that the image numbering in ``icorp`` and in the
            index is consistent.

        """
        self.root = root
        self.ids2files, self.files2ids = self.parse_ids2files(ids2files_map)
        self.icorp = icorp

    def parse_ids2files(self, ids2files_map):
        """Parses the map of IDs to files.

        :type ids2files_map: file
        :param ids2files_map: A file-like object from which the ID-to-file
            mapping will be read. Expected format is two tab-separated columns,
            with the IDs in the first column and files in the second column.
        """

        ids2files = {}
        files2ids = {}

        for line in ids2files_map:
            ID, file = line.strip().split()
            ids2files[ID] = file
            files2ids[file] = ID

        return ids2files, files2ids


    def __getitem__(self, id_or_filename):
        """Gets the image file name, incl. path to root, if an ID is given.
        If a filename is given (relative to the root), returns the corresponding
        ID. Can also parse a filename given together with the root.

        If the ``icorp`` parameter was provided at initialization, can also
        convert integer keys to image files.

        Can parse an iterable of IDs/filenames; returns the corrseponding list
        (may be a mix of IDs and filenames, but that is not recommended).

        :raises: KeyError
        """
        if isinstance(id_or_filename, list) or isinstance(id_or_filename, tuple):
            return [self[item] for item in id_or_filename]
        else:
            if isinstance(id_or_filename, int):
                if self.icorp is not None:
                    iid = self.icorp.id2doc[id_or_filename]
                    return os.path.join(self.root, self.ids2files[iid])
            if id_or_filename in self.ids2files:
                return os.path.join(self.root, self.ids2files[id_or_filename])
            elif id_or_filename in self.files2ids:
                return self.files2ids[id_or_filename]
            elif id_or_filename.startswith(self.root):
                short_filename = id_or_filename[len(self.root):]
                if short_filename in self.files2ids:
                    return self.files2ids[short_filename]

            raise KeyError('Image with ID or filename %s not found.' % id_or_filename)

    def show(self, iid):
        """Given an image ID, opens the image.

        :param iid: An image ID from the source list.
        """
        im = Image.open(self[iid])
        im.show()

    def show_multiple(self, iids, width=1000, height=600):
        """Given a list of image IDs, creates a merged image from all
        of them.

        :param iid: An image ID from the source list.

        :param width: The total width of the output image.

        :param height: The total height of the output image.
        """
        image = self.build_tiled_image(iids, width, height)
        image.show()

    def build_tiled_image(self, iids,
                          similiarities=None, with_order=False,
                          width=1000.0, height=600.0, margin=10.0):
        """Given a list of image IDs, creates a merged image from all
        of them.

        :param iids: A list of image ID from the source list.

        :param similarities: A list that gives for each image a similarity
            to a query "image".

        :param with_order: A flag. If set, will draw in the upper-left corner
            the order of each image.

        :param width: The total width of the output image.

        :param height: The total height of the output image.

        :param margin: The margin of each sub-image.
        """
        n_images = len(iids)
        n_cols, is_sqrt_equal = safire.utils.isqrt(n_images)
        if not is_sqrt_equal:
            n_cols += 1

        n_rows = n_images / n_cols
        if n_rows * n_cols < n_images:
            n_rows += 1

        # Window = region of one sub-image
        w_width = math.floor(width / n_cols)
        w_height = math.floor(height / n_rows)

        # Img = the sub-image itself
        img_width = w_width - 2.0 * margin
        img_height = w_height - 2.0 * margin

        # print 'n_cols:', n_cols
        # print 'n_rows:',n_rows
        # print 'w_width:', w_width
        # print 'w_height:', w_height
        # print 'i_width:', img_width
        # print 'i_height:', img_height
        thumbnail_size = (int(img_width), int(img_height))

        images = self.load_images(iids, similiarities, with_order)

        # New image
        output_size = (int(width), int(height))
        output_image = Image.new('RGB', output_size, color='white')

        img_index = 0
        for start_x in xrange(0, int(height - w_height + 1), int(w_height)):
            for start_y in xrange(0, int(width - w_width + 1), int(w_width)):
                # print 'Starting X:', start_x
                # print 'Starting Y:', start_y
                start_position = (int(start_y + margin),
                                    int(start_x + margin))
                # print 'Start position:', start_position

                if img_index >= len(images):
                    break

                image = images[img_index]

                # print 'Image size:', image.size

                image.thumbnail(thumbnail_size, Image.ANTIALIAS)

                box = (start_position[0], start_position[1],
                       start_position[0] + image.size[0],
                       start_position[1] + image.size[1])

                # print 'Image thumbnail size:', image.size

                output_image.paste(image, box)

                img_index += 1

        return output_image

    def load_image(self, iid):
        """Loads the image with the given image ID."""
        return Image.open(self[iid])

    def load_images(self, iids, similarities=None, with_order=False):
        """
        :param iids: A list of image ID from the source list.

        :param similarities: A list that gives for each image a similarity
            to a query "image".

        :param with_order: A flag. If set, will draw in the upper-left corner
            the order of each image. [NOT IMPLEMENTED]

        :returns: A list of images.
        """
        images = [Image.open(self[iid]) for iid in iids]

        # Add similarities
        if similarities:

            caption_height = 40
            caption_font = ImageFont.truetype('DejaVuSans.ttf', 25)
            caption_color = (190, 190, 190)
            caption_offset = (10,5)

            if len(similarities) != len(images):
                raise ValueError('Inconsistent number of image IDs and similarities (%d vs. %d)!' % (
                    len(similarities), len(images)))

            captioned_images = []
            for img, similarity in zip(images, similarities):
                new_img = Image.new('RGB',
                                    (img.size[0], img.size[1]+caption_height),
                                    (30, 30, 30))
                new_img.paste(img,(0,0))
                sim_text = str(similarity)[:9]
                sim_text_position = (caption_offset[0],
                                     img.size[1] + caption_offset[1])
                draw = ImageDraw.Draw(new_img)
                draw.text(sim_text_position, sim_text, caption_color,
                          font=caption_font)

                # print 'Sim. text:', sim_text
                # print 'Sim. text position:', sim_text_position
                # print 'New image size:', new_img.size

#                new_img.show()

                captioned_images.append(new_img)

            images = captioned_images


        return images