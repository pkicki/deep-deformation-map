#!/usr/bin/env python2.7

from __future__ import print_function

import numpy as np
from scipy import misc
import caffe
import tempfile
from math import ceil


class Flow:

    def __init__(self, args):

        self.args = args
        self.num_blobs = 2
        self.input_data = []
        self.vars = dict()
        self.width = 0
        self.height = 0

        # create temporary container for net weights
        self.tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

        # container for caffe model
        self.net = []

        self.input_dict = {}
        print("End init")

    def setup_images(self, img0, img1):

        if len(img0.shape) < 3:
            self.input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:
            self.input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

        if len(img1.shape) < 3:
            self.input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:
            self.input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

        self.width = self.input_data[0].shape[3]
        self.height = self.input_data[0].shape[2]

        self.vars['TARGET_WIDTH'] = self.width
        self.vars['TARGET_HEIGHT'] = self.height

        divisor = 64.
        self.vars['ADAPTED_WIDTH'] = int(ceil(self.width/divisor) * divisor)
        self.vars['ADAPTED_HEIGHT'] = int(ceil(self.height/divisor) * divisor)

        self.vars['SCALE_WIDTH'] = self.width / float(self.vars['ADAPTED_WIDTH'])
        self.vars['SCALE_HEIGHT'] = self.height / float(self.vars['ADAPTED_HEIGHT'])

        print("Setup images")

    def read_net(self):
        # read net and store it in copy tmp?
        proto = open(self.args.deployproto).readlines()
        for line in proto:
            for key, value in self.vars.items():
                tag = "$%s$" % key
                line = line.replace(tag, str(value))
            self.tmp.write(line)
        self.tmp.flush()

        print("Read net")

    def setup_model(self):

        # setup caffe model
        print("Setup caffe model")
        if not self.args.verbose:
            caffe.set_logging_disabled()

        print("Set GPU id: %d" % self.args.gpu)
        caffe.set_device(self.args.gpu)

        print("Set GPU mode")
        caffe.set_mode_gpu()

        print("Create caffe net: %s" % self.tmp.name)
        self.net = caffe.Net(self.tmp.name, self.args.caffemodel, caffe.TEST)

        print("Setup model")

    def setup_feed_dict(self):
        # setup feed dict for cafe model
        for blob_idx in range(self.num_blobs):
            self.input_dict[self.net.inputs[blob_idx]] = self.input_data[blob_idx]

        print("Setup feed dict")

    def __call__(self, img0, img1):
        # setup current images
        self.setup_images(img0=img0, img1=img1)
        self.read_net()
        self.setup_model()
        self.setup_feed_dict()

        # obtain results
        print('Network forward pass using %s.' % self.args.caffemodel)
        i = 1
        while i <= 5:
            i += 1

            self.net.forward(**self.input_dict)

            containsNaN = False
            for name in self.net.blobs:
                blob = self.net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
                print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')

        # obtain result
        blob = np.squeeze(self.net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        return blob


# def writeFlow(name, flow):
#     f = open(name, 'wb')
#     f.write('PIEH'.encode('utf-8'))
#     np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
#     flow = flow.astype(np.float32)
#     flow.tofile(f)
#     f.flush()
#     f.close()
#
#
# blob = Flow.run()
# writeFlow(args.out, blob)
