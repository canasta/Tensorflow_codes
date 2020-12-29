# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/

*** CHANGED: Input method is changed. Image file to screen capture.
Original code:
https://github.com/tensorflow/models/blob/d7ce21fa4d3b8b204530873ade75637e1313b760/tutorials/image/imagenet/classify_image.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QBoxLayout, QGraphicsOpacityEffect, QSizePolicy, QFrame, QPlainTextEdit
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QBrush, QRegion

import io
from PIL import ImageGrab
from threading import Thread

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
model_dir = './model'
num_top_predictions = 4
# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self):
        label_lookup_path = os.path.join(
            model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        uid_lookup_path = os.path.join(
            model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
            label_lookup_path: string UID to integer node ID.
            uid_lookup_path: string UID to human-readable string.
        Returns:
            dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
            self.model_download_and_extract(model_dir)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)
            self.model_download_and_extract(model_dir)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                    tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name
    
    def model_download_and_extract(self, dest_directory):
        """Download and extract model tar file."""
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
        model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

class ImageProcessClass(QThread):
    resstr = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main = parent

    def run(self):
        fgeo = self.main.frameGeometry()
        geo = self.main.target_rect.geometry()
        capbox = (
            fgeo.left() + geo.left(),
            fgeo.top() + geo.top(),
            fgeo.left() + geo.left() + geo.width(),
            fgeo.top() + geo.top() + geo.height()
        )
        image = ImageGrab.grab(capbox)

        with io.BytesIO() as imagestr:
            image.save(imagestr, format="JPEG")
            
            self.resstr.emit(
                self.run_inference_on_image(imagedata=imagestr.getvalue())
            )

    def run_inference_on_image(self, imagedata):
        """Runs inference on an image.
        Args:
            imagedata: Image Data
        Returns:
            Nothing
        """            
        # Creates graph from saved GraphDef.
        create_graph()

        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor,
                                {'DecodeJpeg/contents:0': imagedata})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup()

            resstr = ''
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                resstr += '%s (score = %.5f)\n' % (human_string, score)
            return resstr

class Form(QWidget):
    """GUI 부분"""
    def __init__(self):
        QWidget.__init__(self, flags=Qt.Widget)

        self.setGeometry(300, 300, 400, 400)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.target_rect = QWidget(parent=self, flags=Qt.Widget)
        self.target_rect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.target_rect.setStyleSheet("background-color: black")
        layout.addWidget(self.target_rect)

        buttons = QWidget()
        buttons.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(buttons)
        layout2 = QBoxLayout(QBoxLayout.LeftToRight)
        buttons.setLayout(layout2)

        self.btn_capture = QPushButton(parent=self, text='분석')
        self.btn_exit = QPushButton(parent=self, text='종료')
        self.btn_capture.clicked.connect(self.on_btn_capture_clicked)
        self.btn_exit.clicked.connect(sys.exit)
        layout2.addWidget(self.btn_capture)
        layout2.addWidget(self.btn_exit)

        self.result_text = QPlainTextEdit()
        self.result_text.setMaximumHeight(100)
        self.result_text.setEnabled(False)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

        self.image_process_thread = ImageProcessClass(self)
        self.image_process_thread.resstr.connect(self.thread_end)

    def update(self):
        windowrect = self.frameGeometry()
        geo = self.target_rect.geometry()
        geo.moveTopLeft(self.target_rect.mapToGlobal(QPoint(0, 0)))
        
        left = windowrect.left() - geo.left()
        top = windowrect.top() - geo.top()
        right = windowrect.right() - geo.right()
        bottom = windowrect.bottom() - geo.bottom()

        windowrect.moveTopLeft(QPoint(0, 0))
        geo.moveTopLeft(
            QPoint(
                self.target_rect.geometry().left(),
                self.target_rect.geometry().top()
            )
        )

        region = QRegion(windowrect.adjusted(left, top, right, bottom))
        region -= QRegion(geo)

        self.setMask(region)

    def resizeEvent(self, event):
        super(Form, self).resizeEvent(event)
        self.update()

    def on_btn_capture_clicked(self):
        if self.btn_capture.isEnabled() and not self.image_process_thread.isRunning():
            self.result_text.clear()
            self.btn_capture.setText('분석 중')
            self.btn_capture.setEnabled(False)

            self.image_process_thread.start()
    
    @pyqtSlot(str)
    def thread_end(self, result):
        self.result_text.setPlainText(result)
        self.btn_capture.setText('분석')
        self.btn_capture.setEnabled(True)

def main(_):
    app = QApplication([])
    form = Form()
    form.show()
    exit(app.exec_())

if __name__ == '__main__':
    tf.app.run(main=main)
