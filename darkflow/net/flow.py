import os
import time
import math
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()


def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None
    profile = list()
    batches = self.framework.shuffle()
    loss_op = self.framework.loss
    batch_per_epoch = self.framework.batch_per_epoch()
    
    loss = 0.0
    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key]
            for key in loss_ph}
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss += fetched[1]

        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        args = [step_now, profile]
        
        ckpt = (i + 1) % batch_per_epoch
        if not ckpt:
            train_loss = loss / batch_per_epoch
        
            validation = self.framework.validation()
            val_loss = compute_validation_loss(self, validation, loss_ph, loss_op)
            
            form = 'step {} - train loss {} - val loss {}'
            self.say(form.format(step_now, train_loss, val_loss))
            profile += [(step_now, train_loss, val_loss)]
            loss = 0.0
            
            _save_ckpt(self, *args)
        else:
            form = 'step {}'
            self.say(form.format(step_now))       
    
    np.savetxt("losses.txt", np.array(profile), delimiter=',', header='step, train_loss, val_loss')
    if ckpt: _save_ckpt(self, *args)

def compute_validation_loss(self, validation, loss_ph, loss_op):
    val_loss = 0.0
    for i, (x_batch, datum) in enumerate(validation):
        val_dict = {
            loss_ph[key]: datum[key]
            for key in loss_ph}
        val_dict[self.inp] = x_batch
        val_dict.update(self.feed)
        
        loss_val = self.sess.run(loss_op, val_dict)
        val_loss = (i * val_loss + loss_val) / (i + 1)
        
    return val_loss

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
        'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp: this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo


def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time();
        last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
                            self.framework.postprocess(
                                prediction, os.path.join(inp_path, this_batch[i])))(*p),
                 enumerate(out))
        stop = time.time();
        last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
