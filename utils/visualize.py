def run(self, log_num=None):
    tensors = (self.model.inputs + 
               self.model.targets + 
               self.model.sample_weights)
    assert len(tensors) == len(self.val_data)

    feed_dict = dict(zip(tensors, self.val_data))
    summary = self.sess.run([self.merged], feed_dict=feed_dict)

    log_num = log_num or self.log_num
    self.writer.add_summary(summary[0], log_num)
    self.log_num += 1

    if self.verbose:
        print("MyTensorBoard saved %s logs" % len(summary))

def _init_logger(self):
    for layer in self.model.layers:
        if any([(spec_name in layer.name) for spec_name in self.layer_names]):
            grads = self._get_grads(layer)
            if grads is not None:
                tf.summary.histogram(layer.name + '_grad', grads)
            if hasattr(layer, 'output'):
                self._log_outputs(layer)

            for weight in layer.weights:
                mapped_weight_name = weight.name.replace(':', '_')
                tf.summary.histogram(mapped_weight_name, weight)

                w_img = self._to_img_format(weight)
                if w_img is not None:
                    tf.summary.image(mapped_weight_name, w_img)
    self.merged = tf.summary.merge_all()
    self._init_writer()
    print("MyTensorBoard initialized")


def _init_writer(self):
    tb_num = 0
    while any([('TB_' + str(tb_num) in fname) for fname in 
               os.listdir(self.base_logdir)]):
        tb_num += 1
    self.logdir = os.path.join(self.base_logdir, 'TB_%s' % tb_num)
    os.mkdir(self.logdir)
    print("New TB logdir created at %s" % self.logdir)

    if self.write_graph:
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
    else:
        self.writer = tf.summary.FileWriter(self.logdir)

def _get_grads(self, layer):
    for weight_tensor in layer.trainable_weights:
        grads = self.model.optimizer.get_gradients(
                     self.model.total_loss, weight_tensor)

        is_indexed_slices = lambda g: type(g).__name__ == 'IndexedSlices'
        return [grad.values if is_indexed_slices(grad) 
                            else grad for grad in grads]

def _log_outputs(self, layer):
    if isinstance(layer.output, list):
        for i, output in enumerate(layer.output):
            tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
    else:
        tf.summary.histogram('{}_out'.format(layer.name), layer.output)