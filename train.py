from data_exmp import *
import numpy as np
import model
import time
from configs import *
from os.path import join

train_list = [TFRECORD_PATH + '/train.tfrecord']
test_list = [TFRECORD_PATH + '/test.tfrecord']

def main():

    train_set = tf.data.TFRecordDataset(train_list)
    train_set = train_set.map(train_parse_exmp).repeat().batch(BATCH_SIZE).shuffle(buffer_size=300)

    test_set = tf.data.TFRecordDataset(test_list)
    test_set = test_set.map(test_parse_exmp).repeat().batch(BATCH_SIZE).shuffle(buffer_size=200)

    train_iterator = train_set.make_initializable_iterator()
    test_iterator = test_set.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_set.output_types, train_set.output_shapes)
    low_res_batch, high_res_batch = iterator.get_next()

    logits = model.inference(low_res_batch)
    training_loss = model.loss(logits, high_res_batch, name='training_loss', weights_decay=0)
    testing_loss = model.loss(logits, high_res_batch, name='testing_loss')

    global_step = tf.Variable(0, trainable=False, name='global_step')

    learning_rate = tf.train.inverse_time_decay(0.001, global_step, 10000, 2)
    tf.summary.scalar('learning_rate', learning_rate)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_loss, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(test_iterator.initializer)
    sess.run(train_iterator.initializer)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=MAX_CKPT_TO_KEEP)
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    train_summary_writer = tf.summary.FileWriter(TRAINING_SUMMARY_PATH + '/logs/train', graph=tf.get_default_graph())
    test_summary_writer = tf.summary.FileWriter(TRAINING_SUMMARY_PATH + '/logs/test', graph=tf.get_default_graph())

    merged_summary_op = tf.summary.merge_all()

    for step in range(1, NUM_TRAINING_STEPS+1):
        start_time = time.time()


        feed_dict = {handle:train_handle}

        _, batch_loss, train_summary, n = sess.run([train_step, training_loss, merged_summary_op, global_step], feed_dict=feed_dict)
        train_summary_writer.add_summary(train_summary, n)

        duration = time.time() - start_time
        assert not np.isnan(batch_loss), 'Model diverged with loss = NaN'

        # show train status
        if step % 10 == 0: 
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = 'step %d, batch_loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (step, batch_loss, examples_per_sec, sec_per_batch))

        # run test dateset and show result
        if step % 1000 == 0:
            feed_dict = {handle:test_handle}
            batch_loss, test_summary, n = sess.run([testing_loss, merged_summary_op, global_step], feed_dict=feed_dict)
            test_summary_writer.add_summary(test_summary, n)
            print('step %d, test loss = %.3f' % (step, batch_loss))

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == NUM_TRAINING_STEPS:
            saver.save(sess, join(CHECKPOINTS_PATH, 'model.ckpt'), global_step=step)

if __name__ == '__main__':
    main()
