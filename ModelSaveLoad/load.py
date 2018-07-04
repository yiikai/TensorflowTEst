import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('.\model\model-900.meta')
    saver.restore(sess,'./model/model-900')
    print(sess.graph)
    graph = tf.get_default_graph()
    print(graph)
    input_x = graph.get_operation_by_name('input_x').outputs[0]
    out = graph.get_operation_by_name('out').outputs[0]
    final = sess.run(out,feed_dict={input_x:[9]})
    print("input 9 , final is :",final)