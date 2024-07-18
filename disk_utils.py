#https://gist.github.com/Aryansh-S/c0991dd28bd525ecd02adaa4b0b6456f
import tensorflow as tf
import numpy as np

# works for tensors of arbitrary types and any level of nesting!
# useful to save the data in any organized containers that aren't easily serializable but can be made into lists of tensors 
# (e.g., nested dictionaries)

#LENGTH = 0 # will be set after tensor list is written (below)

def write_tensor_list(tensor_list, filename):
    type_tensor = tf.constant([str(tensor.dtype)[9:-2] for tensor in tensor_list]) # figures out the dtypes by itself
    serialized_tensor = tf.io.serialize_tensor(type_tensor)
    with tf.io.TFRecordWriter(filename) as writer:
        writer.write(serialized_tensor.numpy())
        for tensor in tensor_list:
            if hasattr(tensor, "to_tensor"):
                tensor = tensor.to_tensor(default_value=-2)
            serialized_tensor = tf.io.serialize_tensor(tensor)
            writer.write(serialized_tensor.numpy())
    #LENGTH = len(tensor_list)

def read_tensor_list(filename, LENGTH): # can also be a list of filenames (where LENGTH is the constant number of tensors per file)
    ret = []
    records = tf.data.TFRecordDataset(filename)
    type_list = []
    for i, record in enumerate(records):
        if i % (LENGTH + 1) == 0:
            type_list = []
            type_list_tensor = tf.io.parse_tensor(record, tf.string)
            type_list = tf.Variable(type_list_tensor).numpy().tolist()
            type_list = [tf.dtypes.as_dtype(tp.decode()) for tp in type_list]
            type_list.insert(0, tf.string) # 0th entry is always just the types
        else:
            ret.append(tf.io.parse_tensor(record, type_list[i % (LENGTH + 1)]))
    return ret