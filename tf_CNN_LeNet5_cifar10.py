LENET5_LIKE_BATCH_SIZE = 32
LENET5_LIKE_FILTER_SIZE = 5
LENET5_LIKE_FILTER_DEPTH = 16
LENET5_LIKE_NUM_HIDDEN = 120

def variables_lenet5_like(filter_size = LENET5_LIKE_FILTER_SIZE, 
                          filter_depth = LENET5_LIKE_FILTER_DEPTH, 
                          num_hidden = LENET5_LIKE_NUM_HIDDEN,
                          image_width = 32, image_height = 32, image_depth = 3, num_labels = 10):
    
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth]))

    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth, filter_depth], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth]))
   
    w3 = tf.Variable(tf.truncated_normal([(image_width // 4)*(image_height // 4)*filter_depth , num_hidden], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))

    w4 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
    
    w5 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return variables

def model_lenet5_like(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)
    layer3_drop = tf.nn.dropout(layer3_actv, 0.5)
    
    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
    layer4_drop = tf.nn.dropout(layer4_actv, 0.5)
    
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits


#Variables used in the constructing and running the graph
num_steps = 10001
display_step = 1000
learning_rate = 0.001
batch_size = 16

#定义数据的基本信息，传入变量
image_width = 32
image_height = 32
image_depth = 3
num_labels = 10





cifar10_folder = './data/cifar10/'
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
c10_image_size = 32 #Ahmet Taspinar的代码缺少了这一语句

with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes')

c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size, c10_image_size, c10_image_depth)

c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
 
        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_

c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size, c10_image_size, c10_image_depth)
del c10_train_dataset
del c10_train_labels

print("训练集包含以下标签: {}".format(np.unique(c10_train_dict[b'labels'])))
print('训练集维度', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('测试集维度', test_dataset_cifar10.shape, test_labels_cifar10.shape)


test_dataset = test_dataset_cifar10
test_labels = test_labels_cifar10
train_dataset = train_dataset_cifar10
train_labels = train_labels_cifar10




graph = tf.Graph()
with graph.as_default():
    #1 首先使用占位符定义数据变量的维度
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)

    #2 然后初始化权重矩阵和偏置向量
    variables = variables_lenet5_like(image_width = image_width, image_height=image_height, image_depth = image_depth, num_labels = num_labels)


    #3 使用模型计算分类
    logits = model_lenet5_like(tf_train_dataset, variables)

    #4 使用带softmax的交叉熵函数计算预测标签和真实标签之间的损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    #5  采用Adam优化算法优化上一步定义的损失函数，给定学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # 执行预测推断
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model_lenet5_like(tf_test_dataset, variables))


with tf.Session(graph=graph) as session:
    #初始化全部变量
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        #在每一次批量中，获取当前的训练数据，并传入feed_dict以馈送到占位符中
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        train_accuracy = accuracy(predictions, batch_labels)
        
        if step % display_step == 0:
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)