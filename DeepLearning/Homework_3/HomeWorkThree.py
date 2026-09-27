#import libs
import tensorflow as tf


#define model class
class DenseModel(tf.Module):
    def __init__(self, activation = tf.nn.relu, output_size = 5):
        #intialize model setting
        super(DenseModel, self).__init__()
        self.output_size = output_size
        self.activation = activation
        self.is_built = False


    def build(self, input_shape):
        #define the model architecture for Layer One
        self.W1 = tf.Variable(tf.random.normal([input_shape[-1], 128]), name='W1')
        self.b1 = tf.Variable(tf.zeros([128]), name='b1')

        #define the model architecture for Layer Two
        self.W2 = tf.Variable(tf.random.normal([128, 64]), name='W2')
        self.b2 = tf.Variable(tf.zeros([64]), name='b2')

        #define the model architecture for Layer Three
        self.W3 = tf.Variable(tf.random.normal([64, self.output_size]), name='W3')
        self.b3 = tf.Variable(tf.zeros([self.output_size]), name='b3')

        #declare model is built
        self.is_built = True

    @tf.function
    def __call__(self, x):
        #check if model is built
        if not self.is_built:
            self.build(x.shape)

        #forward pass through the model
        z1 = tf.add(tf.matmul(x, self.W1), self.b1)
        a1 = self.activation(z1)

        z2 = tf.add(tf.matmul(a1, self.W2), self.b2)
        a2 = self.activation(z2)

        z3 = tf.add(tf.matmul(a2, self.W3), self.b3)
        return z3


# Use 10 outputs 
model = DenseModel(output_size=10)

# Specify any batch size, with 784 features per example.
model.__call__.get_concrete_function(
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32)
)

# Example input for checking model saving and loading.
x = tf.ones([2, 784])
original_output = model(x)
print("Output shape:", original_output.shape)  

tf.saved_model.save(model, "./DeepLearning/Homework_3/saved_model")

loaded_model = tf.saved_model.load("./DeepLearning/Homework_3/saved_model")
loaded_output = loaded_model(x)

tf.debugging.assert_near(original_output, loaded_output)
print("Saving and loading preserved the model's output.")


